/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include "heterbart.h"

#include <limits>

int robart_heterbart_compiled_abi_version() noexcept
{
  return ROBART_HETERBART_ABI_VERSION;
}

std::size_t robart_heterbart_compiled_sizeof() noexcept
{
  return sizeof(heterbart);
}

static inline void inc_safe(std::vector<size_t>& v, size_t need, size_t j) {
   if (v.size() < need) v.assign(need, 0);
   if (j < v.size()) v[j] += 1;
 }
//--------------------------------------------------
void heterbart::pr()
{
   cout << "+++++heterbart object:\n";
   bart::pr();
}
//--------------------------------------------------
void heterbart::draw(double *sigma, rn& gen)
{
   if(formal_cache_.size()!=m)formal_cache_.assign(m,robart::formal::TreeCache{});
   if (track_last_rot_trace_) {
     if (last_rot_accept_.size()!=m) last_rot_accept_.assign(m,0);
     if (last_nbots_.size()!=m)      last_nbots_.assign(m,1);
     if (last_rot_logacc_.size()!=m) last_rot_logacc_.assign(m, 0.0);
     if (last_rot_ll_current_.size()!=m) last_rot_ll_current_.assign(m, NA_REAL);
     if (last_rot_ll_proposed_.size()!=m) last_rot_ll_proposed_.assign(m, NA_REAL);
     if (last_rot_global_ll_current_.size()!=m) last_rot_global_ll_current_.assign(m, NA_REAL);
     if (last_rot_global_ll_proposed_.size()!=m) last_rot_global_ll_proposed_.assign(m, NA_REAL);
     if (last_rot_memb_change_.size()!=m) last_rot_memb_change_.assign(m, NA_REAL);
     if (last_rot_leaf_n_L1_.size()!=m)   last_rot_leaf_n_L1_.assign(m, NA_REAL);
     if (last_rot_leaf_sy_L1_.size()!=m)  last_rot_leaf_sy_L1_.assign(m, NA_REAL);
   }

   const size_t iter_now = ++draw_iter_counter_; // 1-based
   const bool enable_small_rotation = (iter_now >= rot_mh_start_iter_);
   const bool do_rot_diag = (enable_small_rotation && rot_diag_every_ > 0 && (iter_now % rot_diag_every_ == 0));
   const bool do_rot_time = (enable_small_rotation && rot_timing_every_ > 0 && (iter_now % rot_timing_every_ == 0));

   // rotation timing summary (sums across trees for this draw)
   last_rot_timing_m_eff_ = 0;
   if (do_rot_time) last_rot_timing_ms_.assign(8, 0.0);
   else last_rot_timing_ms_.clear();

   const bool rotation_only_mode = (enable_small_rotation && rotation_only_after_warmup_);
   // After warmup, run tree updates multiple times per tree unless rotation-only mode is requested
   const size_t tree_update_reps = rotation_only_mode ? 0 : (enable_small_rotation ? tree_update_reps_after_warmup_ : 1);

   // Depth prior configuration (constant during this draw)
   const DepthPriorSpec dp = get_depth_prior_spec();



   for(size_t j=0;j<m;j++) {
     
      // check whether jth_dinfo is valid
      if (dinfo_vec[j] == nullptr){
        cout << "dinfo_vec[j] is not valid when j = " << j << std::endl << std::endl;
      }
     
      fit(t[j],xinfo_vec[j],p,n,ix_vec[j],ftemp);
      // 
      
      for(size_t k=0;k<n;k++) {
         allfit[k] = allfit[k]-ftemp[k];
         r[k] = y[k]-allfit[k];
      }

      // Test data are optional. Never index dinfo_test_vec unless the
      // tree-specific slot exists; operator[] on an empty vector is UB.
      dinfo* di_test_j =
        (j < dinfo_test_vec.size()) ? dinfo_test_vec[j] : nullptr;
      const size_t np_j = di_test_j ? di_test_j->n : 0;
      
      // Enable the centered rotation MH update at rot_mh_start_iter_.
      if (rot_state_.size() != m) rot_state_.assign(m, robart::RotState{});
      double rot_logacc = 0.0;
      double rot_ll_current = NA_REAL;
      double rot_ll_proposed = NA_REAL;
      double rot_global_ll_current = NA_REAL;
      double rot_global_ll_proposed = NA_REAL;
      double rot_memb_change = NA_REAL;
      double rot_leaf_n_L1 = NA_REAL;
      double rot_leaf_sy_L1 = NA_REAL;
      bool rot_acc = false;
      if (enable_small_rotation) {
        // Collect the split-variable support of the current tree.
        // The pair proposal chooses between:
        //   - inside-support refine move
        //   - one-in / one-out support-expanding move
        // These are two support-selection cases, not separate rotation kernels.
        std::vector<size_t> rot_cand_vars_j;
        robart::collect_split_vars(t[j], p, rot_cand_vars_j);
        const std::vector<size_t>* rot_cand_ptr = &rot_cand_vars_j;

        robart::RotTiming tm;
        robart::RotTiming* tm_ptr = do_rot_time ? &tm : nullptr;

          rot_acc = robart::heter_rotation_centered_update_tree(
           t[j], xinfo_vec[j], dinfo_vec[j], pi, sigma,
           this->rotation_step_size_, this->rotation_angle_bound_,
           this->rotation_num_planes_,
           this->rotation_retune_cutpoints_,
           gen, rot_state_[j], p, n, this->dynamic_xinfo_on_rotation_, this->nc_per_var_, this->rotation_xinfo_policy_,
           np_j, di_test_j, &rot_logacc, &rot_ll_current, &rot_ll_proposed,
           &rot_global_ll_current, &rot_global_ll_proposed,
           &rot_memb_change, &rot_leaf_n_L1, &rot_leaf_sy_L1,

           rot_cand_ptr,
           this->support_expand_prob_,
           do_rot_diag,
           this->rot_accumulate_angles_,
           tm_ptr, dp, &lpv, nullptr, &formal_cache_[j]
          );

        if (do_rot_time && tm_ptr) {
          last_rot_timing_ms_[0] += tm.t_seq_ms;
          last_rot_timing_ms_[1] += tm.t_xprop_ms;
          last_rot_timing_ms_[2] += tm.t_cut_fwd_ms;
          last_rot_timing_ms_[3] += tm.t_diag_ms;
          last_rot_timing_ms_[4] += tm.t_ll_ms;
          last_rot_timing_ms_[5] += tm.t_cut_bwd_ms;
          last_rot_timing_ms_[6] += tm.t_accept_ms;
          last_rot_timing_ms_[7] += tm.t_total_ms;
          last_rot_timing_m_eff_ += 1;
        }
        inc_safe(rot_proposed_, m, j);
        if (rot_acc) inc_safe(rot_accepted_, m, j);
      }

      if (track_last_rot_trace_) {
        // Record whether this tree's proposed rotation was accepted.
        last_rot_accept_[j] = rot_acc ? 1 : 0;
        // Record the proposed change in log marginal likelihood.
        last_rot_logacc_[j] = rot_logacc;
        last_rot_ll_current_[j]  = enable_small_rotation ? rot_ll_current : NA_REAL;
        last_rot_ll_proposed_[j] = enable_small_rotation ? rot_ll_proposed : NA_REAL;
        last_rot_global_ll_current_[j]  = enable_small_rotation ? rot_global_ll_current : NA_REAL;
        last_rot_global_ll_proposed_[j] = enable_small_rotation ? rot_global_ll_proposed : NA_REAL;
        if (enable_small_rotation) {
          last_rot_memb_change_[j] = rot_memb_change;
          last_rot_leaf_n_L1_[j]   = rot_leaf_n_L1;
          last_rot_leaf_sy_L1_[j]  = rot_leaf_sy_L1;
        } else {
          last_rot_memb_change_[j] = NA_REAL;
          last_rot_leaf_n_L1_[j]   = NA_REAL;
          last_rot_leaf_sy_L1_[j]  = NA_REAL;
        }
      }
      ix_vec[j] = dinfo_vec[j]->x;
      if (j < dinfo_test_vec.size() && dinfo_test_vec[j]) {
         ix_test_vec[j] = dinfo_test_vec[j]->x;
      }
      // Do multiple structure updates per tree within this iteration
      for (size_t rep=0; rep<tree_update_reps; ++rep) {
        heterbd(t[j],xinfo_vec[j],dinfo_vec[j],pi,sigma,nv,pv,aug,gen,dp,&lpv,&formal_cache_[j]);
      }
      heterdrmu(t[j],xinfo_vec[j],dinfo_vec[j],pi,sigma,gen);
      if (track_last_rot_trace_) {
        // Record this tree's terminal-node count.
        last_nbots_[j] = t[j].nbots();
      }

      fit(t[j],xinfo_vec[j],p,n,ix_vec[j],ftemp);
      for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
   }
   if(dartOn && !rotation_only_mode) update_coordinate_probabilities(gen);

   // Optional: learn depth-prior hyperparameters (tilt or nu)
   if (!rotation_only_mode) update_depth_prior_hypers(gen);

}
