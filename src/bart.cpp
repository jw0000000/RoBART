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

#include "bart.h"
#include "rotation_mh.h"
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>

//--------------------------------------------------
//constructor
bart::bart():m(200),t(m),pi(),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),di(),dartOn(false),xinfo_vec(m),ix_vec(m),ix_test_vec(m) {
  last_rot_accept_.assign(m, 0);
  last_nbots_.assign(m, 1);
  last_rot_logacc_.assign(m, 0.0);
  last_rot_global_ll_current_.assign(m, NA_REAL);
  last_rot_global_ll_proposed_.assign(m, NA_REAL);
}
bart::bart(size_t im):m(im),t(m),pi(),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),di(),dartOn(false),xinfo_vec(m),ix_vec(m),ix_test_vec(m) {
  last_rot_accept_.assign(m, 0);
  last_nbots_.assign(m, 1);
  last_rot_logacc_.assign(m, 0.0);
  last_rot_global_ll_current_.assign(m, NA_REAL);
  last_rot_global_ll_proposed_.assign(m, NA_REAL);
}
bart::~bart()
{
   if(allfit) delete[] allfit;
   if(r) delete[] r;
   if(ftemp) delete[] ftemp;

   for (size_t i=0;i<dinfo_vec.size();++i){
      delete dinfo_vec[i];
   }
   for (size_t i=0;i<dinfo_test_vec.size();++i){
     delete dinfo_test_vec[i];
   }
   dinfo_vec.clear();
   dinfo_test_vec.clear();
   
   ix_vec.clear();
   ix_test_vec.clear();
   
   xinfo_vec.clear();

}

//--------------------------------------------------
//operators
// Native model instances own buffers and are deliberately noncopyable.
void bart::setxinfo(xinfo& _xi)
{
   formal_cache_.clear();
   size_t p=_xi.size();
   xi.resize(p);
   for(size_t i=0;i<p;i++) {
      if (verbose_) cout << "xi is created at here" << std::endl << std::endl;
      size_t nc=_xi[i].size();
      xi[i].resize(nc);
      for(size_t j=0;j<nc;j++) xi[i][j] = _xi[i][j];
   }
}
//--------------------------------------------------
void bart::setdata(size_t p, size_t n, double *x, double *y, size_t numcut)
{
  if (verbose_) cout << "bart::setdata mtd 1 is called" << std::endl;
  int* nc = new int[p];
  for(size_t i=0; i<p; ++i) nc[i]=numcut;
  this->setdata(p, n, x, y, nc);
  delete [] nc;
  if (verbose_) cout << "eof of bart::setdata mtd 1" << std::endl;
}

void bart::setdata(size_t p, size_t n, double *x, double *y, int *nc)
{
   formal_cache_.clear();
  if (verbose_) cout << "bart::setdata mtd 2 is called" << std::endl;
   this->p=p; this->n=n; this->x=x; this->y=y;
   nc_per_var_.assign(p, 0);
   for (size_t i = 0; i < p; ++i) nc_per_var_[i] = nc[i];
   if(xi.size()==0) {
      if (verbose_) cout << "xi.size() == 0" << std::endl;
      makexinfo(p,n,&x[0],xi,nc);
      if (verbose_) cout << "eof xi.size() == 0" << std::endl << std::endl;
   }

   // // predict mtd is called for fitting the allfit var.
   // predict(p,n,x,allfit); // xinfo_vec[j],ix_vec[j] are used at here

   if(r) delete[] r;
   r = new double[n]();

   if(ftemp) delete[] ftemp;
   ftemp = new double[n]();

   di.n=n; di.p=p; di.x = &x[0]; di.y=r;
   double* end_r = r+n;
   double* end_y = y+n;
   
   size_t break_i = 0;
   if (verbose_) {
     cout << "Start printing p-vals." << std::endl;
     for (double* pp = r; pp != end_r; ++pp){
       cout << "p: " << *pp << std::endl;
       ++break_i;
       if(break_i >= 5) break;
     }
     
     break_i = 0;
     cout << "Start printing y-vals." << std::endl;
     for (double* pp = di.y; pp != end_y; ++pp){
       cout << "y: " << *pp << std::endl;
       ++break_i;
       if(break_i >= 5) break;
     }
   }
   
   nv.assign(p,0);
   pv.assign(p,1./static_cast<double>(p));
   lpv.assign(p,-std::log(static_cast<double>(p)));
   if (verbose_) cout << "eof of bart::setdata mtd 2" << std::endl;
}

void bart::setallfit(size_t p, size_t n, double *x){
  if (verbose_) cout << "bart::setallfit mtd is called!" << std::endl;
  
  if(allfit) delete[] allfit;
  allfit = new double[n]();
  // predict mtd is called for fitting the allfit var.
  predict(p,n,x,allfit); // xinfo_vec[j],ix_vec[j] are used at here
  
}

//--------------------------------------------------
void bart::predict(size_t p, size_t n, double *x, double *fp, bool is_test)
//uses: m,t,xi
{
   // This method evaluates the already registered training/test design; it is
   // not a newdata API. Reject incompatible dimensions before mutating state.
   if (p == 0 || p != this->p)
     throw std::invalid_argument("bart::predict: predictor count differs from initialized design");
   if (n == 0) return; // No pointer dereference or pending-rotation flush.
   if (!fp || !x) throw std::invalid_argument("bart::predict: null data/output pointer");
   if (n != (is_test ? np_ : this->n))
     throw std::invalid_argument("bart::predict: row count changed; register a new model, not a resized cache");
   const size_t expected = robart_contract::product(p, n, "Prediction design");
   if (rot_state_.size() != m || xinfo_vec.size() != m || ix_vec.size() != m)
     throw std::runtime_error("bart::predict: per-tree training state is not initialized");
   if (is_test && (ix_test_vec.size() != m || dinfo_test_vec.size() != m))
     throw std::runtime_error("bart::predict: per-tree test state is not initialized");
   for (size_t j = 0; j < m; ++j) {
     if (xinfo_vec[j].size() != p)
       throw std::runtime_error("bart::predict: xinfo dimension mismatch");
     if (is_test) {
       if (!dinfo_test_vec[j] || !rot_state_[j].has_test ||
           rot_state_[j].Xte.size() != expected ||
           dinfo_test_vec[j]->p != p || dinfo_test_vec[j]->n != n ||
           ix_test_vec[j] != rot_state_[j].Xte.data())
         throw std::runtime_error("bart::predict: inconsistent test-buffer registration");
     } else if (rot_state_[j].Xtr.size() != expected || !ix_vec[j]) {
       throw std::runtime_error("bart::predict: inconsistent training-buffer registration");
     }
   }
   if (is_test) for (size_t j = 0; j < m; ++j) {
     robart::flush_pending_test_rotations(rot_state_[j], p, n);
     ix_test_vec[j] = rot_state_[j].Xte.data();
     dinfo_test_vec[j]->x = ix_test_vec[j];
   }

     // Sum the per-tree predictions.
     // Avoid repeated allocations: reuse a member buffer.
     if (fptemp_buf_.size() < n) fptemp_buf_.resize(n);
     double* fptemp = &(fptemp_buf_[0]);
     std::fill(fp, fp + n, 0.0);
     if (xinfo_vec.size() != m || ix_vec.size() != m) {
       throw std::runtime_error("bart::predict: per-tree training buffers are not initialized");
     }
     if (is_test && ix_test_vec.size() != m) {
       throw std::runtime_error("bart::predict: per-tree test buffers are not initialized");
     }
     for(size_t j=0;j<m;j++) {
       if (xinfo_vec[j].size() != p) {
         throw std::runtime_error("bart::predict: xinfo dimension mismatch");
       }
       if(is_test){
         if (ix_test_vec[j] == nullptr) {
           throw std::runtime_error("bart::predict: null test-design pointer");
         }
         // For test prediction, n equals np.
         fit(t[j], xinfo_vec[j], p, n, ix_test_vec[j], fptemp);
       }else{
         if (ix_vec[j] == nullptr) {
           throw std::runtime_error("bart::predict: null training-design pointer");
         }
         fit(t[j], xinfo_vec[j], p, n, ix_vec[j],      fptemp);
       }
       for(size_t k=0;k<n;k++) fp[k] += fptemp[k];
     }
}

static inline void inc_safe(std::vector<size_t>& v, size_t need, size_t j) {
  if (v.size() < need) v.assign(need, 0);
  if (j < v.size()) v[j] += 1;
}


// --------------------------------------------------
// Depth-prior hyperparameter update (optional)
//   - tilt (for exptilt prior):   tilt ~ Gamma(shape, rate), RW on log(tilt)
//   - nu (for nu^(d+1) prior):    u=2*nu ~ Beta(a,b), RW on logit(u)
//
// This step conditions on the current complete forest, frames, grids and
// coordinate probabilities, because
// the likelihood and leaf-parameter priors do not depend on these depth-hyper
// parameters.
// --------------------------------------------------

namespace {
  static inline double logistic(double x) {
    if (x >= 0.0) {
      const double z = std::exp(-x);
      return 1.0 / (1.0 + z);
    } else {
      const double z = std::exp(x);
      return z / (1.0 + z);
    }
  }


}

void bart::update_depth_prior_hypers(rn& gen)
{
  const bool do_tilt = (depth_prior_type_ == DEPTH_PRIOR_EXPTILT) && sample_depth_tilt_;
  const bool do_nu   = (depth_prior_type_ == DEPTH_PRIOR_NU)      && sample_depth_nu_;
  if (!do_tilt && !do_nu) return;

  // Evaluate the same complete prior as structural and rotation updates.
  auto log_full_treeprior = [&](const DepthPriorSpec& dp) -> double {
    return forest_logprior(dp,lpv,false);
  };

  // --------------------------------------------------
  // 2) MH step for tilt (log scale)
  // --------------------------------------------------
  if (do_tilt) {
    const double eps = 1e-12;
    double tilt_cur = depth_prior_tilt_;
    if (!(tilt_cur > 0.0)) tilt_cur = eps;
    double eta_cur = std::log(tilt_cur);

    // propose
    const double eta_prop = eta_cur + depth_tilt_rw_sd_ * gen.normal();
    const double tilt_prop = std::exp(eta_prop);

    // unnormalized log posterior in eta = log(tilt)
    //   log p(tilt) + log p(trees|tilt) + log|dtilt/deta|
    // where tilt ~ Gamma(shape, rate), and Jacobian contributes +eta.
    auto logpost_eta = [&](double eta) -> double {
      const double tilt = std::exp(eta);
      DepthPriorSpec dp(depth_prior_type_, tilt, depth_prior_nu_);
      const double ltree = log_full_treeprior(dp);
      if (!std::isfinite(ltree)) return -std::numeric_limits<double>::infinity();
      // Gamma(shape,rate): (shape-1)log tilt - rate*tilt  + const
      // + Jacobian log tilt => +eta
      return (depth_tilt_shape_) * eta - depth_tilt_rate_ * tilt + ltree;
    };

    const double lp_cur  = logpost_eta(eta_cur);
    const double lp_prop = logpost_eta(eta_prop);

    depth_tilt_prop_ += 1;
    const double logacc = lp_prop - lp_cur;
    if (std::log(gen.uniform()) < logacc) {
      depth_prior_tilt_ = tilt_prop;
      depth_tilt_acc_ += 1;
    }
  }

  // --------------------------------------------------
  // 3) MH step for nu via u=2*nu in (0,1) (logit scale)
  // --------------------------------------------------
  if (do_nu) {
    // current u
    double u_cur = 2.0 * depth_prior_nu_;
    // keep strictly inside (0,1)
    const double u_eps = 1e-12;
    if (u_cur <= u_eps) u_cur = u_eps;
    if (u_cur >= 1.0 - u_eps) u_cur = 1.0 - u_eps;

    // logit
    double eta_cur = std::log(u_cur) - std::log(1.0 - u_cur);

    // propose on logit scale
    const double eta_prop = eta_cur + depth_nu_rw_sd_ * gen.normal();
    const double u_prop = logistic(eta_prop);

    auto logpost_eta = [&](double eta) -> double {
      const double u = logistic(eta);
      const double nu_depth = 0.5 * u;
      DepthPriorSpec dp(depth_prior_type_, depth_prior_tilt_, nu_depth);
      const double ltree = log_full_treeprior(dp);
      if (!std::isfinite(ltree)) return -std::numeric_limits<double>::infinity();

      // u ~ Beta(a,b): (a-1)log u + (b-1)log(1-u) + const
      // Jacobian du/deta = u(1-u) contributes +log u + log(1-u)
      // => total: a log u + b log(1-u) + const
      return depth_nu_a_ * std::log(u) + depth_nu_b_ * std::log(1.0 - u) + ltree;
    };

    const double lp_cur  = logpost_eta(eta_cur);
    const double lp_prop = logpost_eta(eta_prop);

    depth_nu_prop_ += 1;
    const double logacc = lp_prop - lp_cur;
    if (std::log(gen.uniform()) < logacc) {
      depth_prior_nu_ = 0.5 * u_prop;
      depth_nu_acc_ += 1;
    }
  }
}

double bart::state_logprior_mu()
{
   if (!(pi.tau > 0.0) || !std::isfinite(pi.tau)) {
      return -std::numeric_limits<double>::infinity();
   }

   const double tau2 = pi.tau * pi.tau;
   const double log_const = -0.5 * std::log(2.0 * 3.141592653589793238462643383279502884 * tau2);
   double lp = 0.0;

   for (size_t j = 0; j < m; ++j) {
      tree::npv bots;
      t[j].getbots(bots);
      for (size_t bidx = 0; bidx < bots.size(); ++bidx) {
         tree::tree_p nd = bots[bidx];
         if (!nd) continue;
         const double mu = nd->gettheta();
         if (!std::isfinite(mu)) return std::numeric_limits<double>::quiet_NaN();
         lp += log_const - 0.5 * (mu * mu) / tau2;
      }
   }

   return lp;
}

double bart::forest_logprior(const DepthPriorSpec& dp,const std::vector<double>& logs,bool terminal_only)
{
   if(formal_cache_.size()!=m)formal_cache_.assign(m,robart::formal::TreeCache{});
   double total=0.;
   for(size_t j=0;j<m;++j) {
     const robart::formal::Context ctx=formal_cache_[j].context(xinfo_vec[j],dinfo_vec[j]->x,p,n,&logs);
     const robart::formal::PriorTerms lp=formal_cache_[j].get(t[j],ctx,pi,dp);
     if(!lp.valid)return robart::formal::neginf();
     total+=terminal_only?lp.terminal:lp.total();
   }
   return total;
}

double bart::state_logprior_tree()
{
   return forest_logprior(get_depth_prior_spec(),lpv,false);
}

void bart::update_coordinate_probabilities(rn& gen)
{
   // Preserve the Dirichlet concentration model, but its old conjugate draw is
   // now an independence proposal. Internal-node s_v factors and the symmetric
   // Dirichlet prior cancel; terminal factors must still enter the MH ratio.
   // Counts are actual internal labels, never virtual failed-birth counts.
   std::fill(nv.begin(),nv.end(),0);
   for(size_t j=0;j<m;++j) {
     tree::npv nodes;t[j].getnodes(nodes);
     for(tree* node:nodes)if(node->getl())++nv[node->getv()];
   }
   std::vector<double> proposal;
   draw_s(nv,proposal,theta,gen);
   proposal=robart::formal::normalized_log_weights(p,&proposal);
   const DepthPriorSpec dp=get_depth_prior_spec();
   const double current=forest_logprior(dp,lpv,true);
   const double proposed=forest_logprior(dp,proposal,true);
   if(!std::isfinite(current))throw std::runtime_error("Invalid current DART full-prior factor.");
   if(std::isfinite(proposed) && std::log(gen.uniform())<proposed-current)lpv=proposal;
   draw_theta0(const_theta,theta,lpv,a,b,rho,gen);
   for(size_t j=0;j<p;++j)pv[j]=std::exp(lpv[j]);
}

//--------------------------------------------------
// start from the below mtd
// only for the training set
void bart::draw(double sigma, rn& gen)
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
      if (j == 0 && (rot_proposed_.size() != m || rot_accepted_.size() != m)) {
        reset_rotation_counters();
      }
      
      fit(t[j],xinfo_vec[j],p,n,ix_vec[j],ftemp);
      for(size_t k=0;k<n;k++) {
         allfit[k] = allfit[k]-ftemp[k];
         r[k] = y[k]-allfit[k];
      }
      
      // Read the number of test observations from this tree's data object.
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
          rot_acc = robart::rotation_centered_update_tree(
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
          do_rot_diag, this->rot_accumulate_angles_, tm_ptr, dp, &lpv, nullptr, &formal_cache_[j]
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
        // Record whether this tree's proposed rotation was accepted (0 or 1).
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
        bd(t[j],xinfo_vec[j],dinfo_vec[j],pi,sigma,nv,pv,aug,gen,dp,&lpv,&formal_cache_[j]);
      }
      drmu(t[j],xinfo_vec[j],dinfo_vec[j],pi,sigma,gen);
      if (track_last_rot_trace_) {
        // Record the number of terminal nodes after the tree update.
        last_nbots_[j] = t[j].nbots();
      }

      fit(t[j],xinfo_vec[j],p,n,ix_vec[j],ftemp);
      for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
   }
   if(dartOn && !rotation_only_mode) update_coordinate_probabilities(gen);

   // Optional: learn depth-prior hyperparameters (tilt or nu)
   if (!rotation_only_mode) update_depth_prior_hypers(gen);
}
//--------------------------------------------------
//public functions
void bart::pr() //print to screen
{
   cout << "*****bart object:\n";
   cout << "m: " << m << std::endl;
   cout << "t[0]:\n " << t[0] << std::endl;
   cout << "t[m-1]:\n " << t[m-1] << std::endl;
   cout << "prior and mcmc info:\n";
   pi.pr();
   if(dart){
     cout << "*****dart prior (On):\n";
     cout << "a: " << a << std::endl;
     cout << "b: " << b << std::endl;
     cout << "rho: " << rho << std::endl;
     cout << "augmentation: " << aug << std::endl;
   }
   else cout << "*****dart prior (Off):\n";
   if(p) cout << "data set: n,p: " << n << ", " << p << std::endl;
   else cout << "data not set\n";
}

//--------------------------------------------------
// Assign random rotation to each tree
// Helper function for applyrandrotation mtd
std::vector<double> convertEigenMattoNativeVec(const Eigen::MatrixXd& rotatedMat)
{
  const Eigen::Index len = rotatedMat.size();
  if (len <= 0) return std::vector<double>();
  return std::vector<double>(rotatedMat.data(), rotatedMat.data() + len);
}

void bart::reset_rotation_counters() {
  rot_proposed_.assign(m, 0);
  rot_accepted_.assign(m, 0);
}

void bart::get_rotation_counts(std::vector<size_t>& proposed,
                               std::vector<size_t>& accepted) const {
  proposed = rot_proposed_;
  accepted = rot_accepted_;
}

// Create (cutinfo) xinfo mat for each tree
void bart::applyrandrotation(double *ix, double *ixp, size_t p, size_t n, size_t np, int *nc, rn& gen,
                             bool init_random_rotation){
  formal_cache_.clear();

  if (ix == nullptr || nc == nullptr || p == 0 || n == 0) {
    throw std::invalid_argument("bart::applyrandrotation: invalid training design or dimensions");
  }
  if (np > 0 && ixp == nullptr) {
    throw std::invalid_argument("bart::applyrandrotation: np > 0 with null test design");
  }

  if (user_xinfo_supplied_) {
    if (dynamic_xinfo_on_rotation_ || rotation_xinfo_policy_ == XInfoPolicy::GlobalCoveringUniform)
      throw std::invalid_argument("Custom xinfo requires a fixed, nonglobal grid");
    if (xi.size() != p) throw std::invalid_argument("Custom xinfo predictor dimension mismatch");
    for (size_t v = 0; v < p; ++v) {
      if (nc[v] <= 0 || xi[v].size() != static_cast<size_t>(nc[v]) ||
          !std::is_sorted(xi[v].begin(), xi[v].end()))
        throw std::invalid_argument("Custom xinfo must have sorted candidates matching nc");
      for (double cut : xi[v]) if (!std::isfinite(cut))
        throw std::invalid_argument("Custom xinfo has a nonfinite candidate");
    }
  }

  // (re)initialize all per-tree buffers.  applyrandrotation() is normally
  // called once, but making re-entry safe avoids duplicate dinfo ownership and
  // keeps every pointer vector at exactly m elements.
  for (size_t j = 0; j < dinfo_vec.size(); ++j) delete dinfo_vec[j];
  for (size_t j = 0; j < dinfo_test_vec.size(); ++j) delete dinfo_test_vec[j];
  dinfo_vec.assign(m, nullptr);
  // Keep test pointer vectors tree-aligned even when np == 0.
  // Null entries are passed safely to the rotation kernels.
  dinfo_test_vec.assign(m, nullptr);
  ix_vec.assign(m, nullptr);
  ix_test_vec.assign(m, nullptr);
  xinfo_vec.assign(m, xinfo{});
  rot_state_.assign(m, robart::RotState{});

  np_ = np;
   // (verbose debug) original xi

   // create original x.train mat(n*p)
   Eigen::VectorXd newVec_train(p*n);
   size_t newVec_train_idx=0;

   for(size_t i=0;i<p;i++) {
      for(size_t j=0;j<n;j++) {
         double xx = *(ix+p*j+i);
         newVec_train[newVec_train_idx] = xx;
         newVec_train_idx+=1;
      }
   }
   
   // if np > 0, then create original x.test mat(np*p)
   Eigen::VectorXd newVec_test(p*np);
   size_t newVec_test_idx=0;
   if(np>0){
     
     for(size_t i=0;i<p;i++) {
       for(size_t j=0;j<np;j++) {
         double xx = *(ixp+p*j+i);
         newVec_test[newVec_test_idx] = xx;
         newVec_test_idx+=1;
       }
     }
   }

   Eigen::MatrixXd orig_x_train_mat = Eigen::Map<const Eigen::MatrixXd>(newVec_train.data(), n, p);
   Eigen::MatrixXd t_orig_x_train_mat = orig_x_train_mat.transpose();
   
   Eigen::MatrixXd orig_x_test_mat;
   Eigen::MatrixXd t_orig_x_test_mat;
   if(np>0){
     
     orig_x_test_mat = Eigen::Map<const Eigen::MatrixXd>(newVec_test.data(), np, p);
     t_orig_x_test_mat = orig_x_test_mat.transpose();
     
   }


   Eigen::MatrixXd randRotMat;
   Eigen::MatrixXd rotatedMat_train;
   Eigen::MatrixXd rotatedMat_test;

   // Rotation-invariant fixed grid: |(Qx_i)_v| <= ||x_i||_2 for orthogonal Q.
   xinfo global_covering_xi;
   rotation_global_grid_radius_ = std::numeric_limits<double>::quiet_NaN();
   if (rotation_xinfo_policy_ == XInfoPolicy::GlobalCoveringUniform) {
      build_xinfo_from_design(ix,p,n,nc_per_var_,
                              XInfoPolicy::GlobalCoveringUniform,
                              global_covering_xi,
                              &rotation_global_grid_radius_);
      xi = global_covering_xi;
      if (global_covering_xi.size() != p ||
          !std::isfinite(rotation_global_grid_radius_) ||
          !(rotation_global_grid_radius_ > 0.0)) {
        throw std::runtime_error("bart::applyrandrotation: invalid global-covering grid");
      }
      for (size_t v = 0; v < p; ++v) {
        const size_t expected_nc =
          (v < nc_per_var_.size() && nc_per_var_[v] > 0)
            ? static_cast<size_t>(nc_per_var_[v]) : 0;
        if (global_covering_xi[v].size() != expected_nc ||
            !std::is_sorted(global_covering_xi[v].begin(),
                            global_covering_xi[v].end())) {
          throw std::runtime_error("bart::applyrandrotation: malformed global-covering cutpoint vector");
        }
        for (double z : global_covering_xi[v]) {
          if (!std::isfinite(z)) {
            throw std::runtime_error("bart::applyrandrotation: non-finite global cutpoint");
          }
        }
      }
   }
   
   for(size_t i=0;i<m;++i){
      randRotMat = Eigen::MatrixXd::Identity(p, p);

      if(verbose_ && i<=1){
         cout << "i: " << i << std::endl;
         cout << "(before) randRotMat: " << std::endl << randRotMat << std::endl;
      }
      
      // generateRandomRotationMatrix(..., debug_mode):
      //   debug_mode=true  -> do nothing (keep identity)
      //   debug_mode=false -> generate a random rotation
      generateRandomRotationMatrix(randRotMat, static_cast<int>(p), gen, !init_random_rotation);      

      if(verbose_ && i<=1){
         cout << "i: " << i << std::endl;
         cout << "(after) randRotMat: " << std::endl << randRotMat << std::endl;
      }

      rotatedMat_train = randRotMat * t_orig_x_train_mat; // p*n
      // Convert the p-by-n product to observation-major X[v + p*i] storage.
      
      rot_state_[i].R   = randRotMat; // Store the initial rotation matrix, which may be the identity.
      rot_state_[i].Xtr = convertEigenMattoNativeVec(rotatedMat_train);
      if(rotation_xinfo_policy_==XInfoPolicy::QuantileFixedNC){
        rot_state_[i].canonical_coordinates=true;
        rot_state_[i].raw_train=ix;rot_state_[i].raw_test=ixp;
        rot_state_[i].raw_p=p;rot_state_[i].raw_n=n;rot_state_[i].raw_np=np;
        robart::canonical_project(randRotMat,ix,p,n,rot_state_[i].Xtr.data());
      }
      if (rot_state_[i].Xtr.size() != p*n) {
        throw std::runtime_error("bart::applyrandrotation: training rotation buffer has wrong size");
      }
      ix_vec[i]         = rot_state_[i].Xtr.data();
      if (ix_vec[i] == nullptr) {
        throw std::runtime_error("bart::applyrandrotation: null training rotation buffer");
      }

      if (np > 0) {
        rotatedMat_test   = randRotMat * t_orig_x_test_mat;
        rot_state_[i].Xte = convertEigenMattoNativeVec(rotatedMat_test);
        if(rot_state_[i].canonical_coordinates)
          robart::canonical_project(randRotMat,ixp,p,np,rot_state_[i].Xte.data());
        if (rot_state_[i].Xte.size() != p*np) {
          throw std::runtime_error("bart::applyrandrotation: test rotation buffer has wrong size");
        }
        rot_state_[i].has_test = true;
        ix_test_vec[i]    = rot_state_[i].Xte.data();
        if (ix_test_vec[i] == nullptr) {
          throw std::runtime_error("bart::applyrandrotation: null test rotation buffer");
        }
      }
      
      // debug parts. (No rand. rot. matrix)
      //       }else{



      // for the training dataset
      dinfo *ith_dinfo = nullptr;
      ith_dinfo = new dinfo();
      
      ith_dinfo->p = p;
      ith_dinfo->n = n;
      ith_dinfo->x = ix_vec[i];   // rot_state_[i].Xtr.data()
      ith_dinfo->y = di.y;
      dinfo_vec[i] = ith_dinfo;
      
      // check whether ith_dinfo is valid
      if (dinfo_vec[i] == nullptr){
        cout << "dinfo_vec[i] is not valid when i = " << i << std::endl << std::endl;
      }
      
      // for the test dataset
      if(np > 0){
        dinfo *ith_dinfo_test = nullptr;
        ith_dinfo_test = new dinfo();
        
        ith_dinfo_test->p = p;
        ith_dinfo_test->n = np;
        ith_dinfo_test->x = ix_test_vec[i];  // rot_state_[i].Xte.data()
        // ith_dinfo_test -> y = di.y; // y.test doesn't exist!!
        
        // ith_dinfo -> setdata(p, n, &(vectorXdvec[i])[0], di.y); 
        
        dinfo_test_vec[i] = ith_dinfo_test;
        
        // check whether ith_dinfo is valid
        if (dinfo_test_vec[i] == nullptr){
          cout << "dinfo_test_vec[i] is not valid when i = " << i << std::endl << std::endl;
        }
      }

      // print out dinfo_vec for the training dataset
      if (verbose_ && i==0){
         cout << "dinfo_vec[0]: " << std::endl << dinfo_vec[0] << std::endl; //
         cout << "dinfo_vec[0]->x" << std::endl << dinfo_vec[0]->x << std::endl; //
         cout << "&(dinfo_vec[0]->x)" << std::endl << &(dinfo_vec[0]->x) << std::endl; //
         cout << "dinfo_vec[0]->p" << std::endl << dinfo_vec[0]->p << std::endl; //
         cout << "dinfo_vec[0]->n" << std::endl << dinfo_vec[0]->n << std::endl; //
         cout << std::endl << std::endl;
         
      }

      if (verbose_ && i==1){
        cout << "dinfo_vec[0]: " << std::endl << dinfo_vec[0] << std::endl; //
        cout << "dinfo_vec[0]->x" << std::endl << dinfo_vec[0]->x << std::endl; //
        cout << "&(dinfo_vec[0]->x)" << std::endl << &(dinfo_vec[0]->x) << std::endl; //
        cout << "dinfo_vec[0]->p" << std::endl << dinfo_vec[0]->p << std::endl; //
        cout << "dinfo_vec[0]->n" << std::endl << dinfo_vec[0]->n << std::endl; //
        cout << std::endl << std::endl;
        
        
        cout << "dinfo_vec[1]: " << std::endl << dinfo_vec[1] << std::endl; //
        cout << "dinfo_vec[1]->x" << std::endl << dinfo_vec[1]->x << std::endl; //
        cout << "&(dinfo_vec[1]->x)" << std::endl << &(dinfo_vec[1]->x) << std::endl; //
        cout << "dinfo_vec[1]->p" << std::endl << dinfo_vec[1]->p << std::endl; //
        cout << "dinfo_vec[1]->n" << std::endl << dinfo_vec[1]->n << std::endl; //
        cout << std::endl << std::endl;
      }




      // xinfo
      xinfo ith_xi;
      if (rotation_xinfo_policy_ == XInfoPolicy::GlobalCoveringUniform) {
        ith_xi = global_covering_xi;
      } else if (user_xinfo_supplied_) {
        // Custom positions are defined on each tree's initial coordinate axes.
        // Copy them exactly; rotation OFF + identity initialization gives raw-X cuts.
        ith_xi = xi;
      } else if (dynamic_xinfo_on_rotation_ || rotation_xinfo_policy_ == XInfoPolicy::QuantileFixedNC) {
        // Fixed quantile uses the same floor-index/ties rule as dynamic quantile,
        // but only at initialization. Subsequent grid rebuilding is disabled.
        build_xinfo_from_design(ix_vec[i], p, n, nc_per_var_, rotation_xinfo_policy_, ith_xi);
      } else {
        // Preserve the existing fixed-uniform arithmetic and initialization order.
        makexinfo(p, n, ix_vec[i], ith_xi, nc);
      }

      //    prxi(ith_xi);

      xinfo_vec[i] = ith_xi;


         

   }

   if (verbose_) {
     cout << "xinfo_vec.size(): " << xinfo_vec.size() << std::endl;
     cout << "ix_vec.size(): " << ix_vec.size() << std::endl;
     cout << "dinfo_vec.size(): " << dinfo_vec.size() << std::endl;
     
     if(np > 0){
       cout << "ix_test_vec.size(): " << ix_test_vec.size() << std::endl;
       cout << "dinfo_test_vec.size(): " << dinfo_test_vec.size() << std::endl;
     }
   }

}

void bart::israndrotationvalid(double *ix){
  
  bool is_equal_ix = true;
  bool is_equal_xinfo = true;
  bool is_equal_di_p = true;
  bool is_equal_di_n = true;
  bool is_equal_di_x = true;
  bool is_equal_di_y = true;
  
  for(size_t i=0;i<m;++i){
    
    // 1. cmp with ix
    
    double* ix_end = ix+(n*p);
    
    double* new_xi = ix_vec[i];
    for (double* p = ix; p != ix_end; ++p){
      if(*p != *new_xi){
        is_equal_ix = false;
      }
      ++new_xi;
    }
    
    // 2. cmp with xi
    xinfo ith_tree = xinfo_vec[i];
    for(size_t v=0;v!=xi.size();v++) {
      for(size_t j=0;j!=xi[v].size();j++){
        if(xi[v][j] != ith_tree[v][j]){
          is_equal_xinfo = false;
        }
      }
    }
    
    // 3. cmp with di
    dinfo* ith_dinfo_ptr = dinfo_vec[i];
    if((ith_dinfo_ptr->p) != di.p){
      is_equal_di_p = false;
    }
    if((ith_dinfo_ptr->n) != di.n){
      is_equal_di_n = false;
    }
    
    double* original_di_x = di.x;
    double* original_di_end_x = original_di_x+(n*p);
    
    double* new_di_x = ith_dinfo_ptr -> x;
    double* tmp_x_p = new_di_x;
    
    size_t print_di_x_info_messages_ctn = 0;
    size_t print_di_x_error_messages_ctn = 0;
    for (double* p = original_di_x; p != original_di_end_x; ++p){
      if(*p != *tmp_x_p){
        is_equal_di_x = false;
        if(print_di_x_error_messages_ctn<15){
          cout << "is_equal_di_x == false when original x val is " << *p << " and new x val is " << *tmp_x_p << std::endl;
          ++print_di_x_error_messages_ctn;
        }
      }else{
        //   ++print_di_x_info_messages_ctn;
      }
      ++tmp_x_p;
    }
    
    double* original_di_y = di.y;
    double* original_di_end_y = original_di_y+n;
    
    double* new_di_y = ith_dinfo_ptr -> y;
    double* tmp_y_p = new_di_y;
    
    for (double* p = original_di_y; p != original_di_end_y; ++p){
      if(*p != *tmp_y_p){
        is_equal_di_y = false;
      }
      ++tmp_y_p;
    }
    
  }
  
  if(!is_equal_ix){
    cout << "ix_vec is invalid!!" << std::endl;
  }
  if(!is_equal_xinfo){
    cout << "xinfo_vec is invalid!!" << std::endl;
  }
  if(!is_equal_di_p){
    cout << "dinfo_vec -> p is invalid!!" << std::endl;
  }
  if(!is_equal_di_n){
    cout << "dinfo_vec -> n is invalid!!" << std::endl;
  }
  if(!is_equal_di_x){
    cout << "dinfo_vec -> x is invalid!!" << std::endl;
  }
  if(!is_equal_di_y){
    cout << "dinfo_vec -> y is invalid!!" << std::endl;
  }

}



void bart::compute_alignment_summary(double delta_thresh_deg,
                                     double& A_mean,
                                     double& S_frac,
                                     double& mean_abs_delta,
                                     double& align_mean,
                                     size_t& m_eff) const
{
  A_mean = S_frac = mean_abs_delta = align_mean = 0.0; m_eff = 0;
  if (rot_state_.size() != m) return;
  const double deg2rad = M_PI/180.0;
  double sA=0.0, sS=0.0, sD=0.0, sAl=0.0;
  for (size_t j=0; j<m; ++j) {
    const auto& R = rot_state_[j].R;
    if (R.size()==0) continue;
    double ang, ddeg, al;
    robart::tree_alignment_from_R(R, ang, ddeg, al);
    double A = std::cos(2.0 * ddeg * deg2rad); // Double-angle alignment score.
    sA += A;
    sS += (std::fabs(ddeg) < delta_thresh_deg) ? 1.0 : 0.0;
    sD += std::fabs(ddeg);
    sAl += al;
    ++m_eff;
  }
  if (m_eff>0) {
    A_mean        = sA  / (double)m_eff;
    S_frac        = sS  / (double)m_eff;
    mean_abs_delta= sD  / (double)m_eff;
    align_mean    = sAl / (double)m_eff;
  }
}

void bart::compute_alignment_per_tree(double /*delta_thresh_deg*/,
                                      std::vector<double>& angle_deg,
                                      std::vector<double>& delta_deg,
                                      std::vector<double>& align_score,
                                      std::vector<double>& A_per_tree) const
{
  angle_deg.assign(m, 0.0);
  delta_deg.assign(m, 0.0);
  align_score.assign(m, 0.0);
  A_per_tree.assign(m, 0.0);
  if (rot_state_.size() != m) return;
  const double deg2rad = M_PI/180.0;
  for (size_t j=0; j<m; ++j) {
    const auto& R = rot_state_[j].R;
    if (R.size()==0) continue;
    double ang, ddeg, al;
    robart::tree_alignment_from_R(R, ang, ddeg, al);
    angle_deg[j] = ang;
    delta_deg[j] = ddeg;
    align_score[j]= al;
    A_per_tree[j] = std::cos(2.0 * ddeg * deg2rad);
  }
}

void bart::compute_givens_plane_angles_per_tree(std::vector<double>& out_angles,
                                                double tol) const
{
  const size_t pp = p;
  const size_t nplanes = (pp < 2) ? 0 : (pp * (pp - 1)) / 2;
  out_angles.assign(m * nplanes, 0.0);
  if (nplanes == 0) return;
  if (rot_state_.size() != m) return;

  Eigen::MatrixXd ang;
  for (size_t j = 0; j < m; ++j) {
    const auto& R = rot_state_[j].R;
    if (R.size() == 0) continue;
    robart::givens_plane_angles_from_R(R, ang, tol);

    size_t k = 0;
    for (size_t a = 0; a < pp - 1; ++a) {
      for (size_t b = a + 1; b < pp; ++b) {
        out_angles[j * nplanes + k] = ang((int)a, (int)b);
        ++k;
      }
    }
  }
}
