/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017-2018 Robert McCulloch, Rodney Sparapani
 *                          and Charles Spanbauer
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

#ifndef GUARD_bart_h
#define GUARD_bart_h

#include <ctime>

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "rotation_mh.h"
#include <limits>
#include "input_contract.h"

#include <RcppEigen.h>
// RcppEigen is declared through DESCRIPTION LinkingTo.
//--------------------------------------------------
// use Eigen MatrixXd
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix;

class bart {
public:
   //------------------------------
   //friends
   friend bool bd(tree& x, xinfo& xi, dinfo* di_ptr, pinfo& pi, double sigma,
		  std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen,
		  const DepthPriorSpec& dp, const std::vector<double>* log_weights,
                  robart::formal::TreeCache* state_cache);
   //------------------------------
   //constructor/destructor
   bart();
   bart(size_t m);
   // Descriptor pointers refer into this instance's owned rotated buffers.
   // There is no valid implicit clone/resume operation for a live sampler.
   bart(const bart&) = delete;
   bart(bart&&) = delete;
   ~bart();
   //------------------------------
   //operators
   bart& operator=(const bart&) = delete;
   bart& operator=(bart&&) = delete;
   //------------------------------
   //get,set
   size_t getm() {return m;}
   // void setm(size_t m);
   void setdata(size_t p, size_t n, double *x, double *y, size_t nc=100);
   void setdata(size_t p, size_t n, double *x, double *y, int* nc);
   void setallfit(size_t p, size_t n, double *x);
   void setpi(pinfo& pi) {this->pi = pi;}
   void setprior(double alpha, double beta, double tau)
      {pi.alpha=alpha; pi.mybeta = beta; pi.tau=tau;}
   void setdart(double _a, double _b, double _rho, bool _aug, bool _dart, 
		double _theta=0., double _omega=1.) {
     robart_contract::dart(_a, _b, _rho, _theta, _omega);
     this->a=_a; this->b=_b; this->rho=_rho; this->aug=_aug; 
     this->dart=_dart; this->omega=_omega; 
     if(_theta==0.){
       this->const_theta=false;
       this->theta=1.;
     }
     else{
       this->const_theta=true;
       this->theta=_theta;
     }
}
   void startdart() {this->dartOn=!(this->dartOn);}
   void settau(double tau) {pi.tau=tau;}
   tree& gettree(size_t i ) { return t[i];}
   xinfo& getxinfo() {return xi;}
   void setxinfo(xinfo& _xi);
   std::vector<size_t>& getnv() {return nv;}
   std::vector<double>& getpv() {return pv;}
   const std::vector<double>& get_logpv() const { return lpv; }
   double gettheta() {return theta;}
   bool get_const_theta() const { return const_theta; }
   double getomega() const { return omega; } // Stored compatibility value; no active effect.

   //------------------------------
   //public methods
   void birth(size_t i, size_t nid,size_t v, size_t c, double ml, double mr)
         {t[i].birth(nid,v,c,ml,mr);}
   void death(size_t i,size_t nid, double mu)
         {t[i].death(nid,mu);}
   void pr();
   void tonull() {for(size_t i=0;i!=t.size();i++) t[i].tonull();}
   void predict(size_t p, size_t n, double *x, double *fp, bool is_test=false);

   // ------------------------------------------------------------------
   // Runtime verbosity / tracing controls
   //   - verbose_: controls expensive std::cout / printf debug prints.
   //   - track_last_rot_trace_: controls whether per-draw(per-tree) diagnostic
   //     vectors (last_rot_*) are filled. This is useful for saving detailed
   //     traces to R, but can be disabled to reduce overhead.
   // ------------------------------------------------------------------
   void set_verbose(bool v) { verbose_ = v; }
   bool get_verbose() const { return verbose_; }
   void set_track_last_rot_trace(bool v) { track_last_rot_trace_ = v; }
   bool get_track_last_rot_trace() const { return track_last_rot_trace_; }

   // If true, accumulate accepted Givens angles into RotState::angle_sum/count.
   // This is OFF by default because it adds bookkeeping cost.
   void set_rotation_accumulate_angles(bool v) { rot_accumulate_angles_ = v; }
   bool get_rotation_accumulate_angles() const { return rot_accumulate_angles_; }

   // Experimental controls for diagnostics
   // If true, once small-rotation warmup has ended, only rotation MH is run.
   // Structure, DART and depth-hyper updates are skipped; leaf means are still
   // redrawn to complete the retained leaf-collapsed update.
   void set_rotation_only_after_warmup(bool v) { rotation_only_after_warmup_ = v; }
   bool get_rotation_only_after_warmup() const { return rotation_only_after_warmup_; }


   void set_dynamic_xinfo_on_rotation(bool v) { dynamic_xinfo_on_rotation_ = v; }
   bool get_dynamic_xinfo_on_rotation() const { return dynamic_xinfo_on_rotation_; }
   void set_user_xinfo_supplied(bool v) { user_xinfo_supplied_ = v; }
   bool get_user_xinfo_supplied() const { return user_xinfo_supplied_; }
   void set_rotation_xinfo_policy(XInfoPolicy v) { rotation_xinfo_policy_ = v; }
   XInfoPolicy get_rotation_xinfo_policy() const { return rotation_xinfo_policy_; }
   double get_rotation_global_grid_radius() const { return rotation_global_grid_radius_; }

   // Only the centered bounded, zero-drift kernel is implemented.
   // Rotation OFF is controlled by set_rotation_mh_start_iter()/warmup, not
   // by selecting another proposal kernel.
   void set_centered_rotation_params(double step_size, double angle_bound,
                                     size_t num_planes,
                                     bool retune_cutpoints=true) {
      if (std::isfinite(step_size) && step_size > 0.0)
         rotation_step_size_ = step_size;
      if (std::isfinite(angle_bound) && angle_bound > 0.0)
         rotation_angle_bound_ = angle_bound;
      rotation_num_planes_ = (num_planes == 0 ? 1 : num_planes);
      rotation_retune_cutpoints_ = retune_cutpoints;
   }
   double get_rotation_step_size() const { return rotation_step_size_; }
   double get_rotation_angle_bound() const { return rotation_angle_bound_; }
   size_t get_rotation_num_planes() const { return rotation_num_planes_; }
   bool get_rotation_retune_cutpoints() const { return rotation_retune_cutpoints_; }

   // small-rotation(MH) start control (1-based draw iteration)
   // start_iter=0 : start immediately (default)

   // --- Rotation pair proposal: support-expanding mixture weight ---
   // 0   : legacy inside-support proposal
   // 1   : always try one-in / one-out when the outside support is non-empty
   void set_support_expand_prob(double v) {
      if (!std::isfinite(v)) v = 0.0;
      if (v < 0.0) v = 0.0;
      if (v > 1.0) v = 1.0;
      support_expand_prob_ = v;
   }
   double get_support_expand_prob() const { return support_expand_prob_; }

   // --- Tree depth prior (structure prior) ---
   // Controls the prior probability of splitting a node as a function of its depth.
   // See DepthPriorType / DepthPriorSpec in bartfuns.h.
   void set_depth_prior(int type, double tilt = 0.0, double nu = 0.25) {
      depth_prior_type_ = type;
      depth_prior_tilt_ = tilt;
      depth_prior_nu_   = nu;
   }
   int get_depth_prior_type() const { return depth_prior_type_; }
   double get_depth_prior_tilt() const { return depth_prior_tilt_; }
   double get_depth_prior_nu() const { return depth_prior_nu_; }
   DepthPriorSpec get_depth_prior_spec() const {
      return DepthPriorSpec(depth_prior_type_, depth_prior_tilt_, depth_prior_nu_);
   }



   // --- Optional hyperpriors for depth-prior parameters ---
   // If enabled, we update the depth-prior hyperparameter inside draw() via
   // a single MH step per MCMC iteration (conditional on the current tree ensemble).
   //
   //  * DEPTH_PRIOR_EXPTILT:  tilt > 0, with hyperprior  tilt ~ Gamma(shape, rate)
   //                          (RW proposal on log(tilt))
   //
   //  * DEPTH_PRIOR_NU:       nu in (0, 1/2), implemented via u = 2*nu in (0,1)
   //                          with hyperprior u ~ Beta(a,b)
   //                          (RW proposal on logit(u))
   void set_depth_prior_hyper(bool sample_tilt,
                              double tilt_shape,
                              double tilt_rate,
                              double tilt_rw_sd,
                              bool sample_nu,
                              double nu_a,
                              double nu_b,
                              double nu_rw_sd)
   {
      sample_depth_tilt_ = sample_tilt;
      sample_depth_nu_   = sample_nu;

      if (tilt_shape > 0.0) depth_tilt_shape_ = tilt_shape;
      if (tilt_rate  > 0.0) depth_tilt_rate_  = tilt_rate;
      if (tilt_rw_sd > 0.0) depth_tilt_rw_sd_ = tilt_rw_sd;

      if (nu_a > 0.0) depth_nu_a_ = nu_a;
      if (nu_b > 0.0) depth_nu_b_ = nu_b;
      if (nu_rw_sd > 0.0) depth_nu_rw_sd_ = nu_rw_sd;
   }

   void reset_depth_hyper_counts() {
      depth_tilt_prop_ = depth_tilt_acc_ = 0;
      depth_nu_prop_   = depth_nu_acc_   = 0;
   }

   void get_depth_hyper_counts(size_t& tilt_prop, size_t& tilt_acc,
                               size_t& nu_prop, size_t& nu_acc) const {
      tilt_prop = depth_tilt_prop_;
      tilt_acc  = depth_tilt_acc_;
      nu_prop   = depth_nu_prop_;
      nu_acc    = depth_nu_acc_;
   }

   void set_rotation_mh_start_iter(size_t start_iter) { rot_mh_start_iter_ = start_iter; }

   // Convenience: disable small-rotation proposals for the first `warmup` draws.
   // warmup=0 -> enable from draw 1; warmup=200 -> enable from draw 201.
   void set_rotation_warmup_iters(size_t warmup) {
      rot_mh_start_iter_ = (warmup==0 ? 1 : warmup + 1);
   }

   size_t get_rotation_mh_start_iter() const { return rot_mh_start_iter_; }
   void reset_draw_iter_counter() { draw_iter_counter_ = 0; }

   // After warmup (typically when rotation starts), do multiple tree-structure updates
   // (bd+drmu) per tree within one MCMC iteration.
   // reps>=1. reps=1 reproduces the original behavior.
   void set_tree_update_reps_after_warmup(size_t reps) {
      tree_update_reps_after_warmup_ = (reps < 1 ? 1 : reps);
   }
   size_t get_tree_update_reps_after_warmup() const { return tree_update_reps_after_warmup_; }


   // ------------------------------------------------------------------
   // Optional profiling / diagnostics throttling for rotation MH
   //   every = 0  -> disabled
   //   every = k  -> compute only when (draw_iter_counter_ % k == 0)
   // ------------------------------------------------------------------
   void set_rotation_diag_every(size_t every)   { rot_diag_every_ = every; }
   void set_rotation_timing_every(size_t every) { rot_timing_every_ = every; }
   size_t get_rotation_diag_every() const       { return rot_diag_every_; }
   size_t get_rotation_timing_every() const     { return rot_timing_every_; }


   void draw(double sigma, rn& gen);
//   void draw_s(rn& gen);
   double f(size_t i) {return allfit[i];}

   // Current-state prior diagnostics (for iteration-level posterior tracing)
   double state_logprior_mu();
   double state_logprior_tree();
   double forest_logprior(const DepthPriorSpec&, const std::vector<double>&, bool terminal_only);
   void update_coordinate_probabilities(rn& gen);

   // Per-tree rotation initialization and validation.
   // If init_random_rotation==true, apply a random initial rotation to each tree.
   // If false (default), keep the identity (no initial random rotation).
   void applyrandrotation(double *ix, double *ixp, size_t p, size_t n, size_t np,
                          int *nc, rn& gen, bool init_random_rotation=false);
   void israndrotationvalid(double *ix);


   const xinfo& get_tree_xinfo(size_t j) const { return xinfo_vec[j]; }
   void get_tree_rotation_matrix(size_t j, Eigen::MatrixXd& out) const {
      if (j < rot_state_.size() && rot_state_[j].R.size() != 0) out = rot_state_[j].R;
      else out = Eigen::MatrixXd::Identity((int)p, (int)p);
   }

   // Rotation-MH counters API
   void reset_rotation_counters(); // resize to m and zero-fill
   void get_rotation_counts(std::vector<size_t>& proposed,
                            std::vector<size_t>& accepted) const;

   // Return alignment diagnostics averaged over eligible trees.
   void compute_alignment_summary(double delta_thresh_deg,
                                  double& A_mean,          // E[cos(2*Δ)]
                                  double& S_frac,          // P(|Δ|<delta)
                                  double& mean_abs_delta,  // E[|Δ|] in degrees
                                  double& align_mean,      // E[align_score]
                                  size_t& m_eff) const;    // Number of eligible trees.
 
   // Return per-tree angles, residual angles, alignment scores and A values.
   void compute_alignment_per_tree(double delta_thresh_deg,
                                   std::vector<double>& angle_deg,
                                   std::vector<double>& delta_deg,
                                   std::vector<double>& align_score,
                                   std::vector<double>& A_per_tree) const;

   // ------------------------------------------------------------------
   // Return canonical Givens plane angles in radians for each tree.
   //
   // Decompose each accumulated rotation R_j using a fixed elimination order.
   // Return the corresponding angle for each plane.
   //
   // Output layout:
   // out_angles has length m * nplanes, where nplanes = p*(p-1)/2.
   // out_angles[j*nplanes + k] is the angle in radians for plane (a_k,b_k).
   // Plane indices follow lexicographic order: a=0..p-2, b=a+1..p-1.
   // ------------------------------------------------------------------
   void compute_givens_plane_angles_per_tree(std::vector<double>& out_angles,
                                             double tol = 1e-12) const;

   void get_last_rotation_accepts(std::vector<int>& out) const { out = last_rot_accept_; }
   void get_last_nbots(std::vector<size_t>& out) const { out = last_nbots_; }
   void get_last_rot_logacc(std::vector<double>& out) const { out = last_rot_logacc_; }
   void get_last_rot_memb_change(std::vector<double>& out) const { out = last_rot_memb_change_; }
   void get_last_rot_leaf_n_L1(std::vector<double>& out) const { out = last_rot_leaf_n_L1_; }
   void get_last_rot_leaf_sy_L1(std::vector<double>& out) const { out = last_rot_leaf_sy_L1_; }
   void get_last_rot_ll_current(std::vector<double>& out) const { out = last_rot_ll_current_; }
   void get_last_rot_ll_proposed(std::vector<double>& out) const { out = last_rot_ll_proposed_; }
   void get_last_rot_global_ll_current(std::vector<double>& out) const { out = last_rot_global_ll_current_; }
   void get_last_rot_global_ll_proposed(std::vector<double>& out) const { out = last_rot_global_ll_proposed_; }

   // Rotation-MH timing summary for the last draw() call (only when enabled)
   // Order: seq, xprop, cut_fwd, diag, ll, cut_bwd, accept, total
   void get_last_rot_timing_ms(std::vector<double>& out) const { out = last_rot_timing_ms_; }
   size_t get_last_rot_timing_m_eff() const { return last_rot_timing_m_eff_; }

protected:
   // Update depth-prior hyperparameters (tilt or nu) conditional on current trees.
   // Called at the end of draw() when hyper-learning is enabled.
   void update_depth_prior_hypers(rn& gen);

   size_t m;  //number of trees
   std::vector<tree> t; //the trees
   // Geometry owns validated support/grid snapshots; prior keys include the
   // complete structure and weights/depth parameters, but not residuals/means.
   std::vector<robart::formal::TreeCache> formal_cache_;
   pinfo pi; //prior and mcmc info
   // original input data
   size_t p,n; //x has dim p, n obserations
   double *x,*y;  //x is column stack, pxn
   xinfo xi; //cutpoint info
   //working
   double *allfit; //if the data is set, should be f(x)
   double *r;
   double *ftemp;
   dinfo di;
   bool dart = false, dartOn = false, aug = false, const_theta = false;
   double a = 0.5, b = 1.0, rho = 1.0, theta = 1.0, omega = 1.0;
   std::vector<size_t> nv; // Per-coordinate split counts used by variable selection.
   std::vector<double> pv, lpv; // Variable selection probabilities and their logarithms.
   

   // Rotation kernel controls.  These are intentionally state-independent after
   // initialization; changing them adaptively inside sampling would require
   // diminishing-adaptation arguments.
   double rotation_step_size_ = 0.05;
   double rotation_angle_bound_ = 0.35;
   size_t rotation_num_planes_ = 1;
   bool   rotation_retune_cutpoints_ = true;

   // Probability of proposing a support-expanding rotation pair (one-in / one-out)
   double support_expand_prob_ = 0.0;


   // Depth prior options
   int depth_prior_type_ = DEPTH_PRIOR_POLY;
   double depth_prior_tilt_ = 0.0;
   double depth_prior_nu_ = 0.25;
   

   // --- Optional hyperprior learning for depth prior ---
   // (1) Exponential tilt parameter (DEPTH_PRIOR_EXPTILT)
   bool   sample_depth_tilt_ = false;
   double depth_tilt_shape_  = 2.0;  // Gamma(shape, rate)
   double depth_tilt_rate_   = 8.0;
   double depth_tilt_rw_sd_  = 0.15; // RW sd on log(tilt)
   size_t depth_tilt_prop_   = 0;
   size_t depth_tilt_acc_    = 0;

   // (2) Exponential depth parameter (DEPTH_PRIOR_NU), with u = 2*nu ~ Beta(a,b)
   bool   sample_depth_nu_   = false;
   double depth_nu_a_        = 2.0;  // Beta(a,b) on u=2*nu
   double depth_nu_b_        = 2.0;
   double depth_nu_rw_sd_    = 0.35; // RW sd on logit(u)
   size_t depth_nu_prop_     = 0;
   size_t depth_nu_acc_      = 0;

   // draw() call counter for MCMC iteration (1-based after increment)
   size_t draw_iter_counter_ = 0;
   // small rotation(MH) is enabled only when draw_iter_counter_ >= rot_mh_start_iter_
   // (1-based). Default 0 => enabled from the beginning.
   size_t rot_mh_start_iter_ = 0;

   // # of (bd+drmu) updates per tree per MCMC iteration, after warmup
   size_t tree_update_reps_after_warmup_ = 1;

   // If >0, compute expensive partition diagnostics only every k-th draw
   size_t rot_diag_every_ = 0;
   // If >0, collect rotation timing breakdown only every k-th draw
   size_t rot_timing_every_ = 0;

   // ---- Runtime flags (see setters above) ----
   bool verbose_ = false;
   bool track_last_rot_trace_ = false;
   bool rot_accumulate_angles_ = false;

   // Experimental controls
   bool rotation_only_after_warmup_ = false;
   bool dynamic_xinfo_on_rotation_ = false;
   bool user_xinfo_supplied_ = false; // Explicit fixed candidate locations, not auto-generated Xinfo.
   XInfoPolicy rotation_xinfo_policy_ = XInfoPolicy::UniformFixedNC;
   std::vector<int> nc_per_var_;

   // Temporary buffer to avoid repeated allocations in predict()
   std::vector<double> fptemp_buf_;


   // Per-tree rotated designs and cutpoint grids.
   std::vector<xinfo> xinfo_vec;
   // for training dataset
   std::vector<double*> ix_vec;
   std::vector<dinfo*> dinfo_vec;
   
   // for test dataset
   std::vector<double*> ix_test_vec;
   std::vector<dinfo*> dinfo_test_vec;
   
   // rotation MH persistent state per-tree
   std::vector<robart::RotState> rot_state_;   // Per-tree rotation matrices and buffers.
   size_t np_ = 0;                              // Cached number of test observations.


   // --- Rotation-MH acceptance counters (per tree) ---
   std::vector<size_t> rot_proposed_;  // # proposals per tree
   std::vector<size_t> rot_accepted_;  // # accepts per tree
   std::vector<int>    last_rot_accept_; // size m, 0/1
   std::vector<size_t> last_nbots_;      // size m, # terminal nodes
   std::vector<double> last_rot_logacc_; // size m, last joint log-MH ratio including cutpoint proposal terms
   std::vector<double> last_rot_ll_current_;   // local collapsed ll_old at rotation proposal time
   std::vector<double> last_rot_ll_proposed_;  // local collapsed ll_new at rotation proposal time
   std::vector<double> last_rot_global_ll_current_;  // global Gaussian ll under current tree contribution
   std::vector<double> last_rot_global_ll_proposed_; // global Gaussian ll under proposed tree contribution
   // --- Rotation-MH proposal diagnostics (per tree; computed for the last proposal, even if rejected) ---
   std::vector<double> last_rot_memb_change_; // fraction of observations whose leaf membership changed
   std::vector<double> last_rot_leaf_n_L1_;   // normalized L1 distance of leaf counts (divide by 2n)
   std::vector<double> last_rot_leaf_sy_L1_;  // normalized L1 distance of leaf residual sums (divide by 2*sum|r|)

   // Rotation timing breakdown for the last draw() call.
   // Stored as sums across trees that were timed.
   // Order: seq, xprop, cut_fwd, diag, ll, cut_bwd, accept, total
   std::vector<double> last_rot_timing_ms_;
   size_t last_rot_timing_m_eff_ = 0;

   // Store the actual covering radius for global-grid snapshots and diagnostics.
   double rotation_global_grid_radius_ = std::numeric_limits<double>::quiet_NaN();



};

#endif
