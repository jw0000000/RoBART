/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2018 Robert McCulloch and Rodney Sparapani
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

#include <ctime>
#include "common.h"
#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "heterbart.h"
#include <set>
#include <climits>
#include "rtnorm.h"
#include "lambda.h"
#include "logit_latent.h"

#include "math.h"
#include <cmath>
#include <algorithm>
#include <sstream>

#include <Rcpp.h>
#include "rcpp_input_contract.h"
// RcppEigen is declared through DESCRIPTION LinkingTo.
#include <RcppEigen.h>

using namespace Rcpp;

#ifndef NoRcpp

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)

#endif

#ifndef NoRcpp
namespace {

Rcpp::List xinfo_to_rlist(const xinfo& xi)
{
  Rcpp::List out(xi.size());
  for (size_t v = 0; v < xi.size(); ++v) {
    Rcpp::NumericVector vv(xi[v].size());
    std::copy(xi[v].begin(), xi[v].end(), vv.begin());
    out[v] = vv;
  }
  return out;
}

xinfo rlist_to_xinfo(const Rcpp::List& xi_list)
{
  const size_t p = xi_list.size();
  xinfo xi(p);
  for (size_t v = 0; v < p; ++v) {
    Rcpp::NumericVector vv = xi_list[v];
    xi[v].assign(vv.begin(), vv.end());
  }
  return xi;
}

std::string xinfo_policy_label(XInfoPolicy policy)
{
  if (policy == XInfoPolicy::QuantileFixedNC) return "quantile";
  if (policy == XInfoPolicy::GlobalCoveringUniform) return "global_uniform";
  return "uniform";
}

XInfoPolicy xinfo_policy_from_label(const std::string& label)
{
  if (label == "quantile") return XInfoPolicy::QuantileFixedNC;
  if (label == "global_uniform") return XInfoPolicy::GlobalCoveringUniform;
  return XInfoPolicy::UniformFixedNC;
}

tree tree_from_string(const std::string& s)
{
  tree tr;
  std::stringstream ss(s);
  ss >> tr;
  return tr;
}

void rotate_x_by_R(const Eigen::MatrixXd& R, const double* x, size_t p, size_t n, std::vector<double>& out)
{
  out.assign(p * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t a = 0; a < p; ++a) {
      double sum = 0.0;
      for (size_t b = 0; b < p; ++b) sum += R((int)a, (int)b) * x[b + p * i];
      out[a + p * i] = sum;
    }
  }
}

double gaussian_loglik_weighted(const double* y, const double* f, const double* w, size_t n, double sigma)
{
  if (!y || !f || !w || !std::isfinite(sigma) || sigma <= 0.0) return NA_REAL;
  const double log2pi = std::log(2.0 * 3.141592653589793238462643383279502884);
  double ll = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double sd = w[i] * sigma;
    if (!std::isfinite(sd) || sd <= 0.0) return NA_REAL;
    const double err = y[i] - f[i];
    ll += -0.5 * (log2pi + 2.0 * std::log(sd) + (err * err) / (sd * sd));
  }
  return ll;
}

Rcpp::List build_state_snapshot(bart& bm, size_t p, size_t m_requested,
                                size_t iter1, double sigma)
{
  const size_t m = bm.getm();
  if (m_requested != m) {
    Rcpp::stop("saved-state tree count is inconsistent with the fitted BART object");
  }
  if (p == 0 || m == 0) {
    Rcpp::stop("cannot save a rotation state with zero predictors or zero trees");
  }

  Rcpp::CharacterVector trees((R_xlen_t)m);
  Rcpp::List xinfo_list((R_xlen_t)m);
  const R_xlen_t rot_len =
    static_cast<R_xlen_t>(p) * static_cast<R_xlen_t>(p) *
    static_cast<R_xlen_t>(m);
  Rcpp::NumericVector rot_arr(rot_len, NA_REAL);
  rot_arr.attr("dim") =
    Rcpp::IntegerVector::create((int)p, (int)p, (int)m);

  for (size_t j = 0; j < m; ++j) {
    std::stringstream ss;
    ss.precision(17);
    ss << bm.gettree(j);
    trees[(R_xlen_t)j] = ss.str();

    const xinfo& xij = bm.get_tree_xinfo(j);
    if (xij.size() != p) {
      Rcpp::stop("saved-state xinfo dimension is inconsistent at tree %d",
                 (int)j + 1);
    }
    for (size_t v = 0; v < p; ++v) {
      if (xij[v].empty()) {
        Rcpp::stop("saved-state xinfo is empty at tree %d, variable %d",
                   (int)j + 1, (int)v + 1);
      }
      if (!std::is_sorted(xij[v].begin(), xij[v].end())) {
        Rcpp::stop("saved-state xinfo is not sorted at tree %d, variable %d",
                   (int)j + 1, (int)v + 1);
      }
      for (double z : xij[v]) {
        if (!std::isfinite(z)) {
          Rcpp::stop("saved-state xinfo contains a non-finite value");
        }
      }
    }
    xinfo_list[(R_xlen_t)j] = xinfo_to_rlist(xij);

    Eigen::MatrixXd Rj;
    bm.get_tree_rotation_matrix(j, Rj);
    if ((size_t)Rj.rows() != p || (size_t)Rj.cols() != p ||
        !Rj.allFinite()) {
      Rcpp::stop("saved-state rotation matrix is invalid at tree %d",
                 (int)j + 1);
    }
    for (size_t cc = 0; cc < p; ++cc) {
      for (size_t rr = 0; rr < p; ++rr) {
        rot_arr[(R_xlen_t)(rr + p * cc + p * p * j)] =
          Rj((int)rr, (int)cc);
      }
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("iter") = (int)iter1,
    Rcpp::Named("sigma") = sigma,
    Rcpp::Named("trees") = trees,
    Rcpp::Named("xinfo") = xinfo_list,
    Rcpp::Named("rot_matrix") = rot_arr,
    Rcpp::Named("dynamic_xinfo_on_rotation") =
      bm.get_dynamic_xinfo_on_rotation(),
    Rcpp::Named("rotation_xinfo_policy") =
      xinfo_policy_label(bm.get_rotation_xinfo_policy()),
    Rcpp::Named("coordinate_rule") =
      (bm.get_rotation_xinfo_policy()==XInfoPolicy::QuantileFixedNC ?
        "ordered_dot_v1" : "incremental_givens_v1"),
    Rcpp::Named("user_xinfo_supplied") = bm.get_user_xinfo_supplied(),
    Rcpp::Named("rotation_global_grid_radius") =
      bm.get_rotation_global_grid_radius(),
    Rcpp::Named("discrete_prior") = "split-attempt-full-terminal-v1",
    Rcpp::Named("coordinate_log_prob") = bm.get_logpv(),
    Rcpp::Named("depth_prior_type") = bm.get_depth_prior_type(),
    Rcpp::Named("depth_prior_tilt") = bm.get_depth_prior_tilt(),
    Rcpp::Named("depth_prior_nu") = bm.get_depth_prior_nu()
  );
}

} // anonymous namespace
#endif
// Registered through the package-owned native API in native_api.cpp.
Rcpp::List cgbart(
    int type,
    size_t n,
    size_t p,
    size_t np,
    Rcpp::NumericVector  xv,
    Rcpp::NumericVector  yv,
    Rcpp::NumericVector  xpv,
    size_t m,
    Rcpp::IntegerVector nc_vec,
    size_t nd,
    size_t burn,
    size_t thin,
    double mybeta,
    double alpha,
    double Offset,
    double tau,
    double nu,
    double lambda,
    double sigma,
    Rcpp::NumericVector wv,
    size_t dart_i,
    double theta,
    double omega,
    double epsilon,
    size_t K,
    Rcpp::IntegerVector grp_vec,
    double a,
    double b,
    double rho,
    size_t aug_i,
    size_t printevery,
    Rcpp::NumericMatrix Xinfo,
    size_t rot_warmup_iters = 1000,
    bool init_random_rotation = false,
    size_t tree_update_reps = 1,
    bool debug = false,
    size_t debug_every = 50,
    bool verbose = false,
    bool save_alignment = false,
    bool save_rot_summary = true,
    bool save_rot_trace = false,
    bool save_rot_ll_trace = false,
    bool save_rot_part_trace = false,
    bool save_rot_angles = false,
    size_t rot_trace_every = 1,
    size_t rot_diag_every = 0,
    size_t rot_timing_every = 0,
    int depth_prior_type = 0,
    double depth_prior_tilt = 0.0,
    double depth_prior_nu = 0.25,
    bool sample_depth_tilt = false,
    double depth_tilt_shape = 2.0,
    double depth_tilt_rate  = 8.0,
    double depth_tilt_rw_sd = 0.15,
    bool sample_depth_nu = false,
    double depth_nu_a = 2.0,
    double depth_nu_b = 2.0,
    double depth_nu_rw_sd = 0.35,
    bool save_depth_hyper = false,
    SEXP extra_ctrlSEXP = R_NilValue
)
{
  // Fail safely if cgbart.o and heterbart.o were built against different
  // bart class layouts.  Without this guard a stale heterbart.o can access
  // shifted protected-member offsets and terminate the R process.
  if (robart_heterbart_compiled_abi_version() != ROBART_HETERBART_ABI_VERSION ||
      robart_heterbart_compiled_sizeof() != sizeof(heterbart)) {
    Rcpp::stop(
      "robart native ABI mismatch: heterbart.o is stale relative to bart.h. "
      "Delete src/*.o and the package shared library, then rebuild from a fresh R session."
    );
  }

  robart_contract::schedule(n, p, np, m, nd, burn, thin, printevery,
                            rot_warmup_iters, tree_update_reps, K);
  robart_contract::require(type >= 1 && type <= 3, "type must be 1, 2 or 3");
  robart_contract::require(dart_i <= 1 && aug_i <= 1, "sparse and augment must be 0 or 1");
  robart_contract::require(std::isfinite(alpha) && alpha > 0.0 && alpha < 1.0,
                          "base must lie strictly between 0 and 1");
  robart_contract::require(std::isfinite(mybeta) && mybeta >= 0.0, "power must be finite and nonnegative");
  robart_contract::positive(tau, "effective leaf scale tau");
  robart_contract::positive(nu, "sigma prior degrees of freedom");
  robart_contract::positive(sigma, "initial sigma");
  robart_contract::require(std::isfinite(lambda) && lambda >= 0.0, "lambda must be finite and nonnegative");
  robart_contract::require(std::isfinite(Offset), "offset must be finite");
  robart_contract::dart(a, b, rho, theta, omega);

  // NOTE:

  // R's .Call interface supports only a limited number of top-level arguments
  // (65 SEXP arguments).  Once the experimental rotation/sigma diagnostics were
  // appended one-by-one to cgbart(), the generated Rcpp wrapper exceeded that
  // limit and gbart() started failing with
  //   "too many arguments in foreign function call".
  //
  // Bundle the last group of experimental controls into a single named List so
  // the public gbart() API can stay unchanged while the .Call arity stays below
  // the hard limit.
  bool save_rot_global_ll_trace = false;
  bool save_iteration_trace = false;
  double true_sigma_for_trace = -1.0;
  bool rotation_only_after_warmup = false;
  bool dynamic_xinfo_on_rotation = false;
  bool dynamic_grid_explicit = false;
  bool user_xinfo_supplied = false;
  std::string rotation_xinfo_policy = "uniform";
  double support_expand_prob = 0.5;

  // Centered rotation proposal controls. These live in extra_ctrl to
  // keep the .Call arity below R's argument limit.
  bool rotation_enabled = true;
  // K and epsilon are positional native inputs, not duplicated in extra_ctrl.
  if (!R_FINITE(epsilon) || epsilon <= 0.0 || K < 1 || K > INT_MAX)
    Rcpp::stop("Invalid native K/epsilon.");
  double rotation_step_size = epsilon;
  double rotation_angle_bound = 0.35 * epsilon / 0.15;
  const int rotation_num_planes = static_cast<int>(K);
  bool rotation_retune_cutpoints = true;

  bool freeze_sigma_after_warmup = false;
  double freeze_sigma_value = NA_REAL;
  Rcpp::IntegerVector save_state_iters = Rcpp::IntegerVector(0);

  if (extra_ctrlSEXP != R_NilValue) {
    Rcpp::List extra_ctrl(extra_ctrlSEXP);

    // Reject unknown controls rather than silently accepting a retired API.
    const std::set<std::string> allowed = {
      "rotation_enabled", "save_rot_global_ll_trace", "save_iteration_trace",
      "true_sigma_for_trace", "rotation_only_after_warmup",
      "dynamic_xinfo_on_rotation", "rotation_xinfo_policy", "user_xinfo_supplied",
      "support_expand_prob", "rotation_angle_bound", "rotation_retune_cutpoints",
      "freeze_sigma_after_warmup", "freeze_sigma_value", "save_state_iters"
    };
    Rcpp::CharacterVector ctrl_names = extra_ctrl.names();
    if (ctrl_names.size() != extra_ctrl.size())
      Rcpp::stop("extra_ctrl must be a uniquely named list.");
    std::set<std::string> seen;
    for (R_xlen_t j = 0; j < ctrl_names.size(); ++j) {
      if (Rcpp::CharacterVector::is_na(ctrl_names[j]))
        Rcpp::stop("extra_ctrl has a missing control name.");
      const std::string key = Rcpp::as<std::string>(ctrl_names[j]);
      if (!allowed.count(key) || !seen.insert(key).second)
        Rcpp::stop("Unknown or duplicate extra_ctrl name: %s", key.c_str());
    }

    for (const char* key : {"rotation_enabled", "save_rot_global_ll_trace", "save_iteration_trace",
         "rotation_only_after_warmup", "dynamic_xinfo_on_rotation", "user_xinfo_supplied",
         "rotation_retune_cutpoints", "freeze_sigma_after_warmup"}) {
      if (extra_ctrl.containsElementNamed(key)) robart_contract::sexp_flag(extra_ctrl[key], key);
    }
    if (extra_ctrl.containsElementNamed("save_rot_global_ll_trace"))
      save_rot_global_ll_trace =
        Rcpp::as<bool>(extra_ctrl["save_rot_global_ll_trace"]);

    if (extra_ctrl.containsElementNamed("save_iteration_trace"))
      save_iteration_trace =
        Rcpp::as<bool>(extra_ctrl["save_iteration_trace"]);

    if (extra_ctrl.containsElementNamed("true_sigma_for_trace"))
      true_sigma_for_trace =
        Rcpp::as<double>(extra_ctrl["true_sigma_for_trace"]);

    if (extra_ctrl.containsElementNamed("rotation_only_after_warmup"))
      rotation_only_after_warmup =
        Rcpp::as<bool>(extra_ctrl["rotation_only_after_warmup"]);


    if (extra_ctrl.containsElementNamed("dynamic_xinfo_on_rotation")) {
      dynamic_xinfo_on_rotation = Rcpp::as<bool>(extra_ctrl["dynamic_xinfo_on_rotation"]);
      dynamic_grid_explicit = true;
    }
    if (extra_ctrl.containsElementNamed("user_xinfo_supplied"))
      user_xinfo_supplied = Rcpp::as<bool>(extra_ctrl["user_xinfo_supplied"]);

    if (extra_ctrl.containsElementNamed("rotation_xinfo_policy"))
      rotation_xinfo_policy =
        Rcpp::as<std::string>(extra_ctrl["rotation_xinfo_policy"]);


    if (extra_ctrl.containsElementNamed("support_expand_prob"))
      support_expand_prob =
        Rcpp::as<double>(extra_ctrl["support_expand_prob"]);

    if (extra_ctrl.containsElementNamed("rotation_enabled"))
      rotation_enabled = Rcpp::as<bool>(extra_ctrl["rotation_enabled"]);

    if (extra_ctrl.containsElementNamed("rotation_angle_bound"))
      rotation_angle_bound = Rcpp::as<double>(extra_ctrl["rotation_angle_bound"]);
    if (extra_ctrl.containsElementNamed("rotation_retune_cutpoints"))
      rotation_retune_cutpoints = Rcpp::as<bool>(extra_ctrl["rotation_retune_cutpoints"]);

    if (extra_ctrl.containsElementNamed("freeze_sigma_after_warmup"))
      freeze_sigma_after_warmup =
        Rcpp::as<bool>(extra_ctrl["freeze_sigma_after_warmup"]);

    if (extra_ctrl.containsElementNamed("freeze_sigma_value"))
      freeze_sigma_value =
        Rcpp::as<double>(extra_ctrl["freeze_sigma_value"]);

    if (extra_ctrl.containsElementNamed("save_state_iters"))
      save_state_iters = robart_contract::sexp_snapshot_iterations(
        extra_ctrl["save_state_iters"], burn + nd);
  }

  // Timing is still stored at trace records; reject unsupported schedules
  // rather than silently dropping timing samples. No timing algorithm changes.
  robart_contract::recording_schedule(rot_trace_every, rot_timing_every,
    save_rot_trace || save_rot_ll_trace || save_rot_part_trace || save_rot_global_ll_trace);

  robart_contract::require(std::isfinite(support_expand_prob) && support_expand_prob >= 0.0 &&
                          support_expand_prob <= 1.0, "support_expand_prob must be in [0,1]");
  robart_contract::require(std::isfinite(rotation_angle_bound) && rotation_angle_bound > 0.0 &&
                          rotation_angle_bound < 3.14159265358979323846 - 1e-8,
                          "rotation angle bound must be in (0,pi-1e-8)");
  if (!R_IsNA(freeze_sigma_value)) robart_contract::positive(freeze_sigma_value, "freeze_sigma_value");
  if (!R_IsNA(true_sigma_for_trace) && true_sigma_for_trace != -1.0)
    robart_contract::positive(true_sigma_for_trace, "true_sigma_for_trace");

  // R normally resolves these controls. Direct native callers obey the same rules.
  if (!dynamic_grid_explicit)
    dynamic_xinfo_on_rotation = rotation_enabled && !user_xinfo_supplied && rotation_xinfo_policy != "global_uniform";
  if (rotation_xinfo_policy != "uniform" && rotation_xinfo_policy != "quantile" && rotation_xinfo_policy != "global_uniform")
    Rcpp::stop("Unknown native rotation_xinfo_policy");
  if (dynamic_xinfo_on_rotation && (!rotation_enabled || user_xinfo_supplied || rotation_xinfo_policy == "global_uniform"))
    Rcpp::stop("Dynamic grids require rotation ON and neither custom nor global candidates");
  if (user_xinfo_supplied && (Xinfo.size() == 0 || rotation_xinfo_policy == "global_uniform"))
    Rcpp::stop("Custom grid flag requires Xinfo and is incompatible with global_uniform");

  if (verbose) {
    printf("*****Start process!!!!!*******\n");
  }

  // Process arguments.  Do not index a zero-length Rcpp vector: x.test is
  // optional, and taking &xpv[0] when np == 0 is undefined behaviour and can
  // terminate the R process before any R error can be raised.
  if (n == 0 || p == 0)
    Rcpp::stop("cgbart requires n > 0 and p > 0");

  const R_xlen_t train_len = static_cast<R_xlen_t>(n) * static_cast<R_xlen_t>(p);
  const R_xlen_t test_len  = static_cast<R_xlen_t>(np) * static_cast<R_xlen_t>(p);

  if (xv.size() != train_len)
    Rcpp::stop("x.train has inconsistent length: expected p*n values");
  if (yv.size() != static_cast<R_xlen_t>(n))
    Rcpp::stop("y.train has inconsistent length: expected n values");
  if (wv.size() != static_cast<R_xlen_t>(n))
    Rcpp::stop("weights have inconsistent length: expected n values");
  if (nc_vec.size() != static_cast<R_xlen_t>(p))
    Rcpp::stop("numcut has inconsistent length: expected p values");
  if (np > 0 && xpv.size() != test_len)
    Rcpp::stop("x.test has inconsistent length: expected p*np values");
  if (np == 0 && xpv.size() != 0)
    Rcpp::stop("x.test is nonempty although np == 0");

  for (R_xlen_t i = 0; i < xv.size(); ++i)
    if (!R_FINITE(xv[i])) Rcpp::stop("x.train must be finite after R preprocessing.");
  for (R_xlen_t i = 0; i < xpv.size(); ++i)
    if (!R_FINITE(xpv[i])) Rcpp::stop("x.test must be finite after R preprocessing.");
  for (size_t i = 0; i < n; ++i) {
    if (!R_FINITE(yv[i])) Rcpp::stop("y.train must be finite.");
    if (type != 1 && yv[i] != 0.0 && yv[i] != 1.0)
      Rcpp::stop("Binary y.train must be coded 0 or 1.");
    if (!R_FINITE(wv[i]) || wv[i] <= 0.0 || !R_FINITE(wv[i] * sigma) || wv[i] * sigma <= 0.0)
      Rcpp::stop("Weights and the initial observation SD must be finite and positive.");
  }
  for (size_t j = 0; j < p; ++j)
    if (nc_vec[j] <= 0 || nc_vec[j] == NA_INTEGER)
      Rcpp::stop("Every retained predictor must have at least one cutpoint index.");

  if (Xinfo.size() != 0) {
    if (Xinfo.nrow() != static_cast<int>(p))
      Rcpp::stop("Native Xinfo needs one row per retained predictor.");
    for (size_t j = 0; j < p; ++j) {
      if (nc_vec[j] > Xinfo.ncol()) Rcpp::stop("Native Xinfo has fewer columns than numcut.");
      for (int c = 0; c < nc_vec[j]; ++c)
        if (!R_FINITE(Xinfo(j,c)) || (c > 0 && Xinfo(j,c) < Xinfo(j,c-1)))
          Rcpp::stop("Native Xinfo candidates must be finite and sorted.");
    }
  }

  double *ix = xv.begin();
  double *iy = yv.begin();
  double *ixp = (np > 0) ? xpv.begin() : nullptr;
  int *numcut = nc_vec.begin();
  double *iw = wv.begin();
  bool dart;
  if(dart_i==1) dart=true;
  else dart=false;
  bool aug;
  if(aug_i==1) aug=true;
  else aug=false;
  // grp is currently unused in this translation unit.  Keep a safe pointer
  // for compatibility rather than indexing an empty vector.
  int *grp = grp_vec.size() ? grp_vec.begin() : nullptr;
  (void)grp;
  size_t nkeeptrain = nd/thin;     //Rcpp::as<int>(_inkeeptrain);
  size_t nkeeptest = nd/thin;      //Rcpp::as<int>(_inkeeptest);
  size_t nkeeptreedraws = nd/thin; //Rcpp::as<int>(_inkeeptreedraws);
  Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
  Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);
  Rcpp::NumericVector sdraw(nd+burn);
  Rcpp::NumericMatrix trdraw(nkeeptrain,n);
  Rcpp::NumericMatrix tedraw(nkeeptest,np);

  // ---- optional depth-hyperparameter trace (aligned with yhat.train draws) ----
  const bool want_depth_hyper_trace = save_depth_hyper;
  Rcpp::NumericVector depth_tilt_trace;
  Rcpp::NumericVector depth_nu_trace;
  if (want_depth_hyper_trace) {
    depth_tilt_trace = Rcpp::NumericVector(nkeeptrain, NA_REAL);
    depth_nu_trace   = Rcpp::NumericVector(nkeeptrain, NA_REAL);
  }


  //random number generation
  arn gen;

  heterbart bm(m);
  // K and epsilon each have one authoritative native input.
  // (1b) support-expanding pair proposal mixture weight
  bm.set_support_expand_prob(support_expand_prob);

  // (1c) centered rotation proposal controls
  if (!R_FINITE(rotation_angle_bound) || rotation_angle_bound <= 0.0 ||
      rotation_angle_bound >= M_PI - 1e-8)
    Rcpp::stop("Invalid native rotation angle bound.");
  bm.set_centered_rotation_params(rotation_step_size,
                                   rotation_angle_bound,
                                   (size_t)rotation_num_planes,
                                   rotation_retune_cutpoints);
  // small rotation(MH) warmup control (disable for first rot_warmup_iters draws)
  bm.set_rotation_warmup_iters(rot_warmup_iters);
  if (!rotation_enabled) bm.set_rotation_mh_start_iter(burn + nd + 1);
  // after warmup: do multiple structure updates per tree per MCMC iteration
  bm.set_tree_update_reps_after_warmup(tree_update_reps);
  bm.set_rotation_only_after_warmup(rotation_only_after_warmup);
  bm.set_dynamic_xinfo_on_rotation(dynamic_xinfo_on_rotation);
  bm.set_user_xinfo_supplied(user_xinfo_supplied);
  bm.set_rotation_xinfo_policy(xinfo_policy_from_label(rotation_xinfo_policy));

  // ---- runtime verbosity / tracing controls ----
  bm.set_verbose(verbose);
  // Detailed per-tree per-draw traces are only needed when explicitly requested.
  bm.set_track_last_rot_trace(save_rot_trace || save_rot_ll_trace || save_rot_part_trace || save_rot_global_ll_trace || debug);

  // ---- throttling controls for expensive rotation diagnostics / timing ----
  // rot_trace_every is handled in this file (downsampling what we store to R)
  // rot_diag_every controls *computation* of expensive diagnostics inside draw().
  if (save_rot_part_trace && rot_diag_every == 0) rot_diag_every = rot_trace_every;
  bm.set_rotation_diag_every(rot_diag_every);
  bm.set_rotation_timing_every(rot_timing_every);
  
  if(Xinfo.size()>0) {
    if (Xinfo.nrow() != static_cast<int>(p))
      Rcpp::stop("Xinfo must have p rows");
    for (size_t v = 0; v < p; ++v) {
      if (numcut[v] < 1 || numcut[v] > Xinfo.ncol())
        Rcpp::stop("Each nc value must fit inside Xinfo");
      for (int c = 0; c < numcut[v]; ++c) {
        if (!R_FINITE(Xinfo(v, c)) || (c > 0 && Xinfo(v, c) < Xinfo(v, c-1)))
          Rcpp::stop("Active Xinfo candidates must be finite and sorted");
      }
    }
    xinfo _xi;
    _xi.resize(p);
    for(size_t i=0;i<p;i++) {
      _xi[i].resize(numcut[i]);
      for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
    }
    bm.setxinfo(_xi);
  }

  /* multiple imputation hot deck implementation
   this will cause trouble for multiple threads
   so it must be done prior to calling C++ */
  

  std::stringstream treess;  //string stream to write trees to
  treess.precision(10);
  treess << nkeeptreedraws << " " << m << " " << p << endl;

  if (verbose) {
    printf("*****Calling gbart: type=%d\n", type);
  }

  size_t skiptr=thin, skipte=thin, skiptreedraws=thin;
  

  //--------------------------------------------------
  // print args (IO can be a non-trivial bottleneck; keep silent by default)
  if (verbose) {
    printf("*****Data:\n");
    printf("data:n,p,np: %zu, %zu, %zu\n",n,p,np);
    printf("y1,yn: %lf, %lf\n",iy[0],iy[n-1]);
    printf("x1,x[n*p]: %lf, %lf\n",ix[0],ix[n*p-1]);
    if(np) printf("xp1,xp[np*p]: %lf, %lf\n",ixp[0],ixp[np*p-1]);
    printf("*****Number of Trees: %zu\n",m);
    printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
    printf("*****burn,nd,thin: %zu,%zu,%zu\n",burn,nd,thin);
    cout << "*****Prior:beta,alpha,tau,nu,lambda,offset: "
         << mybeta << ',' << alpha << ',' << tau << ','
         << nu << ',' << lambda << ',' << Offset << endl;
    cout << "*****DepthPrior:type,tilt,nu_depth: "
         << depth_prior_type << ',' << depth_prior_tilt << ',' << depth_prior_nu << endl;
    if(type==1) {
      printf("*****sigma: %lf\n",sigma);
      printf("*****w (weights): %lf ... %lf\n",iw[0],iw[n-1]);
    }
    cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: "
         << dart << ',' << theta << ',' << omega << ',' << a << ','
         << b << ',' << rho << ',' << aug << endl;
    printf("*****printevery: %zu\n",printevery);
  }

  //--------------------------------------------------
  //create temporaries
  double df=n+nu;
  // Automatic storage releases scratch arrays when validation/update throws.
  std::vector<double> z_storage(n), svec_storage(n), sign_storage(type != 1 ? n : 0);
  double *z = z_storage.data();
  double *svec = svec_storage.data();
  double *sign = type != 1 ? sign_storage.data() : nullptr;

  for(size_t i=0; i<n; i++) {
    if(type==1) {
      svec[i] = iw[i]*sigma;
      z[i]=iy[i];
    }
    else {
      svec[i] = 1.;
      if(iy[i]==0) sign[i] = -1.;
      else sign[i] = 1.;
      z[i] = (type==3) ? sign[i]-Offset : sign[i];
    }
  }
  //--------------------------------------------------
  //set up BART model
  bm.setprior(alpha,mybeta,tau);
  bm.set_depth_prior(depth_prior_type, depth_prior_tilt, depth_prior_nu);
  bm.set_depth_prior_hyper(sample_depth_tilt, depth_tilt_shape, depth_tilt_rate, depth_tilt_rw_sd,
                           sample_depth_nu, depth_nu_a, depth_nu_b, depth_nu_rw_sd);
  bm.reset_depth_hyper_counts();
  bm.setdata(p,n,ix,z,numcut);
  bm.setdart(a,b,rho,aug,dart,theta,omega);

  //--------------------------------------------------
  // Initialize per-tree rotated designs and grids; identity is also supported.

  
  // Initial random rotation for each tree (optional).
  // init_random_rotation=false -> keep identity (current default behavior)
  // init_random_rotation=true  -> apply a random rotation at initialization
  bm.applyrandrotation(ix, ixp, p, n, np, numcut, gen, init_random_rotation);

  bm.setallfit(p,n,ix); //Initialize the ensemble fit from its per-tree grids and designs.

  // Optional (expensive) sanity check: verify that per-tree rotated buffers are consistent.
  // Keep OFF by default to avoid slowing down MCMC.
  if (debug && verbose) {
    bm.israndrotationvalid(ix);
  }

  bm.reset_rotation_counters();

  // dart iterations
  std::vector<double> ivarprb (p,0.);
  std::vector<size_t> ivarcnt (p,0);

  //--------------------------------------------------
  // ------------------------------------------------------------------
  // Optional diagnostics containers
  //   - Alignment / canonical Givens angles are stored at the same cadence as
  //     train draws (i.e., `nkeeptrain`).
  //   - Rotation traces are stored on the full MCMC time axis (burn+nd), but
  //     can be downsampled via `rot_trace_every`.
  // ------------------------------------------------------------------

  const size_t total = nd + burn;
  Rcpp::NumericVector dart_theta_trace(dart ? total : 0);
  const size_t nsave_align = (nkeeptrain ? nkeeptrain : nd);
  const double align_delta_thresh_deg = 5.0;

  // ---- alignment summaries (optional) ----
  Rcpp::NumericVector align_A, align_S_frac, align_meanAbsDelta, align_alignMean;
  Rcpp::NumericMatrix align_treeAngle, align_treeDelta;
  if (save_alignment) {
    align_A = Rcpp::NumericVector(nsave_align);
    align_S_frac = Rcpp::NumericVector(nsave_align);
    align_meanAbsDelta = Rcpp::NumericVector(nsave_align);
    align_alignMean = Rcpp::NumericVector(nsave_align);
    align_treeAngle = Rcpp::NumericMatrix(m, nsave_align);
    align_treeDelta = Rcpp::NumericMatrix(m, nsave_align);
  }

  // ---- canonical Givens angles (optional; can be very large) ----
  const size_t nplanes = (p < 2) ? 0 : (p * (p - 1)) / 2;
  Rcpp::NumericVector rot_givensAngle;
  Rcpp::IntegerMatrix rot_planePairs;
  if (save_rot_angles && nplanes > 0 && nsave_align > 0) {
    rot_givensAngle = Rcpp::NumericVector(m * nplanes * nsave_align);
    rot_givensAngle.attr("dim") = Rcpp::IntegerVector::create((int)m, (int)nplanes, (int)nsave_align);

    rot_planePairs = Rcpp::IntegerMatrix(2, (int)nplanes);
    {
      size_t kk = 0;
      for (size_t a2 = 0; a2 < p - 1; ++a2) {
        for (size_t b2 = a2 + 1; b2 < p; ++b2) {
          rot_planePairs(0, (int)kk) = (int)a2 + 1;
          rot_planePairs(1, (int)kk) = (int)b2 + 1;
          ++kk;
        }
      }
    }
  }

  // ---- per-iteration(per tree) rotation traces (optional) ----
  const bool want_any_rot_trace = (save_rot_trace || save_rot_ll_trace || save_rot_part_trace || save_rot_global_ll_trace);
  const size_t trace_stride = rot_trace_every;
  const size_t n_trace_save = want_any_rot_trace ? ((total + trace_stride - 1) / trace_stride) : 0;
  Rcpp::IntegerVector rot_trace_iter;

  Rcpp::IntegerMatrix rot_accept_trace_full;
  Rcpp::IntegerMatrix nbots_trace_full;
  Rcpp::NumericMatrix rot_logacc_trace_full;
  Rcpp::NumericMatrix rot_ll_current_trace_full;
  Rcpp::NumericMatrix rot_ll_proposed_trace_full;
  Rcpp::NumericMatrix global_rot_ll_current_trace_full;
  Rcpp::NumericMatrix global_rot_ll_proposed_trace_full;
  Rcpp::NumericMatrix global_rot_ll_diff_trace_full;
  Rcpp::NumericMatrix rot_memb_change_trace_full;
  Rcpp::NumericMatrix rot_leaf_n_L1_trace_full;
  Rcpp::NumericMatrix rot_leaf_sy_L1_trace_full;

  if (want_any_rot_trace) {
    rot_trace_iter = Rcpp::IntegerVector(n_trace_save);
    if (save_rot_trace) {
      rot_accept_trace_full = Rcpp::IntegerMatrix(n_trace_save, m);
      nbots_trace_full      = Rcpp::IntegerMatrix(n_trace_save, m);
    }
    if (save_rot_ll_trace) {
      rot_logacc_trace_full    = Rcpp::NumericMatrix(n_trace_save, m);
      rot_ll_current_trace_full  = Rcpp::NumericMatrix(n_trace_save, m);
      rot_ll_proposed_trace_full = Rcpp::NumericMatrix(n_trace_save, m);
    }
    if (save_rot_global_ll_trace) {
      global_rot_ll_current_trace_full  = Rcpp::NumericMatrix(n_trace_save, m);
      global_rot_ll_proposed_trace_full = Rcpp::NumericMatrix(n_trace_save, m);
      global_rot_ll_diff_trace_full     = Rcpp::NumericMatrix(n_trace_save, m);
    }
    if (save_rot_part_trace) {
      rot_memb_change_trace_full = Rcpp::NumericMatrix(n_trace_save, m);
      rot_leaf_n_L1_trace_full   = Rcpp::NumericMatrix(n_trace_save, m);
      rot_leaf_sy_L1_trace_full  = Rcpp::NumericMatrix(n_trace_save, m);
    }
  }

  // ---- full-iteration posterior decomposition traces (optional) ----
  Rcpp::NumericVector global_loglik_trace;
  Rcpp::NumericVector global_loglik_true_sigma_trace;
  Rcpp::NumericVector global_logprior_mu_trace;
  Rcpp::NumericVector global_logprior_tree_trace;
  Rcpp::NumericVector global_logprior_sigma_trace;
  Rcpp::NumericVector global_logpost_trace;
  Rcpp::NumericVector iteration_elapsed_sec_trace;
  std::chrono::steady_clock::time_point iteration_clock_start;
  if (save_iteration_trace) {
    global_loglik_trace            = Rcpp::NumericVector(total, NA_REAL);
    global_loglik_true_sigma_trace = Rcpp::NumericVector(total, NA_REAL);
    global_logprior_mu_trace       = Rcpp::NumericVector(total, NA_REAL);
    global_logprior_tree_trace     = Rcpp::NumericVector(total, NA_REAL);
    global_logprior_sigma_trace    = Rcpp::NumericVector(total, NA_REAL);
    global_logpost_trace           = Rcpp::NumericVector(total, NA_REAL);
    iteration_elapsed_sec_trace = Rcpp::NumericVector(total, NA_REAL);
    iteration_clock_start = std::chrono::steady_clock::now();
  }

  // ---- optional saved state snapshots (for angle-profile diagnostics) ----
  std::vector<int> save_state_iters_cpp;
  save_state_iters_cpp.reserve(save_state_iters.size());
  for (int kk = 0; kk < save_state_iters.size(); ++kk) {
    const int it = save_state_iters[kk];
    // Raw inputs were validated before integer conversion and sampler creation.
    save_state_iters_cpp.push_back(it);
  }
  std::sort(save_state_iters_cpp.begin(), save_state_iters_cpp.end());
  save_state_iters_cpp.erase(std::unique(save_state_iters_cpp.begin(), save_state_iters_cpp.end()), save_state_iters_cpp.end());
  const size_t n_state_save = save_state_iters_cpp.size();
  Rcpp::List saved_states(n_state_save);
  size_t save_state_next_idx = 0;

  // ---- per-iteration timing breakdown (optional; cheap when disabled) ----
  const bool want_rot_timing = (rot_timing_every > 0);
  const size_t n_timing_save = want_rot_timing ? ((total + rot_timing_every - 1) / rot_timing_every) : 0;
  Rcpp::IntegerVector rot_timing_iter;
  Rcpp::NumericMatrix rot_timing_ms;
  Rcpp::IntegerVector rot_timing_m_eff;
  if (want_rot_timing) {
    rot_timing_iter = Rcpp::IntegerVector(n_timing_save, NA_INTEGER);
    rot_timing_ms   = Rcpp::NumericMatrix(n_timing_save, 8);
    rot_timing_m_eff= Rcpp::IntegerVector(n_timing_save, NA_INTEGER);
    std::fill(rot_timing_ms.begin(), rot_timing_ms.end(), NA_REAL);
  }

  //--------------------------------------------------
  //temporary storage
  //out of sample fit
  std::vector<double> fhattest_storage(np);
  double* fhattest = np ? fhattest_storage.data() : nullptr;

  //--------------------------------------------------
  //mcmc
  if (verbose) printf("\nMCMC\n");
  size_t trcnt=0; //count kept train draws
  size_t tecnt=0; //count kept test draws
  bool keeptest,/*keeptestme*/keeptreedraw,type1sigest=(type==1 && lambda!=0.);
  bool sigma_frozen_active = false;
  double sigma_frozen_value = sigma;

  time_t tp;
  int time1 = time(&tp), total_dummy=total; // Retain a separate name for the total-iteration value.
  xinfo& xi = bm.getxinfo();

  size_t rot_trace_cnt = 0;
  size_t rot_timing_cnt = 0;
  for(size_t i=0;i<total;i++) {
    const size_t iter1 = i + 1;
    if(verbose && i%printevery==0) printf("done %zu (out of %lu)\n",i,nd+burn);
    if(i==(burn/2)&&dart) bm.startdart();

    const bool sigma_freeze_now = (type == 1 && freeze_sigma_after_warmup && (iter1 >= (rot_warmup_iters + 1)));
    if (sigma_freeze_now) {
      if (!sigma_frozen_active) {
        if (R_FINITE(freeze_sigma_value) && freeze_sigma_value > 0.0) sigma_frozen_value = freeze_sigma_value;
        sigma_frozen_active = true;
      }
      sigma = sigma_frozen_value;
      for (size_t k = 0; k < n; ++k) svec[k] = iw[k] * sigma;
    }

    //draw bart
    bm.draw(svec,gen);
    if (dart) dart_theta_trace[i] = bm.gettheta();

    // (optional) store per-tree rotation diagnostics traces, possibly downsampled
    if (want_any_rot_trace && (iter1 % trace_stride == 0 || iter1 == total)) {
      if (rot_trace_cnt < n_trace_save) {
        rot_trace_iter[rot_trace_cnt] = (int)iter1;

        std::vector<int>    accv;
        std::vector<size_t> nbv;
        std::vector<double> logv;
        std::vector<double> llcurv;
        std::vector<double> llpropv;
        std::vector<double> gllcurv;
        std::vector<double> gllpropv;
        std::vector<double> mcv;
        std::vector<double> lnv;
        std::vector<double> lsyv;

        if (save_rot_trace) {
          bm.get_last_rotation_accepts(accv);
          bm.get_last_nbots(nbv);
        }

	    // (optional) store timing breakdown (only populated on iterations where
	    // bart::draw() actually collects timing, i.e., every rot_timing_every)
	    if (want_rot_timing) {
	      std::vector<double> tms;
	      bm.get_last_rot_timing_ms(tms);
	      if (tms.size() == 8 && rot_timing_cnt < n_timing_save) {
	        rot_timing_iter[rot_timing_cnt] = (int)iter1;
	        for (int c = 0; c < 8; ++c) rot_timing_ms(rot_timing_cnt, c) = tms[(size_t)c];
	        const size_t meff = bm.get_last_rot_timing_m_eff();

	        rot_timing_m_eff[rot_timing_cnt] = (int)meff;
	        ++rot_timing_cnt;
	      }
	    }
        if (save_rot_ll_trace) {
          bm.get_last_rot_logacc(logv);
          bm.get_last_rot_ll_current(llcurv);
          bm.get_last_rot_ll_proposed(llpropv);
        }
        if (save_rot_global_ll_trace) {
          bm.get_last_rot_global_ll_current(gllcurv);
          bm.get_last_rot_global_ll_proposed(gllpropv);
        }
        if (save_rot_part_trace) {
          bm.get_last_rot_memb_change(mcv);
          bm.get_last_rot_leaf_n_L1(lnv);
          bm.get_last_rot_leaf_sy_L1(lsyv);
        }

        for (size_t j=0; j<m; ++j) {
          if (save_rot_trace) {
            rot_accept_trace_full(rot_trace_cnt, j) = (j<accv.size()? accv[j] : NA_INTEGER);
            nbots_trace_full(rot_trace_cnt, j)      = (j<nbv.size()? (int)nbv[j] : NA_INTEGER);
          }
          if (save_rot_ll_trace) {
            rot_logacc_trace_full(rot_trace_cnt, j) = (j<logv.size()? logv[j] : NA_REAL);
            rot_ll_current_trace_full(rot_trace_cnt, j)  = (j<llcurv.size()? llcurv[j] : NA_REAL);
            rot_ll_proposed_trace_full(rot_trace_cnt, j) = (j<llpropv.size()? llpropv[j] : NA_REAL);
          }
          if (save_rot_global_ll_trace) {
            const double gcur = ((type == 1) && j<gllcurv.size()? gllcurv[j] : NA_REAL);
            const double gprop = ((type == 1) && j<gllpropv.size()? gllpropv[j] : NA_REAL);
            global_rot_ll_current_trace_full(rot_trace_cnt, j)  = gcur;
            global_rot_ll_proposed_trace_full(rot_trace_cnt, j) = gprop;
            global_rot_ll_diff_trace_full(rot_trace_cnt, j)     = (R_FINITE(gcur) && R_FINITE(gprop)) ? (gprop - gcur) : NA_REAL;
          }
          if (save_rot_part_trace) {
            rot_memb_change_trace_full(rot_trace_cnt, j) = (j<mcv.size()? mcv[j] : NA_REAL);
            rot_leaf_n_L1_trace_full(rot_trace_cnt, j)   = (j<lnv.size()? lnv[j] : NA_REAL);
            rot_leaf_sy_L1_trace_full(rot_trace_cnt, j)  = (j<lsyv.size()? lsyv[j] : NA_REAL);
          }
        }
        ++rot_trace_cnt;
      }
    }
    if(type1sigest) {
      if (sigma_freeze_now) {
        sigma = sigma_frozen_value;
        sdraw[i] = sigma;
      } else {
        //draw sigma
        double rss=0.;
        for(size_t k=0;k<n;k++) rss += pow((iy[k]-bm.f(k))/(iw[k]), 2.);
        sigma = sqrt((nu*lambda + rss)/gen.chi_square(df));
        sdraw[i]=sigma;
      }
    }

    for(size_t k=0; k<n; k++) {
      if(type==1) svec[k]=iw[k]*sigma;
      else if(type==3){
        // Collapse Z for scale, then restore it with the updated scale.
        robart::update_logit_latent(bm.f(k),Offset,sign[k],svec[k],z[k],gen);
      } else {
        // Preserve the probit update and its RNG order.
        z[k]= sign[k]*rtnorm(sign[k]*bm.f(k), -sign[k]*Offset, svec[k], gen);
      }
    }

    if (save_iteration_trace) {
      const double lp_mu = bm.state_logprior_mu();
      const double lp_tree = bm.state_logprior_tree();
      global_logprior_mu_trace[i] = lp_mu;
      global_logprior_tree_trace[i] = lp_tree;

      if (type == 1) {
        const double log2pi = std::log(2.0 * 3.141592653589793238462643383279502884);

        double ll_cur = 0.0;
        bool ok_cur = true;
        for (size_t k = 0; k < n; ++k) {
          const double sd = svec[k];
          if (!std::isfinite(sd) || sd <= 0.0) { ok_cur = false; break; }
          const double err = iy[k] - bm.f(k);
          ll_cur += -0.5 * (log2pi + 2.0 * std::log(sd) + (err * err) / (sd * sd));
        }
        global_loglik_trace[i] = ok_cur ? ll_cur : NA_REAL;

        if (R_FINITE(true_sigma_for_trace) && true_sigma_for_trace > 0.0) {
          double ll_true = 0.0;
          bool ok_true = true;
          for (size_t k = 0; k < n; ++k) {
            const double sd = iw[k] * true_sigma_for_trace;
            if (!std::isfinite(sd) || sd <= 0.0) { ok_true = false; break; }
            const double err = iy[k] - bm.f(k);
            ll_true += -0.5 * (log2pi + 2.0 * std::log(sd) + (err * err) / (sd * sd));
          }
          global_loglik_true_sigma_trace[i] = ok_true ? ll_true : NA_REAL;
        } else {
          global_loglik_true_sigma_trace[i] = NA_REAL;
        }

        double lp_sigma = 0.0;
        if (lambda != 0.0) {
          if (std::isfinite(sigma) && sigma > 0.0) {
            lp_sigma = -(nu + 1.0) * std::log(sigma) - (nu * lambda) / (2.0 * sigma * sigma);
          } else {
            lp_sigma = NA_REAL;
          }
        }
        global_logprior_sigma_trace[i] = lp_sigma;

        if (R_FINITE(global_loglik_trace[i]) && R_FINITE(lp_mu) && R_FINITE(lp_tree) && R_FINITE(lp_sigma)) {
          global_logpost_trace[i] = global_loglik_trace[i] + lp_mu + lp_tree + lp_sigma;
        } else {
          global_logpost_trace[i] = NA_REAL;
        }
      } else {
        global_loglik_trace[i] = NA_REAL;
        global_loglik_true_sigma_trace[i] = NA_REAL;
        global_logprior_sigma_trace[i] = NA_REAL;
        global_logpost_trace[i] = NA_REAL;
      }
    }

    if (save_state_next_idx < n_state_save && iter1 == (size_t)save_state_iters_cpp[save_state_next_idx]) {
      saved_states[save_state_next_idx] = build_state_snapshot(bm, p, m, iter1, sigma);
      ++save_state_next_idx;
    }

    // --- Debug (rotation diagnostics): print every debug_every-th iteration ---
    if (debug && debug_every > 0) {
      if ((iter1 % debug_every) == 0 || iter1 == total) {

        // (1) Training fit metric (MSE on the working y-scale used in C++: for wbart, y.train is already offset-centered)
        double mse_tr = NA_REAL;
        if (n > 0) {
          double sse = 0.0;
          for (size_t k = 0; k < n; ++k) {
            const double e = iy[k] - bm.f(k);
            sse += e * e;
          }
          mse_tr = sse / (double)n;
        }

        // (2) Rotation counters (cumulative across MCMC)
        std::vector<size_t> rot_prop_tree, rot_acc_tree;
        bm.get_rotation_counts(rot_prop_tree, rot_acc_tree);
        size_t rot_prop_cum = 0, rot_acc_cum = 0;
        for (size_t j = 0; j < rot_prop_tree.size(); ++j) rot_prop_cum += rot_prop_tree[j];
        for (size_t j = 0; j < rot_acc_tree.size(); ++j)  rot_acc_cum  += rot_acc_tree[j];

        // (3) Rotation stats (this iteration, averaged over trees)
        std::vector<int>    acc_last;
        std::vector<size_t> nbots_last;
        std::vector<double> logacc_last, llcur_last;
        std::vector<double> memb_last, leafn_last, leafsy_last;
        bm.get_last_rotation_accepts(acc_last);
        bm.get_last_nbots(nbots_last);
        bm.get_last_rot_logacc(logacc_last);
        bm.get_last_rot_ll_current(llcur_last);
        bm.get_last_rot_memb_change(memb_last);
        bm.get_last_rot_leaf_n_L1(leafn_last);
        bm.get_last_rot_leaf_sy_L1(leafsy_last);

        size_t rot_prop_it = 0, rot_acc_it = 0;
        double sum_logacc = 0.0; size_t cnt_logacc = 0;
        double sum_memb = 0.0;   size_t cnt_memb = 0;
        double sum_leafn = 0.0;  size_t cnt_leafn = 0;
        double sum_leafsy = 0.0; size_t cnt_leafsy = 0;
        double sum_nbots = 0.0;  size_t cnt_nbots = 0;

        for (size_t j = 0; j < m; ++j) {
          // proposed if ll_current was computed for this tree in this iteration
          const bool proposed = (j < llcur_last.size() && R_FINITE(llcur_last[j]));
          if (proposed) rot_prop_it += 1;

          if (j < acc_last.size() && acc_last[j] == 1) rot_acc_it += 1;

          if (j < logacc_last.size() && R_FINITE(logacc_last[j])) {
            sum_logacc += logacc_last[j];
            cnt_logacc += 1;
          }

          if (j < memb_last.size() && R_FINITE(memb_last[j])) {
            sum_memb += memb_last[j];
            cnt_memb += 1;
          }

          if (j < leafn_last.size() && R_FINITE(leafn_last[j])) {
            sum_leafn += leafn_last[j];
            cnt_leafn += 1;
          }

          if (j < leafsy_last.size() && R_FINITE(leafsy_last[j])) {
            sum_leafsy += leafsy_last[j];
            cnt_leafsy += 1;
          }

          if (j < nbots_last.size()) {
            sum_nbots += (double)nbots_last[j];
            cnt_nbots += 1;
          }
        }

        const double acc_rate_it  = (rot_prop_it > 0 ? (double)rot_acc_it / (double)rot_prop_it : NA_REAL);
        const double acc_rate_cum = (rot_prop_cum > 0 ? (double)rot_acc_cum / (double)rot_prop_cum : NA_REAL);

        const double mean_logacc  = (cnt_logacc > 0 ? sum_logacc / (double)cnt_logacc : NA_REAL);
        const double mean_memb    = (cnt_memb > 0 ? sum_memb / (double)cnt_memb : NA_REAL);
        const double mean_leafn   = (cnt_leafn > 0 ? sum_leafn / (double)cnt_leafn : NA_REAL);
        const double mean_leafsy  = (cnt_leafsy > 0 ? sum_leafsy / (double)cnt_leafsy : NA_REAL);
        const double mean_nbots   = (cnt_nbots > 0 ? sum_nbots / (double)cnt_nbots : NA_REAL);

        Rcpp::Rcout << "[debug] iter " << iter1 << "/" << total
                   << " | sigma=" << sigma
                   << " | mse=" << mse_tr
                   << " | rot_it: moves=" << rot_prop_it
                   << " (givens=" << (rot_prop_it * K) << ")"
                   << " acc=" << rot_acc_it << " (";
        if (!R_FINITE(acc_rate_it)) Rcpp::Rcout << "NA";
        else Rcpp::Rcout << (100.0 * acc_rate_it);
        Rcpp::Rcout << "%)"
                   << " | rot_cum: moves=" << rot_prop_cum
                   << " (givens=" << (rot_prop_cum * K) << ")"
                   << " acc=" << rot_acc_cum << " (";
        if (!R_FINITE(acc_rate_cum)) Rcpp::Rcout << "NA";
        else Rcpp::Rcout << (100.0 * acc_rate_cum);
        Rcpp::Rcout << "%)"
                   << " | mean_logacc=" << mean_logacc
                   << " | mean_memb_change=" << mean_memb
                   << " | mean_leaf_n_L1=" << mean_leafn
                   << " | mean_leaf_sy_L1=" << mean_leafsy
                   << " | mean_nbots=" << mean_nbots
                   << std::endl;
      }
    }


    

    if(i>=burn) {
      if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
        for(size_t k=0;k<n;k++) TRDRAW(trcnt,k)=Offset+bm.f(k);
        const size_t k_align = trcnt; // Current stored-draw index populated by TRDRAW.

        if (want_depth_hyper_trace && trcnt < (size_t)depth_tilt_trace.size()) {
          depth_tilt_trace[trcnt] = bm.get_depth_prior_tilt();
          depth_nu_trace[trcnt]   = bm.get_depth_prior_nu();
        }

        // ---- alignment stats (optional) ----
        if (save_alignment) {
          double A,S,md,al; size_t me;
          bm.compute_alignment_summary(align_delta_thresh_deg, A,S,md,al,me);
          align_A[k_align]            = A;
          align_S_frac[k_align]       = S;
          align_meanAbsDelta[k_align] = md;
          align_alignMean[k_align]    = al;

          std::vector<double> ang, delt, als, A_tree;
          bm.compute_alignment_per_tree(align_delta_thresh_deg, ang, delt, als, A_tree);
          for (size_t j=0; j<m; ++j) {
            align_treeAngle(j, k_align) = ang[j];
            align_treeDelta(j, k_align) = delt[j];
          }
        }

        // ---- canonical Givens plane-angle (optional; can be huge) ----
        if (save_rot_angles && nplanes > 0) {
          std::vector<double> gav;
          bm.compute_givens_plane_angles_per_tree(gav);
          const size_t block = m * nplanes;
          const size_t off   = (size_t)k_align * block;
          for (size_t idx = 0; idx < block && idx < gav.size(); ++idx) {
            rot_givensAngle[(R_xlen_t)(off + idx)] = gav[idx];
          }
        }
      
      trcnt+=1;
      }
      keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
      if(keeptest) {
        bm.predict(p,np,ixp,fhattest,true);
        for(size_t k=0;k<np;k++) TEDRAW(tecnt,k)=Offset+fhattest[k];
        tecnt+=1;
      }
      keeptreedraw = nkeeptreedraws && (((i-burn+1) % skiptreedraws) ==0);
      if(keeptreedraw) {
        for(size_t j=0;j<m;j++) {
          treess << bm.gettree(j);

#ifndef NoRcpp
          ivarcnt=bm.getnv();
          ivarprb=bm.getpv();
          size_t k=(i-burn)/skiptreedraws;
          for(size_t j=0;j<p;j++){
            varcnt(k,j)=ivarcnt[j];
            varprb(k,j)=ivarprb[j];
          }
#else
          varcnt.push_back(bm.getnv());
          varprb.push_back(bm.getpv());
#endif
        }
      }
    }

    if (save_iteration_trace) {
      const auto iteration_clock_now = std::chrono::steady_clock::now();
      iteration_elapsed_sec_trace[i] =
        std::chrono::duration<double>(iteration_clock_now - iteration_clock_start).count();
    }
  }
  int time2 = time(&tp);
  if (verbose) {
    printf("time: %ds\n",time2-time1);
    printf("trcnt,tecnt: %zu,%zu\n",trcnt,tecnt);
  }

  // Scratch vectors are released automatically on return or exception.

#ifndef NoRcpp

  // --- rotation-MH acceptance summary (per tree) ---
  // Small but sometimes not needed; allow turning off.
  Rcpp::NumericVector rot_accept_rate;
  Rcpp::IntegerVector rot_proposed, rot_accepted;
  if (save_rot_summary) {
    std::vector<size_t> rot_prop, rot_acc;
    bm.get_rotation_counts(rot_prop, rot_acc);

    rot_accept_rate = Rcpp::NumericVector(m, NA_REAL);
    rot_proposed    = Rcpp::IntegerVector(m, 0);
    rot_accepted    = Rcpp::IntegerVector(m, 0);

    for (size_t j = 0; j < m; ++j) {
      if (j < rot_prop.size()) rot_proposed[j] = static_cast<int>(rot_prop[j]);
      if (j < rot_acc.size())  rot_accepted[j] = static_cast<int>(rot_acc[j]);
      if (rot_proposed[j] > 0) {
        rot_accept_rate[j] =
          static_cast<double>(rot_accepted[j]) / static_cast<double>(rot_proposed[j]);
      }
    }
  }
  Rcpp::List ret;
  ret["coordinate_rule"]=(bm.get_rotation_xinfo_policy()==XInfoPolicy::QuantileFixedNC ?
      "ordered_dot_v1" : "incremental_givens_v1");
  if(type==3){
    ret["logit_latent_update"]="collapsed_scale_then_z_with_offset_v1";
    ret["logit_scale_prior_terms"]=1001;
    ret["logit_scale_prior_approximate"]=true;
  }
  if(type1sigest) ret["sigma"]=sdraw;
  // DART controls are recorded separately from tree snapshots and prediction.
  // omega is stored for API compatibility; it is not read by the active update.
  ret["dart_theta_requested"] = theta;
  ret["dart_theta_fixed"] = bm.get_const_theta();
  ret["dart_theta_last"] = bm.gettheta();
  ret["dart_theta_trace"] = dart_theta_trace;
  ret["dart_theta_trace_first_iter"] = 1;
  ret["dart_theta_trace_recorded_after_tree_update"] = true;
  ret["dart_omega_requested"] = omega;
  ret["dart_omega_stored"] = bm.getomega();
  ret["dart_omega_active"] = false;
  ret["dart_grouped_selection"] = false;
  ret["tau"] = tau;
  ret["sigma_prior_df"] = nu;
  ret["sigma_prior_scale"] = lambda;
  ret["hard_depth_cap"] = false;
  ret["min_child_size_birth"] = 0;
  ret["rotation_enabled"] = rotation_enabled;
  ret["sigma_last"] = sigma;
  ret["rotation_only_after_warmup"] = rotation_only_after_warmup;
  ret["dynamic_xinfo_on_rotation"] = dynamic_xinfo_on_rotation;
  ret["user_xinfo_supplied"] = user_xinfo_supplied;
  ret["grid_mode"] = user_xinfo_supplied ? "fixed_custom" :
    std::string(dynamic_xinfo_on_rotation ? "dynamic_" : "fixed_") + rotation_xinfo_policy;
  ret["rotation_xinfo_policy"] = xinfo_policy_label(bm.get_rotation_xinfo_policy());
  ret["rotation_global_grid_radius"] = bm.get_rotation_global_grid_radius();
  ret["rotation_buffer_storage"] = "std_vector_v3";
  ret["native_heterbart_abi_version"] = robart_heterbart_compiled_abi_version();
  ret["native_heterbart_sizeof"] = static_cast<double>(robart_heterbart_compiled_sizeof());
  ret["native_heterbart_sizeof_cgbart"] = static_cast<double>(sizeof(heterbart));
  ret["support_expand_prob"] = bm.get_support_expand_prob();
  ret["rotation_kernel_label"] = "centered";
  ret["sampler_version"] = "full-discrete-target-stage5A";
  ret["schema_version"] = 9;
  ret["compiled_abi_version"] = ROBART_HETERBART_ABI_VERSION;
  ret["rotation_centered"] = true;
  ret["cutpoint_proposal"] = "single_scale_discrete_gaussian";
  ret["cutpoint_scale_fraction"] = 0.02;
  ret["cutpoint_scale_minimum"] = 1.0;
  ret["discrete_prior"] = "split-attempt-full-terminal-v1";
  ret["cutpoint_center"] = "adjacent-order-statistic-midpoint";
  ret["cutpoint_degenerate_rule"] = "admissible-hold-otherwise-fail";
  ret["tree_update_target"] = "same-full-leaf-collapsed-target-as-rotation";
  ret["dart_update"] = "Dirichlet-independence-MH-terminal-correction";
  ret["augment_proposal"] = aug ? "available-coordinate-proposal-no-virtual-counts" : "full-coordinate-proposal-failure-mass";
  ret["global_logpost_scope"] = "conditional-on-coordinate-and-depth-hyperparameters";
  ret["rotation_step_size"] = bm.get_rotation_step_size();
  ret["rotation_angle_bound"] = bm.get_rotation_angle_bound();
  ret["rotation_num_planes"] = (int)bm.get_rotation_num_planes();
  ret["rotation_retune_cutpoints"] = bm.get_rotation_retune_cutpoints();
  ret["freeze_sigma_after_warmup"] = freeze_sigma_after_warmup;
  ret["freeze_sigma_value"] = sigma_frozen_active ? sigma_frozen_value : NA_REAL;
  ret["rot_warmup_iters"] = (int)rot_warmup_iters;
  ret["yhat.train"]=trdraw;
  ret["yhat.test"]=tedraw;
  ret["varcount"]=varcnt;
  ret["varprob"]=varprb;

  Rcpp::List xiret(xi.size());
  for(size_t i=0;i<xi.size();i++) {
    Rcpp::NumericVector vtemp(xi[i].size());
    std::copy(xi[i].begin(),xi[i].end(),vtemp.begin());
    xiret[i] = Rcpp::NumericVector(vtemp);
  }

  Rcpp::List treesL;
  bool shared_grid = true;
  if (m > 0) for (size_t j = 1; j < m; ++j)
    if (bm.get_tree_xinfo(j) != bm.get_tree_xinfo(0)) { shared_grid = false; break; }
  if (dynamic_xinfo_on_rotation || !shared_grid || m == 0) treesL["cutpoints"] = R_NilValue;
  else treesL["cutpoints"] = xinfo_to_rlist(bm.get_tree_xinfo(0));
  treesL["trees"]=Rcpp::CharacterVector(treess.str());
  ret["treedraws"] = treesL;

  // ---- depth prior hyperparameters (current value + MH stats) ----
  ret["depth_prior_type"] = depth_prior_type;
  ret["depth_tilt_last"]  = bm.get_depth_prior_tilt();
  ret["depth_nu_last"]    = bm.get_depth_prior_nu();

  {
    size_t tprop=0, tacc=0, nprop=0, nacc=0;
    bm.get_depth_hyper_counts(tprop, tacc, nprop, nacc);
    ret["depth_tilt_hyper_proposed"] = (int)tprop;
    ret["depth_tilt_hyper_accepted"] = (int)tacc;
    ret["depth_nu_hyper_proposed"]   = (int)nprop;
    ret["depth_nu_hyper_accepted"]   = (int)nacc;
    ret["depth_tilt_hyper_accept_rate"] = (tprop>0 ? (double)tacc/(double)tprop : NA_REAL);
    ret["depth_nu_hyper_accept_rate"]   = (nprop>0 ? (double)nacc/(double)nprop : NA_REAL);
  }

  if (want_depth_hyper_trace) {
    ret["depth_tilt_trace"] = depth_tilt_trace;
    ret["depth_nu_trace"]   = depth_nu_trace;
  }

  // ---- optional rotation summary ----
  if (save_rot_summary) {
    ret["rot_accept_rate"] = rot_accept_rate;
    ret["rot_proposed"]    = rot_proposed;
    ret["rot_accepted"]    = rot_accepted;
  }

  // ---- optional alignment diagnostics ----
  if (save_alignment) {
    ret["align_A"]             = align_A;
    ret["align_S_frac"]        = align_S_frac;
    ret["align_meanAbsDelta"]  = align_meanAbsDelta;
    ret["align_alignMean"]     = align_alignMean;
    ret["align_treeAngle"]     = align_treeAngle;
    ret["align_treeDelta"]     = align_treeDelta;
    ret["align_delta_thresh"]  = Rcpp::wrap(align_delta_thresh_deg);
  }

  // ---- optional canonical Givens plane-angle trace ----
  if (save_rot_angles) {
    ret["rot_givensAngle"] = rot_givensAngle;
    ret["rot_planePairs"]  = rot_planePairs;
  }

  // ---- optional per-iteration rotation traces ----
  if (want_any_rot_trace) {
    ret["rot_trace_iter"] = rot_trace_iter; // 1-based iteration indices
    if (save_rot_trace) {
      ret["rot_accept_trace_full"] = rot_accept_trace_full;
      ret["nbots_trace_full"]      = nbots_trace_full;
    }
    if (save_rot_ll_trace) {
      ret["rot_logacc_trace_full"]      = rot_logacc_trace_full;
      ret["rot_ll_current_trace_full"]  = rot_ll_current_trace_full;
      ret["rot_ll_proposed_trace_full"] = rot_ll_proposed_trace_full;
    }
    if (save_rot_global_ll_trace) {
      ret["global_rot_ll_current_trace_full"]  = global_rot_ll_current_trace_full;
      ret["global_rot_ll_proposed_trace_full"] = global_rot_ll_proposed_trace_full;
      ret["global_rot_ll_diff_trace_full"]     = global_rot_ll_diff_trace_full;
    }
    if (save_rot_part_trace) {
      ret["rot_memb_change_trace_full"] = rot_memb_change_trace_full;
      ret["rot_leaf_n_L1_trace_full"]   = rot_leaf_n_L1_trace_full;
      ret["rot_leaf_sy_L1_trace_full"]  = rot_leaf_sy_L1_trace_full;
    }
  }

  // ---- optional per-iteration rotation timing ----
  if (want_rot_timing) {
    ret["rot_timing_iter"]  = rot_timing_iter;
    ret["rot_timing_ms"]    = rot_timing_ms;   // rows correspond to rot_timing_iter
    ret["rot_timing_m_eff"] = rot_timing_m_eff;
  }

  if (save_iteration_trace) {
    ret["global_loglik_trace"] = global_loglik_trace;
    ret["global_loglik_true_sigma_trace"] = global_loglik_true_sigma_trace;
    ret["global_logprior_mu_trace"] = global_logprior_mu_trace;
    ret["global_logprior_tree_trace"] = global_logprior_tree_trace;
    ret["global_logprior_sigma_trace"] = global_logprior_sigma_trace;
    ret["global_logpost_trace"] = global_logpost_trace;
    ret["iteration_elapsed_sec_trace"] = iteration_elapsed_sec_trace;
  }

  if (n_state_save > 0) {
    ret["save_state_iters"] = save_state_iters_cpp;
    ret["saved_states"] = saved_states;
  }

  return ret;
#else

#endif

}
// Registered through the package-owned native API in native_api.cpp.
Rcpp::List cgbart_profile_snapshot(
    Rcpp::NumericVector xv,
    Rcpp::NumericVector yv,
    Rcpp::NumericVector wv,
    Rcpp::List state,
    double tau,
    Rcpp::NumericVector theta_grid,
    int tree_index,
    int plane_a,
    int plane_b,
    double sigma_current,
    double sigma_true = NA_REAL)
{
#ifndef NoRcpp
  if (tree_index < 1) Rcpp::stop("tree_index must be >= 1");
  if (plane_a < 1 || plane_b < 1 || plane_a == plane_b) Rcpp::stop("plane indices must be distinct and >= 1");
  if (!R_FINITE(tau) || tau <= 0.0) Rcpp::stop("tau must be positive");
  if (!R_FINITE(sigma_current) || sigma_current <= 0.0) Rcpp::stop("sigma_current must be positive");

  Rcpp::CharacterVector trees_chr = state["trees"];
  Rcpp::List xinfo_state = state["xinfo"];
  const bool dynamic_xinfo_on_rotation = state.containsElementNamed("dynamic_xinfo_on_rotation") ?
    Rcpp::as<bool>(state["dynamic_xinfo_on_rotation"]) : false;
  std::string rotation_xinfo_policy = state.containsElementNamed("rotation_xinfo_policy") ?
    Rcpp::as<std::string>(state["rotation_xinfo_policy"]) : std::string("uniform");
  const bool canonical=state.containsElementNamed("coordinate_rule") &&
    Rcpp::as<std::string>(state["coordinate_rule"])=="ordered_dot_v1";
  Rcpp::NumericVector rot_arr = state["rot_matrix"];
  Rcpp::IntegerVector rot_dim = rot_arr.attr("dim");
  if (rot_dim.size() != 3) Rcpp::stop("state$rot_matrix must have dim c(p,p,m)");

  const size_t p = (size_t)rot_dim[0];
  const size_t p2 = (size_t)rot_dim[1];
  const size_t m = (size_t)rot_dim[2];
  if (p == 0 || p2 != p) Rcpp::stop("invalid rotation matrix dimensions");
  if ((size_t)trees_chr.size() != m || (size_t)xinfo_state.size() != m) Rcpp::stop("state dimensions are inconsistent");

  const size_t n = yv.size();
  if ((size_t)xv.size() != p * n) Rcpp::stop("xv must have length p*n (transposed training matrix)");
  if ((size_t)wv.size() != n) Rcpp::stop("wv must have length n");

  const size_t j0 = (size_t)(tree_index - 1);
  const size_t a = (size_t)(plane_a - 1);
  const size_t b = (size_t)(plane_b - 1);
  if (j0 >= m) Rcpp::stop("tree_index is out of range");
  if (a >= p || b >= p) Rcpp::stop("plane index is out of range");

  double* x = &xv[0];
  double* y = &yv[0];
  double* w = &wv[0];

  std::vector<tree> forest(m);
  std::vector<xinfo> xinfos(m);
  std::vector<Eigen::MatrixXd> Rmats(m);
  std::vector< std::vector<double> > Xcur(m);
  std::vector< std::vector<double> > fcur(m);
  std::vector<double> allfit(n, 0.0);

  for (size_t j = 0; j < m; ++j) {
    forest[j] = tree_from_string(Rcpp::as<std::string>(trees_chr[j]));
    Rcpp::List xi_j = xinfo_state[j];
    xinfos[j] = rlist_to_xinfo(xi_j);

    Eigen::MatrixXd Rj((int)p, (int)p);
    for (size_t cc = 0; cc < p; ++cc) {
      for (size_t rr = 0; rr < p; ++rr) {
        Rj((int)rr, (int)cc) = rot_arr[(R_xlen_t)(rr + p * cc + p * p * j)];
      }
    }
    Rmats[j] = Rj;

    if(canonical){Xcur[j].resize(p*n);
      robart::canonical_project(Rmats[j],x,p,n,Xcur[j].data());
    }else rotate_x_by_R(Rmats[j], x, p, n, Xcur[j]);
    fcur[j].assign(n, 0.0);
    fit(forest[j], xinfos[j], p, n, Xcur[j].data(), fcur[j].data());
    for (size_t i = 0; i < n; ++i) allfit[i] += fcur[j][i];
  }

  std::vector<double> otherfit(n, 0.0), rj(n, 0.0), sigma_vec_cur(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    otherfit[i] = allfit[i] - fcur[j0][i];
    rj[i] = y[i] - otherfit[i];
    sigma_vec_cur[i] = w[i] * sigma_current;
  }

  dinfo di_cur;
  di_cur.n = n; di_cur.p = p; di_cur.x = Xcur[j0].data(); di_cur.y = rj.data();
  const double local_ll_current = robart::tree_log_marglik_hetero(forest[j0], xinfos[j0], &di_cur, sigma_vec_cur.data(), tau);
  const double global_ll_current_sigma_baseline = gaussian_loglik_weighted(y, allfit.data(), w, n, sigma_current);
  const double global_ll_true_sigma_baseline = (R_FINITE(sigma_true) && sigma_true > 0.0)
      ? gaussian_loglik_weighted(y, allfit.data(), w, n, sigma_true)
      : NA_REAL;

  const size_t G = theta_grid.size();
  Rcpp::NumericVector local_collapsed_ll(G, NA_REAL);
  Rcpp::NumericVector global_loglik_current_sigma(G, NA_REAL);
  Rcpp::NumericVector global_loglik_true_sigma(G, NA_REAL);
  Rcpp::NumericVector memb_change(G, NA_REAL);
  Rcpp::NumericVector leaf_n_L1(G, NA_REAL);
  Rcpp::NumericVector leaf_sy_L1(G, NA_REAL);

  std::vector<double> Xprop, fprop(n), fglob(n);
  for (size_t g = 0; g < G; ++g) {
    const double th = theta_grid[g];
    if (!std::isfinite(th)) continue;

    tree tr_prop = forest[j0];
    xinfo xi_prop = xinfos[j0];
    Xprop = Xcur[j0];

    const double c = std::cos(th);
    const double s = std::sin(th);
    if(canonical){Eigen::MatrixXd Qprop=Rmats[j0];
      robart::apply_givens_left_inplace(Qprop,a,b,c,s);
      robart::canonical_project(Qprop,x,p,n,Xprop.data());
    }else robart::apply_givens_inplace(Xprop.data(), p, n, a, b, c, s);
    if (dynamic_xinfo_on_rotation) {
      std::vector<int> nc_local(p, 0);
      for (size_t v = 0; v < p; ++v) nc_local[v] = (int)xinfos[j0][v].size();
      const XInfoPolicy pol = xinfo_policy_from_label(rotation_xinfo_policy);
      build_xinfo_from_design(Xprop.data(), p, n, nc_local, pol, xi_prop);
    } else if(canonical){
      // Deterministic angle sensitivity profile, NOT the stochastic MH proposal.
      std::vector<char> touched(p,0);touched[a]=1;touched[b]=1;
      robart::retune_cutpoints_percentile_match_multi(tr_prop,xi_prop,p,n,
          Xcur[j0].data(),Xprop.data(),touched);
    } else {
      robart::retune_cutpoints_percentile_match(tr_prop, xi_prop, p, n, Xcur[j0].data(), a, b, c, s);
    }

    std::fill(fprop.begin(), fprop.end(), 0.0);
    fit(tr_prop, xi_prop, p, n, Xprop.data(), fprop.data());
    for (size_t i = 0; i < n; ++i) fglob[i] = otherfit[i] + fprop[i];

    dinfo di_prop;
    di_prop.n = n; di_prop.p = p; di_prop.x = Xprop.data(); di_prop.y = rj.data();
    local_collapsed_ll[g] = robart::tree_log_marglik_hetero(tr_prop, xi_prop, &di_prop, sigma_vec_cur.data(), tau);
    global_loglik_current_sigma[g] = gaussian_loglik_weighted(y, fglob.data(), w, n, sigma_current);
    if (R_FINITE(sigma_true) && sigma_true > 0.0) {
      global_loglik_true_sigma[g] = gaussian_loglik_weighted(y, fglob.data(), w, n, sigma_true);
    }

    double mc = NA_REAL, ln = NA_REAL, lsy = NA_REAL;
    robart::compute_partition_diagnostics(forest[j0], tr_prop, xi_prop,
                                          Xcur[j0].data(), Xprop.data(), rj.data(),
                                          p, n, mc, ln, lsy);
    memb_change[g] = mc;
    leaf_n_L1[g] = ln;
    leaf_sy_L1[g] = lsy;
  }

  return Rcpp::List::create(
    Rcpp::Named("theta") = theta_grid,
    Rcpp::Named("tree_index") = tree_index,
    Rcpp::Named("plane") = Rcpp::IntegerVector::create(plane_a, plane_b),
    Rcpp::Named("sigma_current") = sigma_current,
    Rcpp::Named("sigma_true") = sigma_true,
    Rcpp::Named("local_collapsed_ll_current") = local_ll_current,
    Rcpp::Named("global_loglik_current_sigma_current") = global_ll_current_sigma_baseline,
    Rcpp::Named("global_loglik_true_sigma_current") = global_ll_true_sigma_baseline,
    Rcpp::Named("local_collapsed_ll") = local_collapsed_ll,
    Rcpp::Named("global_loglik_current_sigma") = global_loglik_current_sigma,
    Rcpp::Named("global_loglik_true_sigma") = global_loglik_true_sigma,
    Rcpp::Named("memb_change") = memb_change,
    Rcpp::Named("leaf_n_L1") = leaf_n_L1,
    Rcpp::Named("leaf_sy_L1") = leaf_sy_L1
  );
#else
  return Rcpp::List::create();
#endif
}
