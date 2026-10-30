# Package-owned wrappers for registered native routines.
# This file and src/native_api.cpp are maintained together, not regenerated.
# Public interfaces are gbart(), wbart() and robart(); these wrappers are internal.

cgbart <- function(type, n, p, np, xv, yv, xpv, m, nc_vec, nd, burn, thin, mybeta, alpha, Offset, tau, nu, lambda, sigma, wv, dart_i, theta, omega, epsilon, K, grp_vec, a, b, rho, aug_i, printevery, Xinfo, rot_warmup_iters = 1000, init_random_rotation = FALSE, tree_update_reps = 1, debug = FALSE, debug_every = 50, verbose = FALSE, save_alignment = FALSE, save_rot_summary = TRUE, save_rot_trace = FALSE, save_rot_ll_trace = FALSE, save_rot_part_trace = FALSE, save_rot_angles = FALSE, rot_trace_every = 1, rot_diag_every = 0, rot_timing_every = 0, depth_prior_type = 0, depth_prior_tilt = 0.0, depth_prior_nu = 0.25, sample_depth_tilt = FALSE, depth_tilt_shape = 2.0, depth_tilt_rate = 8.0, depth_tilt_rw_sd = 0.15, sample_depth_nu = FALSE, depth_nu_a = 2.0, depth_nu_b = 2.0, depth_nu_rw_sd = 0.35, save_depth_hyper = FALSE, extra_ctrlSEXP = NULL) {
    .Call(`_robart_cgbart`, type, n, p, np, xv, yv, xpv, m, nc_vec, nd, burn, thin, mybeta, alpha, Offset, tau, nu, lambda, sigma, wv, dart_i, theta, omega, epsilon, K, grp_vec, a, b, rho, aug_i, printevery, Xinfo, rot_warmup_iters, init_random_rotation, tree_update_reps, debug, debug_every, verbose, save_alignment, save_rot_summary, save_rot_trace, save_rot_ll_trace, save_rot_part_trace, save_rot_angles, rot_trace_every, rot_diag_every, rot_timing_every, depth_prior_type, depth_prior_tilt, depth_prior_nu, sample_depth_tilt, depth_tilt_shape, depth_tilt_rate, depth_tilt_rw_sd, sample_depth_nu, depth_nu_a, depth_nu_b, depth_nu_rw_sd, save_depth_hyper, extra_ctrlSEXP)
}

cgbart_profile_snapshot <- function(xv, yv, wv, state, tau, theta_grid, tree_index, plane_a, plane_b, sigma_current, sigma_true = NA_real_) {
    .Call(`_robart_cgbart_profile_snapshot`, xv, yv, wv, state, tau, theta_grid, tree_index, plane_a, plane_b, sigma_current, sigma_true)
}

cwbart <- function(n, p, np, xv, yv, xpv, m, nc_vec, nd, burn, mybeta, alpha, tau, nu, lambda, sigma, wv, dart_i, theta, omega, grp_vec, a, b, rho, aug_i, nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws, printevery, Xinfo) {
    .Call(`_robart_cwbart`, n, p, np, xv, yv, xpv, m, nc_vec, nd, burn, mybeta, alpha, tau, nu, lambda, sigma, wv, dart_i, theta, omega, grp_vec, a, b, rho, aug_i, nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws, printevery, Xinfo)
}


# Internal coordinate evaluator; no RNG scope or model fitting.
cgbart_canonical_coordinates <- function(x,Q){
  .Call(`_robart_cgbart_canonical_coordinates`,x,Q)
}

# Deterministic internal audits, not alternative fitting entry points.
formal_tree_audit <- function(x, residual, sd, tree, grid, control = list()) {
  .Call("_robart_formal_tree_audit", x, residual, sd, tree, grid, control, PACKAGE = "robart")
}
formal_cut_audit <- function(x0, x1, grid0, grid1, tree0, tree1, touched, control = list()) {
  .Call("_robart_formal_cut_audit", x0, x1, grid0, grid1, tree0, tree1, touched, control, PACKAGE = "robart")
}
