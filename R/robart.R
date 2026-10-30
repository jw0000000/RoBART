# Convenience interface for centered, tree-specific rotation BART.
# Uses the shared gbart() -> cgbart() path; DART and response-type paths remain.
# This file never accepts truth coordinates or a reference regression function.

.robart_scalar <- function(x, name, lower = -Inf, upper = Inf,
                           integer = FALSE, missing_ok = FALSE,
                           open_lower = FALSE, open_upper = FALSE) {
  ok <- is.numeric(x) && length(x) == 1L && !is.nan(x)
  if (ok && missing_ok && is.na(x)) return(invisible(x))
  ok <- ok && is.finite(x)
  if (ok) ok <- if (open_lower) x > lower else x >= lower
  if (ok) ok <- if (open_upper) x < upper else x <= upper
  if (ok && integer) ok <- x == floor(x) && x <= .Machine$integer.max
  if (!isTRUE(ok)) stop(paste("Invalid", name), call. = FALSE)
  invisible(x)
}
# Logical flags use the single shared validator in input_contract.R.

#' Storage and diagnostic controls for roBART
#' @param rotation_summary Store rotation proposal and acceptance counters.
#' @param rotation_trace Store sampled per-tree rotation and tree-size traces.
#' @param trace_every Interval between saved rotation records; actual iteration numbers are returned.
#' @param diagnostic_every Interval between expensive partition diagnostics; zero disables them.
#' @param timing_every Interval between timing records; zero disables them. When positive, must be a multiple of trace_every and require a trace family.
#' @param likelihood_trace Store local rotation marginal-likelihood diagnostics.
#' @param partition_trace Store membership and leaf-summary changes around proposals.
#' @param angles Store canonical angles of saved frames, not an ordered proposal history.
#' @param global_likelihood_trace Store global Gaussian likelihood diagnostics around proposals.
#' @param iteration_trace Store iteration-level time and available target components.
#' @param snapshots Full-chain iteration numbers for tree/Q/grid snapshots. NULL uses burn-in end and final iteration; integer(0) stores none.
#' @param verbose Enable ordinary native progress output. Errors and warnings are not suppressed.
#' @param print_every Print cadence; NULL uses the convenience entry default.
#' @return A validated named control list. It does not run a model.
#' @export
robart_control <- function(rotation_summary = TRUE, rotation_trace = TRUE,
                           trace_every = 10L, diagnostic_every = 0L,
                           timing_every = 0L, likelihood_trace = FALSE,
                           partition_trace = FALSE, angles = FALSE,
                           global_likelihood_trace = FALSE,
                           iteration_trace = TRUE, snapshots = NULL,
                           verbose = FALSE, print_every = NULL) {
  flags <- c("rotation_summary", "rotation_trace", "likelihood_trace",
             "partition_trace", "angles", "global_likelihood_trace",
             "iteration_trace", "verbose")
  for (nm in flags) .robart_flag(get(nm), nm)
  .robart_scalar(trace_every, "trace_every", lower = 1, integer = TRUE)
  .robart_scalar(diagnostic_every, "diagnostic_every", lower = 0, integer = TRUE)
  .robart_scalar(timing_every, "timing_every", lower = 0, integer = TRUE)
  if (timing_every > 0 && !(rotation_trace || likelihood_trace || partition_trace || global_likelihood_trace))
    stop("timing_every requires a rotation trace family to be enabled.", call. = FALSE)
  if (timing_every > 0 && timing_every %% trace_every != 0)
    stop("timing_every must be a multiple of trace_every to preserve recorded timing iterations.", call. = FALSE)
  if (!is.null(print_every))
    .robart_scalar(print_every, "print_every", lower = 1, integer = TRUE)
  if (!is.null(snapshots)) {
    if (!is.numeric(snapshots) || anyNA(snapshots) || any(!is.finite(snapshots)) ||
        any(snapshots <= 0 | snapshots != floor(snapshots)) ||
        any(snapshots > .Machine$integer.max))
      stop("snapshots must be NULL or positive integer iteration numbers.", call. = FALSE)
    snapshots <- sort(unique(as.integer(snapshots)))
  }
  structure(list(rotation_summary = rotation_summary,
    rotation_trace = rotation_trace, trace_every = as.integer(trace_every),
    diagnostic_every = as.integer(diagnostic_every), timing_every = as.integer(timing_every),
    likelihood_trace = likelihood_trace, partition_trace = partition_trace,
    angles = angles, global_likelihood_trace = global_likelihood_trace,
    iteration_trace = iteration_trace, snapshots = snapshots, verbose = verbose,
    print_every = if (is.null(print_every)) NULL else as.integer(print_every)), class = "robart_control")
}

#' Gaussian or probit tree-specific rotation BART
#'
#' Convenience profile using gbart(): identity initialization, centered rotation
#' from iteration one, dynamic uniform grids when rotation is on, and fixed
#' uniform grids when it is off. DART is disabled in this convenience profile;
#' it remains available through gbart(). There is no hard depth cap.
#' @param x.train Finite numeric training matrix with observations in rows.
#' @param y.train Finite numeric response: nonconstant for wbart, 0/1 for pbart.
#' @param x.test Optional finite numeric test matrix with the same input column order.
#' @param ntree Number of trees; the convenience default remains 200 for both supported types.
#' @param numcut Requested candidate cutpoint count. Preprocessing can produce different per-coordinate counts; not the index-proposal width.
#' @param ndpost Number of retained prediction draws.
#' @param nskip Number of burn-in iterations before retained prediction draws. Distinct from rotation warm_up.
#' @param keepevery Keep one prediction draw per this many post-burn-in iterations.
#' @param K Number of ordered Givens planes in one centered rotation proposal. This is the sole active plane-count control.
#' @param epsilon Positive scale of the normal angle distribution before truncation.
#' @param theta_max Positive truncation bound for centered angles, strictly below pi-1e-8.
#' @param support_expand_prob Probability of trying a used/unused coordinate pair when support expansion is available. Not a second rotation kernel.
#' @param k Leaf shrinkage parameter. Lowercase k is distinct from the rotation plane count K.
#' @param power Power in the polynomial tree split prior base/(1+depth)^power; also used by the tilted-polynomial alternative.
#' @param base Root split probability in the polynomial and tilted-polynomial depth priors.
#' @param sigest Initial or calibration noise standard deviation. If unspecified, the inherited Gaussian path uses the linear-model residual estimate when p<n, otherwise sd(y). Not by itself a sigma-freezing option.
#' @param sigdf Degrees of freedom of the Gaussian noise-variance prior.
#' @param sigquant Calibration probability used in lambda=sigest^2*qchisq(1-sigquant,sigdf)/sigdf when lambda is unspecified.
#' @param lambda Gaussian noise-prior scale, NA or positive in this convenience interface. Use sigma_fixed here; the separate lambda=0 convention remains available in gbart().
#' @param tau.num Leaf scale numerator. NA uses the inherited Gaussian range rule. When omitted for pbart it is 3, giving tau=3/(k*sqrt(ntree)).
#' @param sigma_fixed NULL estimates sigma; a positive value fixes the noise SD from iteration one. This is a different conditional posterior.
#' @param rotation Enable centered rotation proposals. FALSE also uses identity initialization and a fixed uniform grid in this convenience profile.
#' @param seed NULL uses the caller RNG; otherwise set.seed is called before preprocessing. RNGkind is not changed.
#' @param control A robart_control() object defining output and diagnostics.
#' @param type Response model: wbart (Gaussian, default) or pbart (Bernoulli-probit).
#' @param offset NULL uses the inherited sample-mean/inverse-link rule. A supplied offset is a finite response-scale constant for wbart or link-scale constant for pbart.
#' @param output compact groups additional diagnostics/settings; flat preserves the flat field layout. Neither changes sampling.
#' @return A robart object with BART core fields and optional grouped diagnostics. predict() uses only inputs supplied during fitting; arbitrary newdata is not supported.
#' @export
robart <- function(x.train, y.train, x.test = NULL,
                   ntree = 200L, numcut = 100L,
                   ndpost = 1000L, nskip = 9000L, keepevery = 1L,
                   K = 5L, epsilon = 0.08,
                   theta_max = 0.35 * epsilon / 0.15,
                   support_expand_prob = 0.5,
                   k = 2, power = 2, base = 0.95,
                   sigest = NA_real_, sigdf = 3, sigquant = 0.90,
                   lambda = NA_real_, tau.num = NA_real_,
                   sigma_fixed = NULL, rotation = TRUE, seed = NULL,
                   control = robart_control(),
                   type = "wbart", offset = NULL,
                   output = c("compact", "flat")) {
  if (!is.character(type) || length(type) != 1L || is.na(type) ||
      !type %in% c("wbart", "pbart"))
    stop("robart() currently supports type='wbart' and type='pbart'; logit remains a separate validation task.", call. = FALSE)
  output <- match.arg(output)
  if (type == "pbart") {
    for (nm in c("sigest", "lambda"))
      .robart_scalar(get(nm), nm, lower = 0, open_lower = TRUE, missing_ok = TRUE)
    if (!is.null(sigma_fixed) || !is.na(sigest) || !is.na(lambda) ||
        !identical(as.double(sigdf), 3) || !identical(as.double(sigquant), .90))
      stop("sigma_fixed, sigest, lambda, sigdf and sigquant are Gaussian noise controls, not probit controls.", call. = FALSE)
    if (missing(tau.num)) tau.num <- 3
    if (!is.numeric(tau.num) || length(tau.num) != 1L ||
        !is.finite(tau.num) || tau.num <= 0)
      stop("Probit tau.num must be positive; its default is 3.", call. = FALSE)
  }
  if (!is.null(offset)) .robart_scalar(offset, "offset")
  # Strict numeric-input contract avoids silent factor encodings or stochastic
  # missing-value imputation. The shared gbart retains its inherited input interface.
  if (!is.matrix(x.train) || !is.numeric(x.train) || nrow(x.train) < 2L ||
      ncol(x.train) < 1L || anyNA(x.train) || any(!is.finite(x.train)))
    stop("x.train must be a finite numeric matrix with at least two rows.", call. = FALSE)
  if (!is.numeric(y.train) || !is.null(dim(y.train)) ||
      length(y.train) != nrow(x.train) || anyNA(y.train) || any(!is.finite(y.train)))
    stop("y.train must be a finite numeric vector matching x.train.", call. = FALSE)
  if (type == "wbart") {
    if (stats::sd(y.train) <= 0)
      stop("The Gaussian response must be nonconstant.", call. = FALSE)
    if (length(unique(y.train)) == 2L)
      stop("Two-valued responses are not supported by this Gaussian entry; use type='pbart' for 0/1 data.", call. = FALSE)
  } else {
    if (any(!y.train %in% c(0, 1)))
      stop("The probit response must contain only 0 and 1.", call. = FALSE)
    if (is.null(offset) && length(unique(y.train)) < 2L)
      stop("A one-class probit response requires an explicit finite link-scale offset.", call. = FALSE)
  }
  variable <- vapply(seq_len(ncol(x.train)), function(j)
    min(x.train[, j]) < max(x.train[, j]), logical(1L))
  if (!any(variable)) stop("All training predictors are constant.", call. = FALSE)
  if (!is.null(x.test) && (!is.matrix(x.test) || !is.numeric(x.test) ||
      ncol(x.test) != ncol(x.train) || anyNA(x.test) || any(!is.finite(x.test))))
    stop("x.test must be NULL or a finite numeric matrix with matching columns.", call. = FALSE)

  for (nm in c("ntree", "numcut", "ndpost", "keepevery", "K"))
    .robart_scalar(get(nm), nm, lower = 1, integer = TRUE)
  .robart_scalar(nskip, "nskip", lower = 0, integer = TRUE)
  total <- nskip + ndpost * keepevery
  .robart_scalar(total, "total iterations", lower = 1,
                 upper = .Machine$integer.max - 1, integer = TRUE)
  for (nm in c("epsilon", "k", "sigdf"))
    .robart_scalar(get(nm), nm, lower = 0, open_lower = TRUE)
  .robart_scalar(theta_max, "theta_max", lower = 0, upper = pi - 1e-8,
                 open_lower = TRUE, open_upper = TRUE)
  .robart_scalar(support_expand_prob, "support_expand_prob", lower = 0, upper = 1)
  .robart_scalar(power, "power", lower = 0)
  .robart_scalar(base, "base", lower = 0, upper = 1, open_lower = TRUE, open_upper = TRUE)
  .robart_scalar(sigquant, "sigquant", lower = 0, upper = 1, open_lower = TRUE, open_upper = TRUE)
  for (nm in c("sigest", "lambda", "tau.num"))
    .robart_scalar(get(nm), nm, lower = 0, open_lower = TRUE, missing_ok = TRUE)
  .robart_flag(rotation, "rotation")
  if (!is.null(sigma_fixed))
    .robart_scalar(sigma_fixed, "sigma_fixed (standard deviation)", lower = 0, open_lower = TRUE)
  if (!is.null(seed)) .robart_scalar(seed, "seed", lower = 0, integer = TRUE)

  expected <- names(robart_control())
  if (!is.list(control) || anyDuplicated(names(control)) ||
      !identical(sort(names(control)), sort(expected)))
    stop("Use robart_control() to construct the control argument.", call. = FALSE)
  control <- do.call(robart_control, unclass(control))
  snapshots <- control$snapshots
  if (is.null(snapshots)) snapshots <- sort(unique(as.integer(c(nskip, total))))
  snapshots <- snapshots[snapshots > 0L]
  if (any(snapshots > total)) stop("A requested snapshot exceeds total iterations.", call. = FALSE)
  print_every <- control$print_every
  if (is.null(print_every)) print_every <- max(1L, as.integer(total %/% 20L))

  # No automatic RNGkind change. Record and honor the caller's RNG algorithm.
  if (!is.null(seed)) set.seed(as.integer(seed))
  rng_kind <- RNGkind()
  rng_before <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL
  if (is.null(x.test) || nrow(x.test) == 0L) x.test <- matrix(0, 0L, 0L)

  fit <- gbart(x.train = x.train, y.train = y.train, x.test = x.test,
    type = type, offset = offset, ntree = as.integer(ntree), numcut = as.integer(numcut),
    ndpost = as.integer(ndpost), nskip = as.integer(nskip), keepevery = as.integer(keepevery),
    k = k, power = power, base = base, sigest = sigest, sigdf = sigdf,
    sigquant = sigquant, lambda = lambda, tau.num = tau.num,
    sparse = FALSE, augment = FALSE, usequants = FALSE, rm.const = TRUE,
    initial_rotation = FALSE, warm_up = 0L, rotation_enabled = rotation,
    K = as.integer(K), epsilon = epsilon, theta_max = theta_max,
    retune_cutpoints = TRUE,
    support_expand_prob = support_expand_prob,
    dynamic_xinfo_on_rotation = rotation, rotation_xinfo_policy = "uniform",
    rotation_only_after_warmup = FALSE,
    tree_update_reps = 1L, depth_prior = "poly",
    sample_depth_tilt = FALSE, sample_depth_nu = FALSE, save_depth_hyper = FALSE,
    # Fixed sigma must apply from iteration 1 even when rotations are OFF.
    freeze_sigma_after_warmup = !is.null(sigma_fixed),
    freeze_sigma_value = if (is.null(sigma_fixed)) NA_real_ else sigma_fixed,
    save_rot_summary = control$rotation_summary,
    save_rot_trace = control$rotation_trace, rot_trace_every = control$trace_every,
    rot_diag_every = control$diagnostic_every, rot_timing_every = control$timing_every,
    save_rot_ll_trace = control$likelihood_trace, save_rot_part_trace = control$partition_trace,
    save_rot_angles = control$angles, save_rot_global_ll_trace = control$global_likelihood_trace,
    save_iteration_trace = control$iteration_trace, save_state_iters = snapshots,
    verbose = control$verbose, printevery = print_every,
    compute_lpml = FALSE)

  # Validate the immediate fixed-sigma contract against the returned samples.
  if (!is.null(sigma_fixed) &&
      (is.null(fit$sigma) || any(fit$sigma != sigma_fixed)))
    stop("Native fixed-sigma contract failed; do not use this fit.", call. = FALSE)

  fit$draw_iterations <- as.integer(nskip + seq_len(ndpost) * keepevery)
  fit$iteration <- seq_len(total)
  fit$settings <- list(profile = if (type == "wbart") "formal-uniform-gaussian-stage5A" else "formal-uniform-probit-stage5A",
    sampler = "full-discrete-target-stage5A", discrete_prior = "split-attempt-full-terminal-v1", type = type,
    link = if (type == "wbart") "identity" else "probit",
    response_unit = if (type == "wbart") "response" else "probability",
    n_train = nrow(x.train), n_test = nrow(x.test),
    rotation = rotation, ntree = as.integer(ntree), numcut = as.integer(numcut),
    K = as.integer(K), epsilon = epsilon, theta_max = theta_max,
    support_expand_prob = support_expand_prob, hard_depth_cap = FALSE, min_child_size_birth = 0L,
    depth_prior = "poly", base = base, power = power, k = k,
    cutpoint_proposal = "single_scale_discrete_gaussian",
    cutpoint_scale_fraction = 0.02, cutpoint_scale_minimum = 1.0,
    retune_cutpoints = TRUE, grid = fit$grid_mode,
    grid_mode_selection = fit$grid_mode_selection, custom_xinfo = fit$user_xinfo_supplied,
    sigma_fixed = if (type == "wbart") sigma_fixed else NULL,
    sigma_prior_df = if (type == "wbart") fit$sigma_prior_df else NULL,
    sigma_prior_scale = if (type == "wbart") fit$sigma_prior_scale else NULL,
    tau = fit$tau, offset = fit$offset,
    nskip = as.integer(nskip), ndpost = as.integer(ndpost), keepevery = as.integer(keepevery),
    seed = seed, rng_kind = rng_kind, input_p = ncol(x.train), retained_columns = fit$rm.const,
    control = control, base_entry = "gbart")
  fit$rng_state_before <- rng_before
  fit$rng_state_after <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL
  class(fit) <- c("robart", type)
  if (output == "compact") .robart3b_pack(fit) else fit
}

#' @export
print.robart <- function(x, ...) {
  print(summary.robart(x), ...)
  invisible(x)
}

#' @export
predict.robart <- function(object, newdata = NULL, type = c("mean", "draws"), ...,
                           scale = c("response", "link"), data = c("test", "train")) {
  type <- match.arg(type)
  scale <- match.arg(scale)
  data <- match.arg(data)
  if (length(list(...))) stop("Unused prediction arguments were supplied.", call. = FALSE)
  if (!is.null(newdata))
    stop(paste("Arbitrary newdata prediction is not implemented.",
      "Supply x.test during fitting. Rotated prediction requires each draw's tree, Q and grid;",
      "the legacy static predictor is not used."), call. = FALSE)
  model <- .robart3b_model(object)
  prefix <- if (model == "pbart" && scale == "response") "prob" else "yhat"
  nm <- paste(prefix, data, sep = ".")
  d <- object[[nm, exact = TRUE]]
  if (!is.matrix(d) || nrow(d) == 0L || ncol(d) == 0L)
    stop(paste("No", data, "predictions were saved. Supply the corresponding inputs during fitting."), call. = FALSE)
  if (type == "draws") return(d)
  ans <- object[[paste0(nm, ".mean"), exact = TRUE]]
  if (is.null(ans)) ans <- colMeans(d)
  if (length(ans) != ncol(d)) stop("Prediction summary shape is inconsistent.", call. = FALSE)
  ans
}
