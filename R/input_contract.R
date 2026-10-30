# Input checks for the shared gbart -> cgbart entry. No RNG calls or imputation
# policy changes are made here. The native boundary checks the resolved inputs.
.robart_scalar_number <- function(x, name, lower = -Inf, upper = Inf,
                                  lower_open = FALSE, upper_open = FALSE,
                                  allow_na = FALSE) {
  if (allow_na && length(x) == 1L && is.null(dim(x)) &&
      (is.numeric(x) || is.logical(x)) && is.na(x) && !is.nan(x))
    return(invisible(TRUE))
  if (!is.numeric(x) || length(x) != 1L || !is.null(dim(x)))
    stop(paste0(name, " must be one numeric value."), call. = FALSE)
  ok <- is.finite(x) && (if (lower_open) x > lower else x >= lower)
  ok <- ok && (if (upper_open) x < upper else x <= upper)
  if (!ok) stop(paste0(name, " is outside its finite allowed range."), call. = FALSE)
  invisible(TRUE)
}

.robart_integer_count <- function(x, name, zero = FALSE) {
  .robart_scalar_number(x, name, lower = if (zero) 0 else 1,
                        upper = .Machine$integer.max)
  if (x != floor(x)) stop(paste0(name, " must be an integer."), call. = FALSE)
  invisible(TRUE)
}

.robart_flag <- function(x, name, allow_na = FALSE) {
  if (!is.logical(x) || length(x) != 1L || (!allow_na && is.na(x)))
    stop(paste0(name, " must be TRUE or FALSE", if (allow_na) " or NA" else "", "."), call. = FALSE)
  invisible(TRUE)
}

.robart_validate_raw_design <- function(x, name, transposed = FALSE, allow_empty = FALSE) {
  if (allow_empty && length(x) == 0L) return(invisible(TRUE))
  if (transposed) {
    if (!is.matrix(x) || !is.numeric(x) || nrow(x) < 1L || ncol(x) < 1L)
      stop(paste0(name, " must be a nonempty numeric p-by-n matrix when transposed=TRUE."), call. = FALSE)
    columns <- lapply(seq_len(nrow(x)), function(j) x[j, ])
  } else if (is.factor(x) || (is.numeric(x) && is.null(dim(x)))) {
    columns <- list(x)
  } else if (is.data.frame(x)) {
    if (nrow(x) < 1L || ncol(x) < 1L)
      stop(paste0(name, " must have rows and predictors."), call. = FALSE)
    columns <- unname(x)
  } else if (is.matrix(x) && is.numeric(x) && nrow(x) > 0L && ncol(x) > 0L) {
    columns <- lapply(seq_len(ncol(x)), function(j) x[, j])
  } else stop(paste0(name, " must be a numeric vector/matrix, factor, or supported data.frame."), call. = FALSE)
  for (j in seq_along(columns)) {
    z <- columns[[j]]
    if (!(is.numeric(z) || is.factor(z)))
      stop(paste0(name, " predictor ", j, " must be numeric or a factor."), call. = FALSE)
    if (!length(z) || all(is.na(z)))
      stop(paste0(name, " predictor ", j, " has no observed values; hot-deck imputation cannot terminate."), call. = FALSE)
    if (is.numeric(z) && any(is.infinite(z)))
      stop(paste0(name, " contains infinite values."), call. = FALSE)
  }
  invisible(TRUE)
}

.robart_validate_gbart_arguments <- function(env) {
  v <- function(name) get(name, envir = env, inherits = FALSE)
  type <- v("type")
  if (!is.character(type) || length(type) != 1L || is.na(type) ||
      !type %in% c("wbart", "pbart", "lbart"))
    stop("type must be 'wbart', 'pbart' or 'lbart'.", call. = FALSE)
  .robart_integer_count(v("ntype"), "ntype")
  if (v("ntype") != match(type, c("wbart", "pbart", "lbart")))
    stop("type and ntype disagree.", call. = FALSE)
  for (nm in c("ntree", "ndpost", "keepevery", "printevery", "tree_update_reps", "K"))
    .robart_integer_count(v(nm), nm)
  for (nm in c("nskip", "warm_up", "debug_every", "rot_diag_every", "rot_timing_every"))
    .robart_integer_count(v(nm), nm, zero = TRUE)
  .robart_integer_count(v("rot_trace_every"), "rot_trace_every")
  if (v("warm_up") >= .Machine$integer.max ||
      v("ndpost") > floor((.Machine$integer.max - v("nskip")) / v("keepevery")))
    stop("Iteration budget or warm_up+1 exceeds the supported integer range.", call. = FALSE)
  for (nm in c("transposed", "rm.const", "usequants", "sparse", "augment", "initial_rotation",
               "rotation_enabled", "retune_cutpoints", "debug", "verbose", "hostname",
               "save_alignment", "save_rot_summary", "save_rot_trace", "save_rot_ll_trace",
               "save_rot_part_trace", "save_rot_angles", "save_rot_global_ll_trace",
               "save_iteration_trace", "rotation_only_after_warmup",
               "freeze_sigma_after_warmup", "sample_depth_tilt", "sample_depth_nu", "compute_lpml"))
    .robart_flag(v(nm), nm)
  .robart_flag(v("save_depth_hyper"), "save_depth_hyper", allow_na = TRUE)
  has_trace_family <- any(vapply(c("save_rot_trace", "save_rot_ll_trace",
                                  "save_rot_part_trace", "save_rot_global_ll_trace"),
                                v, logical(1), USE.NAMES = FALSE))
  .robart_validate_recording_controls(
    v("rot_trace_every"), v("rot_timing_every"), has_trace_family,
    v("save_state_iters"),
    as.double(v("nskip")) + as.double(v("ndpost")) * as.double(v("keepevery")))
  for (nm in c("k", "sigdf", "a", "b", "omega", "epsilon"))
    .robart_scalar_number(v(nm), nm, lower = 0, lower_open = TRUE)
  .robart_scalar_number(v("theta"), "theta", lower = 0)
  if (length(v("rho"))) .robart_scalar_number(v("rho"), "rho", lower = 0, lower_open = TRUE)
  .robart_scalar_number(v("base"), "base", lower = 0, upper = 1, lower_open = TRUE, upper_open = TRUE)
  .robart_scalar_number(v("power"), "power", lower = 0)
  .robart_scalar_number(v("sigquant"), "sigquant", lower = 0, upper = 1, lower_open = TRUE, upper_open = TRUE)
  .robart_scalar_number(v("support_expand_prob"), "support_expand_prob", lower = 0, upper = 1)
  .robart_scalar_number(v("theta_max"), "theta_max", lower = 0, upper = pi - 1e-8, lower_open = TRUE, upper_open = TRUE)
  for (nm in c("sigest", "tau.num", "freeze_sigma_value", "true_sigma_for_trace"))
    .robart_scalar_number(v(nm), nm, lower = 0, lower_open = TRUE, allow_na = TRUE)
  .robart_scalar_number(v("lambda"), "lambda", lower = 0, allow_na = TRUE)
  if (!is.na(v("lambda")) && v("lambda") == 0 && is.na(v("sigest")))
    stop("lambda=0 requires a finite positive sigest (known sigma).", call. = FALSE)
  # An explicitly supplied binary offset is on the link scale, not a probability.
  # The inverse-link of mean(y) is used only for an omitted offset in gbart().
  if (length(v("offset"))) .robart_scalar_number(v("offset"), "offset")
  y <- v("y.train")
  if (!(is.numeric(y) || is.logical(y)) || !is.null(dim(y)) ||
      !length(y) || any(!is.finite(y)))
    stop("y.train must be a nonempty finite numeric response vector.", call. = FALSE)
  if (type != "wbart" && any(!y %in% c(0, 1)))
    stop("Binary y.train must contain only 0 and 1.", call. = FALSE)
  if (type != "wbart" && !length(v("offset")) && (all(y == 0) || all(y == 1)))
    stop("Single-class binary data require an explicit finite link-scale offset; the automatic offset would be infinite.",
         call. = FALSE)
  w <- v("w")
  if (!is.numeric(w) || !is.null(dim(w)) || length(w) != length(y) ||
      any(!is.finite(w)) || any(w <= 0))
    stop("w must contain one finite positive observation SD multiplier per response.", call. = FALSE)
  nc <- v("numcut")
  if (!is.numeric(nc) || !length(nc) || any(!is.finite(nc)) ||
      any(nc < 1 | nc != floor(nc) | nc > .Machine$integer.max))
    stop("numcut must contain positive integer candidate counts.", call. = FALSE)
  if (!v("transposed") && length(nc) != 1L)
    stop("Non-transposed preprocessing requires a scalar numcut; use xinfo for per-predictor candidates.", call. = FALSE)
  .robart_validate_raw_design(v("x.train"), "x.train", v("transposed"))
  .robart_validate_raw_design(v("x.test"), "x.test", v("transposed"), allow_empty = TRUE)
  x <- v("x.train")
  n <- if (v("transposed")) ncol(x) else if (is.null(dim(x))) length(x) else nrow(x)
  if (n != length(y)) stop("Training rows and response length disagree.", call. = FALSE)
  invisible(TRUE)
}

.robart_validate_processed_design <- function(x, xt, y, w, nc) {
  if (!is.matrix(x) || !is.numeric(x) || nrow(x) < 1L || ncol(x) < 1L)
    stop("No usable predictor remains after preprocessing.", call. = FALSE)
  if (!is.matrix(xt) || !is.numeric(xt) ||
      (length(xt) > 0L && nrow(xt) != nrow(x)))
    stop("Expanded training/test predictor dimensions disagree.", call. = FALSE)
  if (length(y) != ncol(x) || length(w) != ncol(x))
    stop("Processed response/weight dimensions disagree.", call. = FALSE)
  if (length(nc) != nrow(x) || anyNA(nc) || any(nc < 1 | nc != floor(nc)))
    stop("Every retained predictor needs a positive candidate count.", call. = FALSE)
  invisible(TRUE)
}

.robart_validate_effective_scales <- function(tau, sigma, lambda, offset, y, w, x, xt) {
  .robart_scalar_number(tau, "effective leaf scale tau", lower = 0, lower_open = TRUE)
  .robart_scalar_number(sigma, "effective initial sigma", lower = 0, lower_open = TRUE)
  .robart_scalar_number(lambda, "effective lambda", lower = 0)
  .robart_scalar_number(offset, "effective offset")
  if (any(!is.finite(y)) || any(!is.finite(x)) || any(!is.finite(xt)))
    stop("Native inputs must be finite after preprocessing and imputation.", call. = FALSE)
  if (any(!is.finite(sigma * w)) || any(sigma * w <= 0))
    stop("Initial observation SDs overflow or underflow; check sigest and w.", call. = FALSE)
  invisible(TRUE)
}

# NULL and a length-zero numeric vector request no gbart snapshots. The
# convenience entry resolves its own NULL default before it calls gbart().
.robart_validate_recording_controls <- function(trace_every, timing_every,
                                                has_trace_family,
                                                snapshots, total) {
  .robart_integer_count(trace_every, "rot_trace_every")
  .robart_integer_count(timing_every, "rot_timing_every", zero = TRUE)
  .robart_flag(has_trace_family, "has_trace_family")
  .robart_integer_count(total, "total iterations")
  if (timing_every > 0L && !has_trace_family)
    stop("rot_timing_every requires a rotation trace family to be enabled.", call. = FALSE)
  if (timing_every > 0L && timing_every %% trace_every != 0)
    stop("rot_timing_every must be a multiple of rot_trace_every.", call. = FALSE)
  if (!is.null(snapshots) &&
      (!is.numeric(snapshots) || !is.null(dim(snapshots)) ||
       anyNA(snapshots) || any(!is.finite(snapshots)) ||
       any(snapshots < 1 | snapshots != floor(snapshots) | snapshots > total)))
    stop("save_state_iters must be NULL or positive integer iterations not exceeding total iterations.", call. = FALSE)
  invisible(TRUE)
}
