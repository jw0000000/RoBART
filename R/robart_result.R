# Presentation only. These helpers neither fit a model nor consume random numbers.
# Native and gbart fields are preserved, including NULL and non-applicable values.

.robart3b_model <- function(object) {
  s <- object[["settings", exact = TRUE]]
  type <- if (is.list(s)) s[["type", exact = TRUE]] else NULL
  if (is.null(type)) {
    type <- if (inherits(object, "pbart")) "pbart" else if (inherits(object, "wbart")) "wbart" else NULL
  }
  if (!is.character(type) || length(type) != 1L || !type %in% c("wbart", "pbart"))
    stop("This accessor requires a Gaussian or probit roBART result.", call. = FALSE)
  type
}

.robart3b_destination <- function(name, model) {
  core <- c("yhat.train", "yhat.train.mean", "yhat.test", "yhat.test.mean",
    "prob.train", "prob.train.mean", "prob.test", "prob.test.mean", "sigma", "sigma.mean",
    "LPML", "varcount", "varcount.mean", "varprob", "varprob.mean", "treedraws",
    "ndpost", "offset", "rm.const", "w", "proc.time", "draw_iterations", "iteration")
  if (name %in% core) return("core")
  if (name == "settings") return("settings")
  if (name %in% c("saved_states", "save_state_iters")) return("states")
  if (model == "pbart" && (name %in% c("sigma_last", "sigma_prior_df", "sigma_prior_scale",
      "true_sigma_for_trace", "freeze_sigma_after_warmup", "freeze_sigma_value") ||
      grepl("^global_", name))) return("metadata.not_applicable")
  if (grepl("^align_", name)) return("diagnostics.alignment")
  if (grepl("^rot_(time|timing|total_time)", name) || name == "iteration_elapsed_sec_trace")
    return("diagnostics.timing")
  if (grepl("^global_", name)) return("diagnostics.likelihood")
  if (grepl("^rot_", name) || name == "nbots_trace_full") return("diagnostics.rotation")
  if (grepl("^dart_", name)) return("diagnostics.dart")
  if (grepl("^depth_", name) && grepl("trace|hyper|last$", name)) return("diagnostics.depth")
  if (name == "sigma_last") return("diagnostics.sigma")
  if (grepl("^(compiled_|native_|rng_|sampler_|output_|snapshot_|trace_|test_buffer|train_buffer)", name) ||
      name %in% c("hostname", "robart_version", "robart_schema_version", "schema_version", "rotation_test_storage", "rotation_buffer_storage")) return("metadata.runtime")
  config <- c("tau", "sigma_prior_df", "sigma_prior_scale", "true_sigma_for_trace",
    "freeze_sigma_after_warmup", "freeze_sigma_value", "dynamic_xinfo_on_rotation",
    "grid_mode", "grid_mode_automatic", "grid_mode_selection", "grid_policy_requested",
    "hard_depth_cap", "min_child_size_birth", "depth_prior_type", "cutpoint_proposal",
    "cutpoint_scale_fraction", "cutpoint_scale_minimum", "cutpoint_center", "discrete_prior",
    "cutpoint_degenerate_rule", "tree_update_target", "augment_proposal", "user_xinfo_supplied", "support_expand_prob")
  if (name %in% config || grepl("^rotation_", name)) return("settings.native")
  # Future fields stay available rather than silently disappearing.
  "metadata.extra"
}

.robart3b_pack <- function(object) {
  if (identical(attr(object, "robart_result_format", exact = TRUE), "compact-v1")) return(object)
  if (!is.list(object) || is.null(names(object)) || anyNA(names(object)) ||
      any(names(object) == "") || anyDuplicated(names(object)))
    stop("A result must be a uniquely named list before grouping fields.", call. = FALSE)
  reserved <- intersect(c("diagnostics", "states", "metadata"), names(object))
  if (length(reserved)) stop("Result already contains reserved grouped fields: ", paste(reserved, collapse = ", "), call. = FALSE)
  model <- .robart3b_model(object)
  settings <- object[["settings", exact = TRUE]]
  if (!is.list(settings) || is.null(names(settings)) || anyDuplicated(names(settings)) || "native" %in% names(settings))
    stop("The convenience settings list is missing, duplicated, or uses the reserved name 'native'.", call. = FALSE)
  dest <- vapply(names(object), .robart3b_destination, character(1L), model = model, USE.NAMES = FALSE)
  out <- object[dest == "core"]
  original.settings.names <- names(settings)
  settings$native <- object[dest == "settings.native"]
  out$settings <- settings
  diag.names <- c("rotation", "likelihood", "timing", "alignment", "dart", "depth", "sigma")
  out$diagnostics <- stats::setNames(lapply(diag.names, function(x)
    object[dest == paste0("diagnostics.", x)]), diag.names)
  out$states <- object[dest == "states"]
  attrs <- attributes(object)
  attrs$names <- NULL
  metadata <- list(
    api_version = "1.0.0",
    runtime = object[dest == "metadata.runtime"],
    not_applicable = object[dest == "metadata.not_applicable"],
    extra = object[dest == "metadata.extra"],
    meaning = list(
      sigma = if (model == "wbart") "Gaussian full-iteration SD trace when supplied by native; no synthetic known-sigma trace."
              else "Probit latent SD is fixed at 1; there is no Gaussian residual-sigma posterior.",
      probability = if (model == "pbart") "prob fields transform each latent-predictor draw by pnorm before averaging." else NULL,
      local_rotation_score = "Leaf-integrated likelihood conditional on the working response and SD, not an observed Bernoulli likelihood.",
      global_trace = if (model == "pbart") "Inherited Gaussian global target fields are non-applicable and remain archived without relabelling."
                     else "Tree prior includes all internal-coordinate/cut and terminal factors, conditional on coordinate/depth hyperparameters.",
      prediction = "Only training inputs and x.test supplied during fitting; arbitrary newdata is not implemented.",
      grid_scope = "This convenience entry uses uniform grids. Advanced gbart quantile snapshots require the versioned ordered_dot_v1 evaluator; grouping alone does not evaluate coordinates."),
    layout = list(version = 1L, fields = names(object), destinations = dest,
      settings_names = original.settings.names, attributes = attrs))
  out$metadata <- metadata
  class(out) <- class(object)
  attr(out, "robart_result_format") <- "compact-v1"
  out
}

#' Restore the flat roBART result layout
#' @param object A compact or flat result returned by robart().
#' @return A flat list retaining every original field, class and value. No fitting or RNG use.
#' @export
robart_flat <- function(object) {
  if (!is.list(object)) stop("object must be a roBART result list.", call. = FALSE)
  if (!identical(attr(object, "robart_result_format", exact = TRUE), "compact-v1")) return(object)
  metadata <- object[["metadata", exact = TRUE]]
  layout <- metadata[["layout", exact = TRUE]]
  if (!is.list(layout) || !identical(layout[["version", exact = TRUE]], 1L))
    stop("Unknown or missing compact result layout.", call. = FALSE)
  fields <- layout[["fields", exact = TRUE]]
  dest <- layout[["destinations", exact = TRUE]]
  if (length(fields) != length(dest) || anyDuplicated(fields)) stop("Corrupted result layout.", call. = FALSE)
  get.exact <- function(x, nm) {
    if (!is.list(x) || !nm %in% names(x)) stop("Required result field was removed: ", nm, call. = FALSE)
    x[[nm, exact = TRUE]]
  }
  out <- stats::setNames(vector("list", length(fields)), fields)
  for (i in seq_along(fields)) {
    nm <- fields[i]
    where <- dest[i]
    if (where == "core") {
      value <- get.exact(object, nm)
    } else if (where == "settings") {
      st <- get.exact(object, "settings")
      sn <- layout[["settings_names", exact = TRUE]]
      if (!all(sn %in% names(st))) stop("Original convenience settings were removed.", call. = FALSE)
      value <- st[sn]
    } else {
      parts <- strsplit(where, ".", fixed = TRUE)[[1L]]
      branch <- get.exact(object, parts[1L])
      if (length(parts) == 2L) branch <- get.exact(branch, parts[2L])
      value <- get.exact(branch, nm)
    }
    # Single-bracket assignment retains a stored NULL field.
    out[i] <- list(value)
  }
  attrs <- layout[["attributes", exact = TRUE]]
  attrs$names <- fields
  attributes(out) <- attrs
  out
}

#' Extract a plotted series without refitting
#' @param object A result returned by robart(), compact or flat.
#' @param variable auto selects Gaussian sigma or probit probability. Other choices are sigma, probability and latent.
#' @param index A previously chosen column of the saved input predictions.
#' @param data Saved test or training predictions.
#' @param include_warmup For sigma, include all native iterations; FALSE selects only the retained prediction iterations.
#' @return A two-column data.frame with actual full-chain iteration and value. Prediction traces only contain retained draws.
#' @export
robart_trace <- function(object, variable = c("auto", "sigma", "probability", "latent"),
                         index = 1L, data = c("test", "train"), include_warmup = TRUE) {
  variable <- match.arg(variable)
  data <- match.arg(data)
  model <- .robart3b_model(object)
  .robart_flag(include_warmup, "include_warmup")
  if (variable == "auto") variable <- if (model == "wbart") "sigma" else "probability"
  if (variable == "sigma") {
    if (model != "wbart") stop("Probit does not estimate a Gaussian residual sigma. Use a probability or latent trace.", call. = FALSE)
    values <- object[["sigma", exact = TRUE]]
    if (!is.numeric(values) || !length(values))
      stop("No sigma trace was stored; a known scalar sigma is not fabricated into a posterior trace.", call. = FALSE)
    iterations <- object[["iteration", exact = TRUE]]
    if (is.null(iterations)) iterations <- seq_along(values)
    if (length(iterations) != length(values)) stop("Sigma and iteration lengths differ.", call. = FALSE)
    if (!include_warmup) {
      keep <- object[["draw_iterations", exact = TRUE]]
      if (!length(keep) || any(!keep %in% iterations)) stop("Retained iteration metadata is unavailable or inconsistent.", call. = FALSE)
      values <- values[match(keep, iterations)]
      iterations <- keep
    }
  } else {
    if (variable == "probability" && model != "pbart") stop("Probability traces require a probit result.", call. = FALSE)
    .robart_scalar(index, "index", lower = 1, integer = TRUE)
    prefix <- if (variable == "probability") "prob" else "yhat"
    values <- object[[paste(prefix, data, sep = "."), exact = TRUE]]
    if (!is.matrix(values) || index > ncol(values) || nrow(values) == 0L)
      stop("The requested saved prediction column is unavailable.", call. = FALSE)
    values <- values[, as.integer(index)]
    iterations <- object[["draw_iterations", exact = TRUE]]
    if (is.null(iterations) || length(iterations) != length(values))
      stop("Full-chain prediction iteration metadata is unavailable. Do not guess burn-in or thinning.", call. = FALSE)
  }
  ans <- data.frame(iteration = iterations, value = unname(values))
  attr(ans, "quantity") <- variable
  attr(ans, "data") <- if (variable == "sigma") "chain" else data
  ans
}

#' @export
summary.robart <- function(object, ...) {
  if (length(list(...))) stop("Unused summary arguments were supplied.", call. = FALSE)
  s <- object[["settings", exact = TRUE]]
  if (!is.list(s)) stop("Convenience settings are unavailable.", call. = FALSE)
  model <- .robart3b_model(object)
  answer <- list(type = model, link = if (model == "wbart") "identity" else "probit",
    ntree = s[["ntree", exact = TRUE]], nskip = s[["nskip", exact = TRUE]],
    ndpost = s[["ndpost", exact = TRUE]], keepevery = s[["keepevery", exact = TRUE]],
    rotation = s[["rotation", exact = TRUE]], K = s[["K", exact = TRUE]],
    epsilon = s[["epsilon", exact = TRUE]], grid = s[["grid", exact = TRUE]],
    offset = object[["offset", exact = TRUE]],
    result_format = if (identical(attr(object, "robart_result_format", exact = TRUE), "compact-v1")) "compact" else "flat")
  if (model == "wbart") {
    fixed <- s[["sigma_fixed", exact = TRUE]]
    answer$sigma <- list(mode = if (is.null(fixed)) "estimated" else "fixed",
      fixed_value = fixed, mean = object[["sigma.mean", exact = TRUE]])
  } else answer$latent_sd <- 1
  prefix <- if (model == "pbart") "prob" else "yhat"
  for (data in c("train", "test")) {
    d <- object[[paste(prefix, data, sep = "."), exact = TRUE]]
    ans <- object[[paste(prefix, data, "mean", sep = "."), exact = TRUE]]
    if (is.matrix(d) && ncol(d) > 0L) {
      if (is.null(ans)) ans <- colMeans(d)
      answer[[paste0(data, "_prediction")]] <- list(n = ncol(d), mean_range = range(ans))
    }
  }
  class(answer) <- "summary.robart"
  answer
}

#' @export
print.summary.robart <- function(x, ...) {
  if (length(list(...))) stop("Unused print arguments were supplied.", call. = FALSE)
  cat(if (x$type == "wbart") "roBART: Gaussian regression\n" else "roBART: Bernoulli-probit regression\n")
  cat("Trees:", x$ntree, " Retained draws:", x$ndpost, " Burn-in:", x$nskip, " Thin:", x$keepevery, "\n")
  cat("Rotation:", x$rotation, " K:", x$K, " epsilon:", x$epsilon, " Grid:", x$grid, "\n")
  if (x$type == "wbart") {
    cat("Sigma:", x$sigma$mode, " Retained mean:", x$sigma$mean, "\n")
  } else {
    cat("Probit latent SD: 1 (fixed); predictions are draw-wise posterior probabilities.\n")
  }
  cat("Output:", x$result_format, "; only saved training/test inputs are supported.\n")
  invisible(x)
}
