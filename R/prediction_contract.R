# The inherited static predictor has no per-draw frame/grid representation.
# Preserve it for legacy wbart/cwbart results, but never send a cgbart result there.
.robart_require_static_prediction <- function(object, newdata) {
  native_fields <- c("compiled_abi_version", "coordinate_rule", "rotation_enabled",
                     "rotation_kernel_label", "grid_mode", "rotation_centered")
  if (inherits(object, "robart") || any(native_fields %in% names(object)))
    stop(paste("Static BART prediction cannot evaluate a gbart/roBART result.",
      "For saved inputs use yhat.test.mean or prob.test.mean, or predict() on a robart() result.",
      "Supply x.test during fitting; arbitrary newdata for rotated states is not yet supported."),
      call. = FALSE)
  trees <- object[["treedraws", exact = TRUE]]
  if (!is.list(trees) || !is.list(trees[["cutpoints", exact = TRUE]]) ||
      !length(trees[["cutpoints", exact = TRUE]]) ||
      !length(trees[["trees", exact = TRUE]]))
    stop("No usable static tree draws and common cutpoints are available.", call. = FALSE)
  if (is.null(dim(newdata)) || length(dim(newdata)) != 2L ||
      nrow(newdata) < 1L || ncol(newdata) < 1L)
    stop("Static prediction requires nonempty, two-dimensional newdata.", call. = FALSE)
  if (anyNA(newdata) || (is.numeric(newdata) && any(!is.finite(newdata))))
    stop("Static prediction does not accept missing or nonfinite newdata.", call. = FALSE)
  invisible(TRUE)
}
