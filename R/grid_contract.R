# Internal grid contracts shared by gbart validation and tests.
# Candidate counts still come from bartModelMatrix; this helper never reads Y.
.robart_prepare_xinfo <- function(xinfo) {
  if (is.null(xinfo) || length(xinfo) == 0L) return(matrix(0, 0L, 0L))
  if (is.list(xinfo) && !is.data.frame(xinfo)) {
    good <- vapply(xinfo, function(z) is.numeric(z) && is.null(dim(z)) &&
      length(z) > 0L && all(is.finite(z)), logical(1L))
    if (!length(good) || !all(good))
      stop("Each custom xinfo list entry must be a nonempty finite numeric vector.", call. = FALSE)
    nc <- lengths(xinfo)
    out <- matrix(NA_real_, length(xinfo), max(nc))
    for (j in seq_along(xinfo)) out[j, seq_len(nc[j])] <- xinfo[[j]]
    xinfo <- out
  }
  if (!is.matrix(xinfo) || !is.numeric(xinfo) || nrow(xinfo) == 0L || ncol(xinfo) == 0L)
    stop("Custom xinfo must be a numeric matrix or a list of cutpoint vectors.", call. = FALSE)
  for (j in seq_len(nrow(xinfo))) {
    z <- xinfo[j, ]; used <- which(!is.na(z))
    if (!length(used) || !identical(unname(used), seq_len(length(used))) ||
        any(!is.finite(z[used])) || is.unsorted(z[used], strictly = FALSE))
      stop("Each xinfo row needs sorted finite candidates followed only by optional NA padding.", call. = FALSE)
  }
  xinfo
}

.robart_resolve_grid <- function(rotation_enabled, dynamic, policy, usequants,
                                 custom_xinfo) {
  if (!is.logical(rotation_enabled) || length(rotation_enabled) != 1L || is.na(rotation_enabled))
    stop("rotation_enabled must be TRUE or FALSE.", call. = FALSE)
  if (!is.logical(usequants) || length(usequants) != 1L || is.na(usequants))
    stop("usequants must be TRUE or FALSE.", call. = FALSE)
  if (!is.null(dynamic) && (!is.logical(dynamic) || length(dynamic) != 1L || is.na(dynamic)))
    stop("dynamic_xinfo_on_rotation must be NULL, TRUE or FALSE.", call. = FALSE)
  requested_policy <- policy
  if (policy == "auto") policy <- if (usequants) "quantile" else "uniform"
  if (custom_xinfo && policy == "global_uniform")
    stop("global_uniform and custom xinfo cannot be used together.", call. = FALSE)
  automatic <- is.null(dynamic)
  if (automatic) dynamic <- rotation_enabled && !custom_xinfo && policy != "global_uniform"
  if (dynamic && !rotation_enabled)
    stop("Rotation is disabled: use dynamic_xinfo_on_rotation=NULL or FALSE for a fixed grid.", call. = FALSE)
  if (dynamic && custom_xinfo)
    stop("Custom xinfo is fixed: use dynamic_xinfo_on_rotation=NULL or FALSE.", call. = FALSE)
  if (dynamic && policy == "global_uniform")
    stop("global_uniform is fixed: use dynamic_xinfo_on_rotation=NULL or FALSE.", call. = FALSE)
  list(dynamic = dynamic, policy = policy, requested_policy = requested_policy,
       automatic = automatic, custom = custom_xinfo,
       mode = if (custom_xinfo) "fixed_custom" else paste0(if (dynamic) "dynamic_" else "fixed_", policy),
       selection = if (!automatic) "explicit" else if (custom_xinfo) "custom_xinfo" else
         if (policy == "global_uniform") "global_uniform" else if (!rotation_enabled) "rotation_disabled" else "rotation_enabled")
}
