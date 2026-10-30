# Base-R package test: no network, prior stage or extra testing dependency.
library(robart)
local({
  expected <- c(`_robart_formal_tree_audit`=6L, `_robart_formal_cut_audit`=8L, `_robart_cgbart_canonical_coordinates`=2L,
    `_robart_cgbart`=60L, `_robart_cgbart_profile_snapshot`=11L,
    `_robart_cwbart`=31L, cabart=31L, cdraw_lambda_i=4L, clbart=24L,
    cpbart=27L, cpwbart=3L, crtgamma=4L, crtnorm=4L, mc_cores_openmp=0L)
  reg <- getDLLRegisteredRoutines(getLoadedDLLs()[["robart"]])[[".Call", exact=TRUE]]
  stopifnot(setequal(names(reg), names(expected)))
  actual <- vapply(reg[names(expected)], function(z) z[["numParameters", exact=TRUE]], integer(1L))
  stopifnot(identical(unname(actual), unname(expected)))
  set.seed(721L)
  seed <- .Random.seed
  raw <- rep(list(NULL), 60L)
  raw[1:5] <- list(1L, -1L, 3L, 0L, numeric())
  call_raw <- function(a) do.call(.Call, c(list("_robart_cgbart"), a,
                                          list(PACKAGE="robart")))
  err <- tryCatch(call_raw(raw), error=function(e)e)
  stopifnot(inherits(err,"error"), grepl("n",conditionMessage(err),fixed=TRUE),
            identical(seed,.Random.seed))
  raw[2L] <- list(60L); raw[8L] <- list(2L)
  raw[10L] <- list(3L); raw[12L] <- list(0L)
  err <- tryCatch(call_raw(raw), error=function(e)e)
  stopifnot(inherits(err,"error"), grepl("thin",conditionMessage(err),fixed=TRUE),
            identical(seed,.Random.seed))
  for (cl in c("robart","wbart","pbart","lbart"))
    stopifnot(is.function(getS3method("predict",cl)))
  flat <- structure(list(compiled_abi_version=2026100601L),class="wbart")
  err <- tryCatch(predict(flat,matrix(0,2L,3L)),error=function(e)e)
  stopifnot(inherits(err,"error"),
            grepl("Static BART prediction",conditionMessage(err),fixed=TRUE),
            identical(seed,.Random.seed))
  flag <- getFromNamespace(".robart_flag","robart")
  stopifnot("allow_na" %in% names(formals(flag)))
  flag(NA,"flag",allow_na=TRUE)
  stopifnot(inherits(tryCatch(flag(NA,"flag"),error=function(e)e),"error"))
  cat("Registered API, raw-input guards, S3 registration and partial-name contracts passed.\n")
})
