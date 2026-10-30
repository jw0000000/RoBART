# Test the legacy static entry before the other fitting regressions.
# These short checks do not certify MCMC convergence.
library(robart)
local({
  tol <- 2e-10
  artifact_dir <- Sys.getenv("ROBART_TEST_ARTIFACT_DIR", unset = "")
  if (!nzchar(artifact_dir)) artifact_dir <- tempfile("robart_static_check_")
  if (!dir.exists(artifact_dir)) dir.create(artifact_dir, recursive = TRUE)
  stopifnot(dir.exists(artifact_dir))
  checks <- list()
  add <- function(id, ok, value = NA_real_) {
    checks[[length(checks) + 1L]] <<- data.frame(check = id, passed = isTRUE(ok),
                                               value = as.numeric(value))
    utils::write.csv(do.call(rbind, checks),
                    file.path(artifact_dir, "static_checks.csv"), row.names = FALSE)
    if (!isTRUE(ok)) stop(paste("Static prediction check failed:", id,
                               "Artifacts:", artifact_dir), call. = FALSE)
  }
  eq <- function(a, b) isTRUE(all.equal(a, b, tolerance = tol, check.attributes = FALSE))
  # Independent traversal of the serialized tree text. No native predictor call.
  reference_prediction <- function(f, X) {
    td <- f[["treedraws", exact = TRUE]]
    cuts <- td[["cutpoints", exact = TRUE]]
    values <- scan(text = td[["trees", exact = TRUE]][1L], what = double(), quiet = TRUE)
    stopifnot(length(values) >= 3L)
    S <- as.integer(values[1L]); M <- as.integer(values[2L]); P <- as.integer(values[3L])
    stopifnot(ncol(X) == P, S > 0L, M > 0L)
    cursor <- 4L; out <- matrix(0, S, nrow(X))
    for (s in seq_len(S)) for (tree_index in seq_len(M)) {
      nn <- as.integer(values[cursor]); cursor <- cursor + 1L
      stopifnot(nn > 0L, cursor + 4L * nn - 1L <= length(values))
      nodes <- matrix(values[seq.int(cursor, length.out = 4L * nn)], ncol = 4L, byrow = TRUE)
      cursor <- cursor + 4L * nn
      for (i in seq_len(nrow(X))) {
        node_id <- 1
        repeat {
          k <- match(node_id, nodes[, 1L])
          stopifnot(!is.na(k))
          left <- 2 * node_id
          if (!(left %in% nodes[, 1L])) {
            out[s, i] <- out[s, i] + nodes[k, 4L]
            break
          }
          variable <- as.integer(nodes[k, 2L]) + 1L
          cut_index <- as.integer(nodes[k, 3L]) + 1L
          stopifnot(variable >= 1L, variable <= P,
                    cut_index >= 1L, cut_index <= length(cuts[[variable]]))
          threshold <- cuts[[variable]][cut_index]
          node_id <- if (X[i, variable] < threshold) left else left + 1
        }
      }
    }
    stopifnot(cursor == length(values) + 1L)
    out + f[["mu", exact = TRUE]]
  }
  set.seed(41L)
  x <- matrix(rnorm(180L), 60L, 3L)
  xt <- x[1:4, , drop = FALSE]
  y <- 7 + sin(x[, 1L] + .3 * x[, 2L]) + .5 * rnorm(60L)
  custom <- list(c(-.7, -.21, -.05, .12, .31, .7),
                 c(-1, -.11, -.03, .13, .5, 1.1),
                 c(-.8, -.2, -.01, .02, .22, .8))
  variants <- list(
    midpoint59 = list(),
    custom_nonuniform = list(xinfo = do.call(rbind, custom)),
    uniform_cont = list(cont = TRUE, numcut = 9L),
    thinning3 = list(keepevery = 3L, fmean = 7)
  )
  for (id in names(variants)) {
    args <- list(x.train = x, y.train = y, x.test = xt, ntree = 5L,
                 nskip = 2L, ndpost = 3L, printevery = 1000L)
    for (nm in names(variants[[id]])) args[[nm]] <- variants[[id]][[nm]]
    set.seed(51L)
    f <- do.call(wbart, args)
    before <- .Random.seed
    pr <- predict(f, xt, mc.cores = 1L, openmp = FALSE)
    manual <- reference_prediction(f, xt)
    pr_train <- predict(f, x, mc.cores = 1L, openmp = FALSE)
    saveRDS(list(fit = f, arguments = args, replay = pr, manual = manual,
                 training_replay = pr_train, rng_after_fit = before),
            file.path(artifact_dir, paste0(id, "_before_assertion.rds")))
    add(paste0(id, ":prediction_does_not_consume_rng"), identical(before, .Random.seed))
    target <- f[["yhat.test", exact = TRUE]]
    error <- max(abs(pr - target))
    add(paste0(id, ":shape"), identical(dim(pr), dim(target)))
    add(paste0(id, ":test_draws_match"), eq(pr, target) && error <= tol, error)
    add(paste0(id, ":independent_replay"), eq(manual, target) && max(abs(manual-target)) <= tol,
        max(abs(manual-target)))
    add(paste0(id, ":train_draws_match"), eq(pr_train, f[["yhat.train", exact=TRUE]]),
        max(abs(pr_train-f[["yhat.train", exact=TRUE]])))
    if (id == "midpoint59") {
      cuts <- f[["treedraws", exact = TRUE]][["cutpoints", exact = TRUE]]
      for (j in seq_len(ncol(x))) {
        v <- sort(unique(x[, j])); expected <- .5 * (head(v, -1L) + tail(v, -1L))
        add(paste0(id, ":cut_column_", j), identical(unname(cuts[[j]]), expected))
      }
    }
    if (id == "custom_nonuniform") {
      cuts <- f[["treedraws", exact = TRUE]][["cutpoints", exact = TRUE]]
      for (j in seq_len(ncol(x)))
        add(paste0(id, ":cut_column_", j), identical(unname(cuts[[j]]), custom[[j]]))
    }
    rds <- file.path(artifact_dir, paste0(id, "_serialization.rds"))
    saveRDS(f, rds)
    reloaded <- readRDS(rds)
    add(paste0(id, ":RDS_replay"), identical(pr, predict(reloaded, xt, mc.cores=1L, openmp=FALSE)))
    add(paste0(id, ":post_RDS_rng"), identical(before, .Random.seed))
  }
  cat("Early static-prediction checks passed. Artifacts:", artifact_dir, "\n")
})
