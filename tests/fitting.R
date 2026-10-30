# Short package regressions; these are not convergence or predictive benchmarks.
library(robart)
local({
  eq <- function(a,b,tol=1e-12) isTRUE(all.equal(a,b,tolerance=tol,check.attributes=FALSE))
  scientific <- function(f,rng) {
    fields <- c("yhat.train","yhat.test","prob.train","prob.test","sigma",
      "sigma.mean","sigma_last","offset","tau","treedraws","varcount",
      "varprob","saved_states","save_state_iters","rot_proposed","rot_accepted",
      "dart_theta_trace")
    out <- unclass(f)[intersect(fields,names(f))]
    out[["RNG"]] <- rng; out
  }
  set.seed(41L)
  x <- matrix(rnorm(180),60L,3L); xt <- x[1:4,,drop=FALSE]
  y <- 7+sin(x[,1]+.3*x[,2])+.5*rnorm(60L)
  a <- list(x.train=x,y.train=y,x.test=xt,type="wbart",offset=7,
    ntree=20L,numcut=10L,nskip=2L,ndpost=4L,keepevery=3L,
    initial_rotation=FALSE,warm_up=0L,K=5L,epsilon=.08,
    rotation_xinfo_policy="uniform",sparse=FALSE,augment=FALSE,
    save_state_iters=c(5L,14L),verbose=FALSE,printevery=1000L)
  set.seed(100L); off <- do.call(gbart,c(a,list(compute_lpml=FALSE))); r0 <- .Random.seed
  set.seed(100L); on <- do.call(gbart,c(a,list(compute_lpml=TRUE))); r1 <- .Random.seed
  stopifnot(identical(scientific(off,r0),scientific(on,r1)))
  si <- on[["sigma",exact=TRUE]][c(5L,8L,11L,14L)]
  ell <- vapply(seq_along(y),function(i)
    -.5*log(2*pi)-log(si)-.5*((y[i]-on[["yhat.train",exact=TRUE]][,i])/si)^2,
    numeric(4L))
  logcpo <- vapply(seq_along(y),function(i){
    z <- -ell[,i]; mx <- max(z);log(4)-mx-log(sum(exp(z-mx)))
  },numeric(1L))
  stopifnot(eq(on[["LPML",exact=TRUE]],sum(logcpo)),eq(on[["sigma.mean",exact=TRUE]],mean(si)))
  # Compact formatting, serialization, accessors and probability means.
  g <- robart(x,y,x.test=xt,ntree=20L,nskip=2L,ndpost=4L,seed=42L)
  f <- robart_flat(g)
  pack <- getFromNamespace(".robart3b_pack","robart")
  stopifnot(identical(f,robart_flat(pack(f))))
  tmp <- tempfile(fileext=".rds");on.exit(unlink(tmp),add=TRUE)
  saveRDS(g,tmp); reread <- readRDS(tmp)
  stopifnot(identical(predict(g),predict(reread)),identical(robart_flat(g),robart_flat(reread)))
  tr <- robart_trace(g,"sigma",include_warmup=FALSE)
  stopifnot(identical(tr$iteration,3:6),eq(tr$value,f[["sigma",exact=TRUE]][3:6]))
  z <- as.double(x[,1]+x[,2]>0)
  b <- robart(x,z,x.test=xt,type="pbart",offset=-.5,
               ntree=20L,nskip=2L,ndpost=4L,seed=42L)
  bf <- robart_flat(b)
  stopifnot(eq(predict(b),colMeans(pnorm(bf[["yhat.test",exact=TRUE]]))))
  stopifnot(is.null(b[["sigma",exact=TRUE]]),
    inherits(tryCatch(robart_trace(b,"sigma"),error=function(e)e),"error"))
  set.seed(43L)
  lf <- gbart(x,z,x.test=xt,type="lbart",offset=-.5,ntree=20L,
    nskip=2L,ndpost=4L,keepevery=1L,initial_rotation=FALSE,warm_up=0L,
    verbose=FALSE,compute_lpml=FALSE)
  stopifnot(eq(lf[["prob.test.mean",exact=TRUE]],colMeans(plogis(lf[["yhat.test",exact=TRUE]]))))
  # The legacy static serial path remains available and package-specific.
  set.seed(51L)
  stat <- wbart(x,y,x.test=xt,ntree=5L,nskip=2L,ndpost=3L,printevery=1000L)
  pr <- predict(stat,xt,mc.cores=1L,openmp=FALSE)
  artifact_dir <- Sys.getenv("ROBART_TEST_ARTIFACT_DIR", unset = "")
  if (nzchar(artifact_dir)) {
    saveRDS(list(fit = stat, replay = pr, expected = stat[["yhat.test", exact = TRUE]],
                 x = x, y = y, x.test = xt),
            file.path(artifact_dir, "legacy_static_before_assertion.rds"))
  }
  cat("Legacy static prediction max absolute error:",
      format(max(abs(pr-stat[["yhat.test", exact = TRUE]])), digits = 17), "\n")
  stopifnot(eq(pr,stat[["yhat.test",exact=TRUE]],2e-10))
  cat("Short Gaussian/probit/logit, layout and static prediction regressions passed.\n")
})
