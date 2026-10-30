library(robart)
source("support/replay.R",local=TRUE)
local({
  set.seed(61L)
  x <- matrix(rnorm(180),60L,3L)
  x <- x[rep(seq_len(30L),each=2L),,drop=FALSE]
  xt <- x[c(1L,7L,18L,43L),,drop=FALSE]
  y <- as.double(x[,1]+.4*x[,2] > 0)
  set.seed(62L)
  f <- gbart(x,y,x.test=xt,type="pbart",offset=0,ntree=20L,
    numcut=10L,nskip=2L,ndpost=6L,keepevery=1L,
    initial_rotation=FALSE,warm_up=0L,rotation_xinfo_policy="quantile",
    save_state_iters=c(3L,8L),verbose=FALSE,compute_lpml=FALSE)
  project <- getFromNamespace("cgbart_canonical_coordinates","robart")
  states <- f[["saved_states",exact=TRUE]]
  stopifnot(length(states)==2L)
  for(k in seq_along(states)) {
    st <- states[[k]]; ii <- c(3L,8L)[k]-2L
    stopifnot(identical(st[["coordinate_rule",exact=TRUE]],"ordered_dot_v1"))
    Q <- st[["rot_matrix",exact=TRUE]]
    for(j in seq_len(dim(Q)[3L]))
      stopifnot(identical(unname(project(x,Q[,,j])),unname(.c3_r_project(x,Q[,,j]))))
    got <- .c3_replay(st,x,f[["offset",exact=TRUE]],project=project)
    test <- .c3_replay(st,xt,f[["offset",exact=TRUE]],project=project)
    stopifnot(max(abs(got$prediction-f[["yhat.train",exact=TRUE]][ii,]))<2e-10,
      max(abs(test$prediction-f[["yhat.test",exact=TRUE]][ii,]))<2e-10)
  }
  cat("All-row quantile reconstruction passed, including exact training copies.\n")
})
