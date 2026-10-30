# The expected files are independently generated, not computed by the native API.
# Pure-R fixture checks can run before package compilation.
.formal5_cases <- function(root) {
  ids <- c("uniform", "weighted", "ties", "zero_range")
  lapply(stats::setNames(ids, ids), function(id) {
    d <- file.path(root, id)
    dat <- utils::read.csv(file.path(d, "design.csv"), check.names=FALSE)
    list(id=id, x=as.matrix(dat[, c("x1","x2"), drop=FALSE]),
      residual=dat[["residual",exact=TRUE]], sd=dat[["sd",exact=TRUE]],
      grid=if(id=="ties")list(c(1/3,1/3),.5)else list(c(1/3,2/3),.5),
      control=list(tau=.6,log_prob=log(if(id=="weighted")c(.2,.8)else c(.5,.5))),
      states=utils::read.csv(file.path(d,"states.csv"),check.names=FALSE),
      q0=as.matrix(utils::read.csv(file.path(d,"q0.csv"),header=FALSE)),
      q1=as.matrix(utils::read.csv(file.path(d,"q1.csv"),header=FALSE)))
  })
}
.formal5_fixture_check <- function(root) {
  result <- list()
  for(z in .formal5_cases(root)) {
    s <- z[["states",exact=TRUE]]
    stopifnot(!anyDuplicated(s[["tree",exact=TRUE]]),
      abs(sum(exp(s[["log_prior",exact=TRUE]]))-1)<1e-13)
    for(k in c("q0","q1")) {
      q <- z[[k,exact=TRUE]]
      stopifnot(identical(dim(q),rep(nrow(s),2L)),all(is.finite(q)),min(q)> -1e-15,
        max(abs(rowSums(q)-1))<1e-13)
      post <- exp(s[["log_prior",exact=TRUE]]+s[["log_marginal",exact=TRUE]])
      post <- post/sum(post)
      trans <- q
      for(i in seq_len(nrow(q)))for(j in seq_len(ncol(q)))if(i!=j) {
        trans[i,j] <- if(q[i,j]>0 && q[j,i]>0)
          q[i,j]*min(1,post[j]*q[j,i]/(post[i]*q[i,j]))else 0
      }
      diag(trans) <- 0; diag(trans) <- 1-rowSums(trans)
      flow <- post*trans
      stopifnot(max(abs(flow-t(flow)))<1e-13,
        max(abs(as.vector(post%*%trans)-post))<1e-13)
    }
    result[[length(result)+1L]] <- data.frame(check=paste0("independent_fixture:",z$id),passed=TRUE,states=nrow(s))
  }
  do.call(rbind,result)
}
