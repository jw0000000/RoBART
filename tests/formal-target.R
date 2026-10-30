# Deterministic tests of the production target and proposal code via registered
# native audit entries. These are not a second sampler and consume no RNG.
library(robart)
source("support/formal-reference.R",local=TRUE)
local({
  close <- function(a,b,tol=3e-11) isTRUE(all.equal(a,b,tolerance=tol,check.attributes=FALSE))
  audit <- getFromNamespace("formal_tree_audit","robart")
  cut <- getFromNamespace("formal_cut_audit","robart")
  .formal5_fixture_check("support/finite")
  set.seed(9471L); rng <- .Random.seed
  for(z in .formal5_cases("support/finite")) {
    st <- z$states; n <- nrow(st)
    for(aug in c(FALSE,TRUE)) {
      q <- matrix(0,n,n); target <- exp(st$log_prior+st$log_marginal);target<-target/sum(target)
      expected <- z[[if(aug)"q1"else"q0",exact=TRUE]]
      P <- matrix(0,n,n)
      for(i in seq_len(n)) {
        a <- audit(z$x,z$residual,z$sd,st$tree[i],z$grid,
                   c(z$control,list(augment=aug)))
        stopifnot(isTRUE(a$valid),close(a$log_prior,st$log_prior[i]),
          close(a$log_marginal,st$log_marginal[i]))
        for(move in a$moves) {
          j <- match(move$tree,st$tree)
          stopifnot(!is.na(j),j!=i,isTRUE(move$valid),expected[i,j]>0,expected[j,i]>0,
            close(move$logq_forward,log(expected[i,j])),
            close(move$logq_reverse,log(expected[j,i])),
            close(move$log_prior_ratio,st$log_prior[j]-st$log_prior[i]),
            close(move$log_likelihood_ratio,st$log_marginal[j]-st$log_marginal[i]))
          q[i,j] <- q[i,j]+exp(move$logq_forward)
          P[i,j] <- P[i,j]+exp(move$logq_forward+min(0,move$logacc))
        }
      }
      diag(q)<-1-rowSums(q);diag(P)<-1-rowSums(P)
      stopifnot(close(q,expected),min(P)> -1e-13,close(rowSums(P),rep(1,n)),
        max(abs(target*P-t(target*P)))<1e-12,
        max(abs(as.vector(target%*%P)-target))<1e-12)
    }
  }
  # Midpoint example: sorted values 0,1,4,5; L=2; nearest-grid center=2.
  tree <- function(c)paste0("3\n1 0 ",c," 0\n2 0 0 0\n3 0 0 0\n")
  x <- matrix(c(0,1,4,5),ncol=1); grid <- list(c(1,2,3,4))
  a <- cut(x,x,grid,grid,tree(1),tree(2),TRUE)
  stopifnot(close(a$logq_forward,-1.353732798546091),isTRUE(a$source_restored))
  a <- cut(x,x,list(c(10,11,12,13)),grid,tree(1),tree(1),TRUE)
  stopifnot(identical(a$logq_forward,0),isTRUE(a$source_restored))
  a <- cut(x,x,list(c(10,11,12,13)),grid,tree(1),tree(2),TRUE)
  stopifnot(identical(a$logq_forward,-Inf),isTRUE(a$source_restored))
  a <- cut(x,x,list(c(-4,-3,-2,-1)),grid,tree(1),tree(1),TRUE)
  stopifnot(identical(a$logq_forward,0))
  # Nonzero prior ratio at fixed topology and labels.
  t0 <- "5\n1 0 2 0\n2 0 1 0\n4 0 0 0\n5 0 0 0\n3 0 0 0\n"
  t1 <- sub("1 0 2 0","1 0 3 0",t0,fixed=TRUE)
  xx <- rbind(c(0,0),c(6,6));g <- list(1:5,1:5)
  p0<-audit(xx,c(0,0),c(1,1),t0,g)$log_prior
  p1<-audit(xx,c(0,0),c(1,1),t1,g)$log_prior
  stopifnot(close(p1-p0,log(644/1023)))
  # Empty leaves are in support; observation count alone does not truncate it.
  p0<-audit(matrix(c(3,4),ncol=1),c(0,0),c(1,1),tree(0),list(c(1,2)))
  stopifnot(isTRUE(p0$valid),is.finite(p0$log_marginal))
  p0<-audit(matrix(c(0,0),ncol=1),c(0,0),c(1,1),tree(0),list(c(1,2)))
  stopifnot(!isTRUE(p0$valid),identical(p0$log_prior,-Inf))
  stopifnot(identical(rng,.Random.seed))
  cat("Finite full target, actual BD probabilities, strict support and midpoint/hold rules passed.\n")
})
