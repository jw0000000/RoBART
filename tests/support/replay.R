# Test-only Stage3C replay helpers; not a public newdata predictor.
# Test-only helpers. Sourcing this file does not fit or load a model.
.c3_get <- function(x, name) x[[name, exact = TRUE]]
.c3_equal <- function(a,b,tol=2e-10) isTRUE(all.equal(a,b,tolerance=tol,check.attributes=FALSE))
.c3_r_project <- function(X,Q) {
  stopifnot(is.matrix(X), is.matrix(Q), nrow(Q)==ncol(X), ncol(Q)==ncol(X))
  ans <- matrix(0,nrow(X),ncol(X))
  for (i in seq_len(nrow(X))) for (a in seq_len(ncol(X))) {
    value <- 0
    for (k in seq_len(ncol(X))) { product <- Q[a,k]*X[i,k]; value <- value+product }
    ans[i,a] <- value
  }
  ans
}
.c3_replay <- function(state,X,offset,project=NULL) {
  X <- as.matrix(X); p <- ncol(X); Q <- .c3_get(state,"rot_matrix")
  trees <- .c3_get(state,"trees"); grids <- .c3_get(state,"xinfo")
  rule <- .c3_get(state,"coordinate_rule")
  stopifnot(length(dim(Q))==3L,dim(Q)[1L]==p,dim(Q)[2L]==p,
            dim(Q)[3L]==length(trees),length(grids)==length(trees))
  if (identical(rule,"ordered_dot_v1") && is.null(project))
    stop("Canonical snapshots require the versioned native coordinate evaluator.",call.=FALSE)
  pred <- rep(offset,nrow(X)); routes <- matrix(NA_real_,nrow(X),length(trees))
  near <- rep(FALSE,nrow(X))
  for(j in seq_along(trees)) {
    v <- scan(text=trees[j],what=double(),quiet=TRUE)
    count <- v[1L]; stopifnot(count>=1,count==floor(count),length(v)==1+4*count)
    tab <- matrix(v[-1L],ncol=4L,byrow=TRUE); ids <- tab[,1L]
    stopifnot(!anyDuplicated(ids),ids[1L]==1,all(ids<2^52))
    q <- matrix(Q[,,j],p,p)
    Z <- if(identical(rule,"ordered_dot_v1")) project(X,q) else X%*%t(q)
    values <- numeric(nrow(X)); stack <- list(list(id=1,rows=seq_len(nrow(X))))
    while(length(stack)) {
      node <- stack[[length(stack)]]; stack <- stack[-length(stack)]
      k <- match(node$id,ids); stopifnot(!is.na(k))
      left <- match(2*node$id,ids,nomatch=0L); right <- match(2*node$id+1,ids,nomatch=0L)
      stopifnot((left==0L)==(right==0L))
      if(left==0L) {
        values[node$rows] <- tab[k,4L]; routes[node$rows,j] <- node$id
      } else {
        a <- as.integer(tab[k,2L])+1L; c <- as.integer(tab[k,3L])+1L
        stopifnot(a>=1,a<=p,c>=1,c<=length(grids[[j]][[a]]))
        val <- Z[node$rows,a]; cut <- grids[[j]][[a]][c]
        # The band only describes the test fixture. NO row is excluded.
        near[node$rows[abs(val-cut)<=1e-10*pmax(1,abs(val),abs(cut))]] <- TRUE
        lt <- val<cut
        stack[[length(stack)+1L]] <- list(id=2*node$id,rows=node$rows[lt])
        stack[[length(stack)+1L]] <- list(id=2*node$id+1,rows=node$rows[!lt])
      }
    }
    pred <- pred+values
  }
  stopifnot(!anyNA(routes),all(is.finite(pred)))
  list(prediction=pred,routes=routes,near=near)
}
