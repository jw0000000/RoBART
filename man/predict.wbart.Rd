\name{predict.wbart}
\alias{predict.wbart}
\alias{predict.pbart}
\alias{predict.lbart}
\title{Guarded prediction from inherited static BART tree draws}
\description{Registered methods for the inherited static tree representation.
They explicitly reject common gbart and rotated robart results because those
require a per-draw frame and grid evaluator. No rotated newdata support is added.}
\usage{
\method{predict}{wbart}(object, newdata, mc.cores = 1,
  openmp = (mc.cores.openmp() > 0), ...)
\method{predict}{pbart}(object, newdata, mc.cores = 1,
  openmp = (mc.cores.openmp() > 0), ...)
\method{predict}{lbart}(object, newdata, mc.cores = 1,
  openmp = (mc.cores.openmp() > 0), ...)
}
\arguments{
\item{object}{An inherited static BART fit with serialized trees and a common cutpoint list.}
\item{newdata}{Nonempty complete numeric design with the processed predictor order. Factor schema learning is not provided.}
\item{mc.cores}{Prediction thread/worker count. The supplied package test targets the serial path; parallel prediction is not newly certified.}
\item{openmp}{Whether to use the native prediction path; the default tests compiled OpenMP availability.}
\item{...}{Passed to inherited prediction helpers.}
}
\value{For wbart, posterior test draws (or their mean when requested through
inherited controls). Binary legacy methods return latent and probability draws
and probability means. The separate legacy binary fitting engines are not newly
certified by registering these methods.}
\details{Use x.test while fitting gbart(), and read yhat.test.mean or
prob.test.mean. predict() on a robart() object already supports saved inputs.
Even a gbart fit with rotation disabled is conservatively rejected by these
static methods until its complete state contract is covered by a new predictor.}
\seealso{\code{\link{wbart}}, \code{\link{robart}}}
\examples{
set.seed(12)
x <- matrix(rnorm(90), 30, 3)
y <- x[, 1]^2 + rnorm(30, sd = 0.5)
f <- wbart(x, y, ntree = 5L, nskip = 2L, ndpost = 3L,
           printevery = 1000L)
dim(predict(f, x[1:3, , drop = FALSE], mc.cores = 1L, openmp = FALSE))
}
