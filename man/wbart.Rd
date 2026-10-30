\name{wbart}
\alias{wbart}
\title{wBART model (for the continuous f(x) case)}
\usage{
wbart(
  x.train,
  y.train,
  x.test = matrix(0, 0, 0),
  sparse = FALSE,
  theta = 0,
  omega = 1,
  a = 0.5,
  b = 1,
  augment = FALSE,
  rho = NULL,
  xinfo = matrix(0, 0, 0),
  usequants = FALSE,
  cont = FALSE,
  rm.const = TRUE,
  sigest = NA,
  sigdf = 3,
  sigquant = 0.9,
  k = 2,
  power = 2,
  base = 0.95,
  sigmaf = NA,
  lambda = NA,
  fmean = mean(y.train),
  w = rep(1, length(y.train)),
  ntree = 200L,
  numcut = 100L,
  ndpost = 1000L,
  nskip = 100L,
  keepevery = 1L,
  nkeeptrain = ndpost,
  nkeeptest = ndpost,
  nkeeptestmean = ndpost,
  nkeeptreedraws = ndpost,
  printevery = 100L,
  transposed = FALSE
)
}
\arguments{
\item{x.train}{Training predictors in rows; transposed=TRUE bypasses preprocessing.}
\item{y.train}{Numeric Gaussian response vector matching the training observations.}
\item{x.test}{Test predictors supplied during fitting, or an empty matrix.}
\item{sparse}{Enable the inherited DART update in the legacy cwbart path.}
\item{theta}{Legacy DART concentration argument; cwbart retains its inherited setter behavior. Use gbart for the audited fixed-theta contract.}
\item{omega}{Legacy compatibility argument; no new statistical effect is assigned.}
\item{a}{First shape parameter for the inherited DART concentration prior.}
\item{b}{Second shape parameter for the inherited DART concentration prior.}
\item{augment}{Normalize birth-coordinate probabilities over admissible coordinates. The shared structural engine uses the full recursive prior, not the earlier virtual-count augmentation.}
\item{rho}{DART scaling; NULL resolves to the processed predictor count.}
\item{xinfo}{List or matrix of candidate cutpoints, passed through bartModelMatrix.}
\item{usequants}{Use the inherited R quantile candidate rule when applicable.}
\item{cont}{Use evenly spaced interior cutpoints for nonconstant predictors.}
\item{rm.const}{Remove constant columns during model-matrix preprocessing.}
\item{sigest}{Initial/calibration Gaussian noise SD when lambda is not supplied.}
\item{sigdf}{Degrees of freedom of the Gaussian noise-variance prior.}
\item{sigquant}{Probability used to calibrate the noise-prior scale.}
\item{k}{Leaf-mean shrinkage parameter, not the rotation plane count.}
\item{power}{Exponent of the inherited depth-dependent split probability.}
\item{base}{Root split probability.}
\item{sigmaf}{Optional numerator for tau=sigmaf/sqrt(ntree); NA uses the response range.}
\item{lambda}{Noise-prior scale; a supplied value initializes sigma at sqrt(lambda). This legacy entry does not implement gbart's known-sigma convention.}
\item{fmean}{Constant subtracted before fitting and added back to returned predictions.}
\item{w}{Observation-specific standard-deviation multipliers.}
\item{ntree}{Number of additive trees.}
\item{numcut}{Requested candidate-cutpoint count for each predictor.}
\item{ndpost}{Post-burn-in budget before multiplication by keepevery.}
\item{nskip}{Number of burn-in sweeps.}
\item{keepevery}{Multiplier for the native post-burn-in sweep budget.}
\item{nkeeptrain}{Number of training draws to retain; incompatible divisors are reset to ndpost by the inherited wrapper.}
\item{nkeeptest}{Number of test draws to retain.}
\item{nkeeptestmean}{Number of test sweeps used in the separately accumulated test mean.}
\item{nkeeptreedraws}{Number of serialized tree draws to retain for static prediction.}
\item{printevery}{Positive progress interval. The legacy engine prints progress independently of robart verbose controls.}
\item{transposed}{If TRUE, inputs have predictors in rows and observations in columns.}
}
\value{
posterior samples of the wBART model
}
\description{
wBART model (for the continuous f(x) case)
}

\details{
The shared native birth/death and DART implementations now use the full recursive
discrete target without a minimum occupancy gate. This changes prior versions
of this fork; it is not an unchanged upstream BART sampler.
This is the inherited axis-aligned wbart/cwbart entry, not the
rotation-enabled gbart entry. Rotation proposals are explicitly disabled, all
frames are identity matrices, and every live tree uses the exact common grid
returned in treedraws. Static leaf values are serialized at double round-trip
precision. Its original data handling, DART behavior and
storage budgets are retained. In particular, fixed DART theta and known-sigma
contracts of gbart must not be assumed for this separate legacy engine.
Use finite, compatible numeric matrices for static new-data prediction;
store tree draws and retain the processed predictor order.}
\seealso{\code{\link{gbart}}, \code{\link{robart}}, \code{\link{predict.wbart}}}
\examples{
set.seed(6)
x <- matrix(rnorm(120), 40, 3)
y <- sin(x[, 1]) + rnorm(40, sd = 0.5)
f <- wbart(x, y, x.test = x[1:4, , drop = FALSE], ntree = 5L,
           nskip = 2L, ndpost = 3L, printevery = 1000L)
dim(f$yhat.test)
}
