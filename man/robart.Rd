\name{robart}
\alias{robart}
\alias{print.robart}
\alias{summary.robart}
\alias{print.summary.robart}
\alias{predict.robart}
\title{Gaussian and probit regression with centered tree-specific rotations}
\description{Convenience interface to the common \code{gbart} path,
using identity initialization, dynamic uniform endpoint grids when rotation is
on, and single-scale joint cutpoint retuning. No hard depth cap is added.}
\usage{
robart(x.train, y.train, x.test = NULL,
  ntree = 200L, numcut = 100L,
  ndpost = 1000L, nskip = 9000L, keepevery = 1L,
  K = 5L, epsilon = 0.08, theta_max = 0.35 * epsilon / 0.15,
  support_expand_prob = 0.5, k = 2, power = 2, base = 0.95,
  sigest = NA_real_, sigdf = 3, sigquant = 0.90,
  lambda = NA_real_, tau.num = NA_real_,
  sigma_fixed = NULL, rotation = TRUE, seed = NULL,
  control = robart_control(), type = "wbart", offset = NULL,
  output = c("compact", "flat"))
\method{print}{robart}(x, ...)
\method{summary}{robart}(object, ...)
\method{print}{summary.robart}(x, ...)
\method{predict}{robart}(object, newdata = NULL,
  type = c("mean", "draws"), ..., scale = c("response", "link"),
  data = c("test", "train"))
}
\arguments{
\item{x.train}{Finite numeric matrix, observations in rows. Constant columns use the inherited removal policy.}
\item{y.train}{Finite numeric response. Gaussian requires nonconstant data with at least three distinct values, as before. Probit requires 0/1. A one-class probit response requires an explicit finite offset.}
\item{x.test}{Finite numeric matrix with matching input columns, or NULL.}
\item{ntree}{Number of trees. The convenience default remains 200 for both supported models. This does not change the separate defaults of gbart().}
\item{numcut}{Requested number of candidate cutpoints per predictor.}
\item{ndpost}{Number of retained prediction draws.}
\item{nskip}{Burn-in iterations before retained prediction draws. Rotation starts at iteration one.}
\item{keepevery}{Thinning interval; retained full-chain iterations are nskip+s*keepevery.}
\item{K}{Number of ordered Givens planes, distinct from lowercase k.}
\item{epsilon}{Scale of the centered normal angle proposal before truncation.}
\item{theta_max}{Positive angle bound in radians.}
\item{support_expand_prob}{Support-expanding pair selection probability.}
\item{k}{Leaf shrinkage parameter.}
\item{power,base}{Polynomial tree-prior parameters.}
\item{sigest,sigdf,sigquant,lambda}{Inherited Gaussian noise controls. Lambda is NA or positive in this convenience entry; lambda=0 remains in gbart(). For probit these must retain their inactive defaults.}
\item{tau.num}{Gaussian: NA retains the range-based scale. Probit: when omitted, use 3 and tau=3/(k*sqrt(ntree)). An explicitly supplied probit value must be positive.}
\item{sigma_fixed}{Gaussian only: NULL estimates sigma; a positive SD fixes it from the first update. Probit has fixed latent SD 1, not a residual-sigma posterior.}
\item{rotation}{Enable rotations. FALSE gives identity initialization with fixed uniform grids.}
\item{seed}{NULL uses the caller RNG; a nonnegative integer sets the seed before preprocessing. RNGkind is unchanged.}
\item{control}{A robart_control() object.}
\item{type}{In robart(): wbart for Gaussian, pbart for Bernoulli-probit. In predict(): mean or draws. Logit is not exposed in robart() in this release; its existing gbart path is not removed.}
\item{offset}{NULL uses the inherited response mean or inverse-link of the sample proportion. A supplied finite value is in response units for Gaussian and link units for probit.}
\item{output}{compact groups additional fields without dropping values; flat retains the flat layout. This is presentation, not a sampling option.}
\item{x,object}{A fitted robart object, or the returned small summary for its print method.}
\item{newdata}{Must be NULL. Only inputs supplied during fitting can be used.}
\item{scale}{Response predictions by default: Gaussian means or probit probabilities. Link returns the stored predictor, including offset.}
\item{data}{Use saved test predictions by default, or training predictions.}
\item{...}{Additional arguments are not supported.}
}
\details{
The fitting route remains robart -> gbart -> cgbart. Both joint rotation and
birth/death now use the full recursive discrete prior. No minimum child-size
gate remains. Cutpoint centers are adjacent-order-statistic midpoints, and
invalid sequential proposals fail without retry. These are target/proposal
changes, not a speed-only or presentation-only patch. Gaussian and probit use the shared native engine. The common logit entry in
gbart() uses an offset-aware scale-then-latent update with a finite mixture. DART remains in
gbart(); the convenience profile uses sparse=FALSE, augment=FALSE. Binary
latent-variable augmentation is distinct from DART augmentation.

The Gaussian defaults, seed application point, dynamic/fixed behavior and
storage schedules are preserved. The newly appended model and output arguments
do not change positions of the pre-existing arguments. Default output is now
compact, an explicit presentation change; legacy flat-field scripts can call
robart_flat(fit) or request output="flat". Direct gbart() output is unchanged.

Core yhat/prob/sigma/varcount/varprob/treedraws fields remain at the top level.
Settings, additional diagnostics, snapshots and implementation metadata are
grouped. Every original field is retained, including NULL, NA and non-applicable
Gaussian fields in binary results. robart_flat() reverses grouping without
refitting or duplicating a hidden full fit. This is not memory compression.

Probit probability means average pnorm over individual predictor draws, not
pnorm of the mean predictor. Summary reports no Gaussian sigma for probit.
The predict method dispatches for robart before inherited static BART methods
and rejects arbitrary newdata. Tree strings alone do not represent all rotated
states; the tree, Q, per-tree grid, offset and iteration must remain together.

The convenience grid remains uniform. Advanced quantile grids in gbart() use the versioned ordered_dot_v1 evaluator
for coordinates, including training knots. Use that evaluator, not an arbitrary
BLAS product, to replay such snapshots. Legacy snapshots are not repaired.
The shared full-prior DART extension is documented in gbart(); arbitrary newdata prediction remains unavailable. No claim of
full-forest mixing or classification superiority follows from an API test.
}
\value{A list of class c("robart","wbart") or c("robart","pbart"). Compact results
add settings, diagnostics, states and metadata groups. Native snapshots use schema 9 and a recorded coordinate rule; the R compact
layout has its own version 1. summary() returns a small
summary.robart object; predict() returns saved posterior means or draws.}
\seealso{\code{\link{gbart}}, \code{\link{robart_flat}}, \code{\link{robart_trace}}}

\examples{
set.seed(10)
x <- matrix(rnorm(120), 40, 3)
y <- sin(x[, 1]) + rnorm(40, sd = 0.5)
f <- robart(x, y, x.test = x[1:4, , drop = FALSE], ntree = 5L,
            nskip = 2L, ndpost = 3L, seed = 11L)
summary(f)
predict(f)
robart_trace(f, variable = "sigma")
}
