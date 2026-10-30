\name{robart-package}
\alias{robart-package}
\alias{robart-package-overview}
\docType{package}
\title{Tree-specific rotations for Bayesian additive regression trees}
\description{
Bayesian additive regression trees with a separate orthogonal rotation for
each tree. The convenience interface supports Gaussian regression and
Bernoulli-probit classification, posterior draws and sampling diagnostics.
}
\details{
\code{robart()} accepts finite numeric matrices and provides compact or flat
output. Prediction uses the training inputs and test inputs supplied during
fitting; arbitrary new-data prediction after fitting is not implemented.
Use \code{x.test} when fitting and \code{predict()} to retrieve saved
predictions. Gaussian prediction draws describe the regression function;
they do not add observation-noise draws.

\code{gbart()} exposes advanced grid, DART and depth-prior settings. Its common
Bernoulli-logit path uses a finite 1001-term normal-scale-mixture approximation.
The convenience \code{robart()} interface supports only Gaussian and probit
models. Inherited factor and parallel wrappers, general rotated new-data
prediction and binary extreme-tail LPML retain separate limitations described
in the function help. Package tests do not establish posterior convergence.

The \code{seed} argument of \code{robart()} calls \code{set.seed()} before
preprocessing without changing \code{RNGkind()}. The inherited \code{seed}
argument of \code{gbart()} does not seed that fitting path; call
\code{set.seed()} before fitting with \code{gbart()}.

Substantial code is derived from the BART R package. Original copyright
notices and GPL version 2 or any later version terms are retained. See the
installed COPYRIGHTS and CITATION files for provenance and attribution.

Native registration, NAMESPACE and Rd help are maintained explicitly in the
source. Standard installation requires an R-compatible C++17 compiler and
the declared dependencies; it does not require earlier package installations,
private data or a development binding-generation step.
}
\author{Jeongung Heo; inherited BART contributors are listed in the package
metadata and installed COPYRIGHTS file.}
\references{
Sparapani, R., Spanbauer, C. and McCulloch, R. (2021).
Nonparametric Machine Learning and Efficient Computation with Bayesian Additive
Regression Trees: The BART R Package. Journal of Statistical Software, 97(1),
1--66. \doi{10.18637/jss.v097.i01}.
}
\seealso{\code{\link{robart}}, \code{\link{gbart}}, \code{\link{wbart}},
\code{\link{robart_control}}, \code{\link{robart_trace}},
\code{\link{robart_flat}}}
\keyword{package}
