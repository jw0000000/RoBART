\name{robart_trace}
\alias{robart_trace}
\title{Extract a sigma or saved prediction series with iteration indices}
\description{Read a trace from a compact or flat convenience result without refitting.}
\usage{robart_trace(object, variable = c("auto", "sigma", "probability", "latent"),
  index = 1L, data = c("test", "train"), include_warmup = TRUE)}
\arguments{
\item{object}{A Gaussian or probit robart result.}
\item{variable}{auto chooses sigma for Gaussian and probability for probit.}
\item{index}{Previously chosen prediction column; ignored for sigma.}
\item{data}{Saved test or training inputs for a prediction trace.}
\item{include_warmup}{For sigma, use the full chain when TRUE or only retained prediction iterations when FALSE. Prediction traces contain retained draws only, regardless of this argument.}}
\details{Probit has no estimated Gaussian residual sigma. Requesting it causes an
error. The helper never manufactures a sigma trace from sigma_last or prior
scale. Probability is a draw-wise probit transformation. Latent denotes the
stored predictor including offset; for Gaussian it is the regression mean.
Missing iteration metadata are not replaced by guessed burn-in/thinning.}
\value{A data frame with columns iteration and value, and quantity/data attributes.
Iteration numbers are full-chain indices. No ESS or convergence conclusion is computed.}

\examples{
set.seed(13)
x <- matrix(rnorm(90), 30, 3)
y <- x[, 1] + rnorm(30, sd = 0.5)
f <- robart(x, y, ntree = 5L, nskip = 2L, ndpost = 3L, seed = 14L)
robart_trace(f, variable = "sigma", include_warmup = FALSE)
}
