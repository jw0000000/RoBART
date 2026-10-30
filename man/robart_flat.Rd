\name{robart_flat}
\alias{robart_flat}
\title{Recover the flat result layout without losing fields}
\description{Reverse the compact presentation of a robart object. No model fitting or random draws occur.}
\usage{robart_flat(object)}
\arguments{\item{object}{A compact or flat result. Flat lists are returned unchanged.}}
\details{All original field names, values, NULL entries, order and attributes are
restored. Sampling arrays are not recomputed. A compact result with a removed
required field causes an error rather than a silent incomplete reconstruction.
This function does not turn a fit into an ordinary axis-aligned BART model and
does not enable the legacy static predictor on rotated states.}
\value{A flat result list.}

\examples{
# robart_flat reverses grouping without fitting or changing draws.
x <- list(sigma = c(0.5, 0.6))
identical(robart_flat(x), x)
}
