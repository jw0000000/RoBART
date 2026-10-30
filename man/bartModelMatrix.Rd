\name{bartModelMatrix}
\alias{bartModelMatrix}
\title{Converting the data as the input data for the BART model}
\usage{
bartModelMatrix(
  X,
  numcut = 0L,
  usequants = FALSE,
  type = 7,
  rm.const = FALSE,
  cont = FALSE,
  xinfo = NULL
)
}
\arguments{
\item{X}{Numeric vector/matrix, factor or data frame to encode; observations are rows.}
\item{numcut}{Requested candidate count. Zero returns only the numeric design matrix.}
\item{usequants}{Whether to use R quantile candidates when the inherited preprocessing reaches that branch.}
\item{type}{R quantile interpolation type for preprocessing, not a response-family code.}
\item{rm.const}{Whether constant numeric columns are removed.}
\item{cont}{Use equally spaced interior candidates for nonconstant numeric columns.}
\item{xinfo}{Optional list or matrix supplying candidate values; matrix padding uses NA.}
}
\value{With numcut=0, a numeric design matrix. Otherwise a list with X,
numcut, rm.const (an R column subscript, potentially negative), xinfo and grp.
Factor coding is inherited; it does not learn a reusable train/test level schema.}
\description{
Converting the data as the input data for the BART model
}

\examples{
x <- data.frame(a = c(0, 1, 2, 3), b = factor(c("a", "b", "a", "b")))
z <- bartModelMatrix(x, numcut = 3L, rm.const = TRUE)
names(z)
dim(z$X)
}
