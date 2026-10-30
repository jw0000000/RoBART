\name{robart_control}
\alias{robart_control}
\title{Storage and diagnostic controls for roBART}
\description{Separates storage and diagnostic choices from the statistical model and rotation proposal.}
\usage{
robart_control(rotation_summary = TRUE, rotation_trace = TRUE,
  trace_every = 10L, diagnostic_every = 0L, timing_every = 0L,
  likelihood_trace = FALSE, partition_trace = FALSE, angles = FALSE,
  global_likelihood_trace = FALSE, iteration_trace = TRUE,
  snapshots = NULL, verbose = FALSE, print_every = NULL)
}
\arguments{
\item{rotation_summary}{Save rotation proposal/acceptance counts.}
\item{rotation_trace}{Save sampled per-tree rotation acceptance and leaf-count traces.}
\item{trace_every}{Stored rotation trace interval. Actual iteration numbers are retained.}
\item{diagnostic_every}{Interval for expensive rotation partition diagnostics. Zero disables, unless required by partition_trace.}
\item{timing_every}{Rotation timing interval; zero disables. Timing storage also requires a rotation trace family to be enabled and an interval divisible by trace_every.}
\item{likelihood_trace}{Store local hard-target and acceptance diagnostic traces.}
\item{partition_trace}{Store partition-change diagnostics.}
\item{angles}{Store canonical angles of saved frames, not an ordered history of proposed rotation angles; potentially large.}
\item{global_likelihood_trace}{Store global rotation likelihood diagnostics.}
\item{iteration_trace}{Store full iteration-level likelihood, prior and timing series. The elapsed-time field is cumulative from MCMC start, not a per-iteration duration.}
\item{snapshots}{NULL stores burn-in endpoint and final state. integer(0) disables snapshots. Positive integers select iterations.}
\item{verbose}{Ordinary native argument and progress output. FALSE suppresses these messages; errors and warnings remain visible.}
\item{print_every}{Native progress interval. NULL uses approximately twenty intervals.}
}
\value{A validated control list. Unknown control names are rejected.}

\details{Trace row numbers are not MCMC iteration numbers; use the returned iteration vector. Snapshot replay requires the tree, rotation matrix, per-tree grid and response offset from the same draw.}

\examples{
robart_control(rotation_trace = FALSE, snapshots = integer(0))
}
