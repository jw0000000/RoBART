% Generated documentation for the maintained common entry.
\name{gbart}
\alias{gbart}
\title{General BART with tree-specific centered rotations}
\description{Shares the common BART/DART preprocessing and sampler entry with robart. Gaussian, probit and logit response paths are retained.}
\usage{
gbart(
  x.train,
  y.train,
  x.test=matrix(0,0,0),
  type='wbart',
  ntype=as.integer( factor(type, levels=c('wbart', 'pbart', 'lbart'))),
  sparse=FALSE,
  theta=0,
  omega=1,
  a=0.5,
  b=1,
  augment=FALSE,
  rho=NULL,
  xinfo=matrix(0,0,0),
  usequants=FALSE,
  rm.const=TRUE,
  sigest=NA,
  sigdf=3,
  sigquant=0.90,
  k=2,
  power=2,
  base=0.95,
  lambda=NA,
  tau.num=c(NA, 3, 6)[ntype],
  offset=NULL,
  w=rep(1, length(y.train)),
  ntree=c(200L, 50L, 50L)[ntype],
  numcut=100L,
  ndpost=1000L,
  nskip=100L,
  keepevery=c(1L, 10L, 10L)[ntype],
  printevery=100L,
  transposed=FALSE,
  hostname=FALSE,
  mc.cores = 1L,
  nice = 19L,
  seed = 99L,
  K = 5L,
  epsilon = 0.08,
  theta_max = 0.35 * epsilon / 0.15,
  retune_cutpoints = TRUE,
  warm_up=1000,
  initial_rotation = TRUE,
  tree_update_reps = 1,
  support_expand_prob = 0.5,
  depth_prior = c("poly", "exptilt", "nu"),
  depth_tilt = 0,
  depth_nu = 0.25,
  sample_depth_tilt = FALSE,
  depth_tilt_shape = 2,
  depth_tilt_rate = 8,
  depth_tilt_rw_sd = 0.15,
  sample_depth_nu = FALSE,
  depth_nu_a = 2,
  depth_nu_b = 2,
  depth_nu_rw_sd = 0.35,
  save_depth_hyper = NA,
  debug = FALSE,
  debug_every = 50L,
  verbose = FALSE,
  save_alignment = FALSE,
  save_rot_summary = TRUE,
  save_rot_trace = FALSE,
  save_rot_ll_trace = FALSE,
  save_rot_part_trace = FALSE,
  save_rot_angles = FALSE,
  rot_trace_every = 1L,
  rot_diag_every = 0L,
  rot_timing_every = 0L,
  save_rot_global_ll_trace = FALSE,
  save_iteration_trace = FALSE,
  true_sigma_for_trace = NA_real_,
  rotation_only_after_warmup = FALSE,
  dynamic_xinfo_on_rotation = NULL,
  rotation_xinfo_policy = c("auto", "uniform", "quantile", "global_uniform"),
  freeze_sigma_after_warmup = FALSE,
  freeze_sigma_value = NA_real_,
  save_state_iters = integer(0),
  compute_lpml = TRUE,
  rotation_enabled = TRUE
)
}
\arguments{
\item{x.train}{Training predictors: rows are observations unless transposed=TRUE. The inherited model-matrix preprocessing is retained.}
\item{y.train}{Finite numeric response vector of length n. Binary values are 0 or 1. A constant Gaussian response needs valid explicit scales if automatic calibration is degenerate.}
\item{x.test}{Optional test predictors with the training column order. An empty matrix requests no test predictions.}
\item{type}{Response model: "wbart" (Gaussian), "pbart" (probit), or "lbart" (logit). The common logit path uses a scale-then-latent update with offset and a finite 1001-term mixture approximation.}
\item{ntype}{Internal response code (1, 2, or 3), normally derived from type. Do not set inconsistently with type.}
\item{sparse}{Learn split-coordinate probabilities with a symmetric Dirichlet prior. The Dirichlet count update is used as an independence MH proposal with the full terminal-factor correction.}
\item{theta}{DART concentration: zero requests learning when sparse=TRUE; a finite positive value is fixed from initialization. This value now reaches the native setter.}
\item{omega}{Positive legacy DART value forwarded and recorded for compatibility. It has no effect in the current active predictor-level DART update; grouped selection is not implemented here.}
\item{a}{First shape parameter used by the inherited DART concentration update.}
\item{b}{Second shape parameter used by the inherited DART concentration update.}
\item{augment}{FALSE proposes coordinates from all probabilities and treats unavailable draws as failures; TRUE uses admissible-coordinate normalized proposals. Actual forward/reverse probabilities are included. No virtual augmentation counts are used.}
\item{rho}{DART concentration scaling parameter; NULL resolves to the number of processed predictors.}
\item{xinfo}{Fixed candidate locations supplied as a numeric matrix (predictors in rows and candidate indices in columns, trailing NA padding allowed) or list. The automatic grid mode is fixed for custom input. Values are copied unchanged to every initial tree frame; use initial_rotation=FALSE for raw predictor axes. Explicit dynamic or global mode is rejected.}
\item{usequants}{Initial model-matrix cutpoint rule. Also resolves rotation_xinfo_policy="auto". This R preprocessing differs from the native dynamic-quantile rule.}
\item{rm.const}{Remove constant predictor columns during the inherited model-matrix preprocessing.}
\item{sigest}{Initial or calibration noise standard deviation. If unspecified, the inherited Gaussian path uses the linear-model residual estimate when p<n, otherwise sd(y). Not by itself a sigma-freezing option.}
\item{sigdf}{Degrees of freedom of the Gaussian noise-variance prior.}
\item{sigquant}{Calibration probability used in lambda=sigest^2*qchisq(1-sigquant,sigdf)/sigdf when lambda is unspecified.}
\item{k}{Leaf shrinkage parameter. Lowercase k is distinct from the rotation plane count K.}
\item{power}{Power in the polynomial tree split prior base/(1+depth)^power; also used by the tilted-polynomial alternative.}
\item{base}{Root split probability in the polynomial and tilted-polynomial depth priors.}
\item{lambda}{Gaussian noise-prior scale. With finite lambda the inherited initialization uses sqrt(lambda); the known-sigma lambda=0 convention is retained.}
\item{tau.num}{Numerator used for the Gaussian leaf standard deviation tau=tau.num/(k*sqrt(ntree)). When missing for Gaussian regression, use half the centered response range.}
\item{offset}{Optional finite constant: response-scale for Gaussian and link-scale for probit/logit. Omitted binary offsets use the inverse link of the sample proportion; single-class binary data require an explicit finite offset. The common logit path updates the marginalized scale using the full link predictor before redrawing the latent response. Its 1001-term scale prior remains a finite approximation; the separate legacy lbart() entry is unchanged.}
\item{w}{One finite positive observation SD multiplier per response. Zero, negative, missing and infinite values are rejected.}
\item{ntree}{Number of trees; default 200 for Gaussian and 50 for probit or logit.}
\item{numcut}{Positive requested candidate count; scalar for the inherited non-transposed preprocessing. Transposed input accepts a scalar or one count per retained predictor. Custom xinfo supplies candidate positions, not a proposal width.}
\item{ndpost}{Number of retained prediction draws.}
\item{nskip}{Number of burn-in iterations before retained prediction draws. Distinct from rotation warm_up.}
\item{keepevery}{Keep one prediction draw per this many post-burn-in iterations.}
\item{printevery}{Positive integer progress-print interval. Zero is rejected before native modulo arithmetic; it is not an output-OFF convention in this version.}
\item{transposed}{If TRUE, supply native-oriented predictor matrices (predictors by observations) and bypass model-matrix preprocessing.}
\item{hostname}{Whether to record the hostname on Unix. Disabled on other platforms.}
\item{mc.cores}{Inherited compatibility argument; this gbart body does not create parallel workers.}
\item{nice}{Inherited compatibility argument; this gbart body does not change process priority.}
\item{seed}{Inherited compatibility argument. It does not call set.seed in gbart; call set.seed before fitting, or use robart(seed=...).}
\item{K}{Number of ordered Givens planes in one centered rotation proposal. This is the sole active plane-count control.}
\item{epsilon}{Positive scale of the normal angle distribution before truncation.}
\item{theta_max}{Positive truncation bound for centered angles, strictly below pi-1e-8.}
\item{retune_cutpoints}{Jointly propose affected node cutpoint indices after a rotation. Uses one discrete Gaussian with internal scale max(1,0.02*W), where W is the feasible index count.}
\item{warm_up}{Number of initial iterations with rotation updates disabled. Does not delay posterior storage independently of nskip.}
\item{initial_rotation}{Use random initial tree frames when TRUE, otherwise identity frames. Independent of whether later rotation proposals are enabled.}
\item{tree_update_reps}{Number of birth/death and leaf updates per tree after rotation warm-up.}
\item{support_expand_prob}{Probability of trying a used/unused coordinate pair when support expansion is available. Not a second rotation kernel.}
\item{depth_prior}{Depth prior: polynomial (poly), exponentially tilted polynomial (exptilt), or nu^(depth+1) (nu). There is no statistical hard depth cap.}
\item{depth_tilt}{Nonnegative exponential tilt for the exptilt depth prior.}
\item{depth_nu}{Split-decay parameter for the nu depth prior; the existing validation keeps it in (0,0.5).}
\item{sample_depth_tilt}{Update the depth tilt using its existing hyperprior and random-walk step.}
\item{depth_tilt_shape}{Shape of the Gamma prior used for the depth tilt.}
\item{depth_tilt_rate}{Rate of the Gamma prior used for the depth tilt.}
\item{depth_tilt_rw_sd}{Random-walk scale on the log depth-tilt parameter.}
\item{sample_depth_nu}{Update the nu-depth parameter using its existing hyperprior and random-walk step.}
\item{depth_nu_a}{First Beta shape for u=2*depth_nu.}
\item{depth_nu_b}{Second Beta shape for u=2*depth_nu.}
\item{depth_nu_rw_sd}{Random-walk scale on logit(2*depth_nu).}
\item{save_depth_hyper}{Save depth-hyperparameter traces. NA resolves to TRUE if either depth hyperparameter is sampled.}
\item{debug}{Enable additional native debug checks and trace collection.}
\item{debug_every}{Cadence for the native debug output/checks.}
\item{verbose}{Enable ordinary native progress and argument output. FALSE suppresses these messages; explicitly requested debug output, errors and warnings remain independent.}
\item{save_alignment}{Save the inherited fixed 45-degree reference diagnostic. It is not a general signal-recovery or convergence measure.}
\item{save_rot_summary}{Save per-tree rotation proposal and acceptance summaries.}
\item{save_rot_trace}{Save sampled rotation acceptance and tree-size traces.}
\item{save_rot_ll_trace}{Save local hard-marginal rotation likelihood traces.}
\item{save_rot_part_trace}{Save partition-change traces. A nonpositive rot_diag_every is raised to rot_trace_every when required.}
\item{save_rot_angles}{Save canonical frame-angle summaries; these are not the chronological proposal-angle sequence.}
\item{rot_trace_every}{Storage interval for rotation trace records. Use the returned rot_trace_iter, not row number, as time.}
\item{rot_diag_every}{Cadence for expensive partition diagnostics; zero disables their calculation unless another requested diagnostic requires it.}
\item{rot_timing_every}{Cadence for rotation timing collection; zero disables timing. Positive values require at least one of save_rot_trace, save_rot_ll_trace, save_rot_part_trace or save_rot_global_ll_trace, and must be a multiple of rot_trace_every. Timing is recorded only after rotation warm-up; it is not made independent of trace storage.}
\item{save_rot_global_ll_trace}{Save the global Gaussian likelihood diagnostic around rotation proposals.}
\item{save_iteration_trace}{Save iteration-level elapsed-time and available target components.}
\item{true_sigma_for_trace}{Optional reference SD for diagnostics only. Does not fix or calibrate the sampled sigma.}
\item{rotation_only_after_warmup}{Freeze structural/DART/depth-hyper updates after rotation warm-up while continuing rotation and the final leaf-mean redraw. This conditions on the retained topology.}
\item{dynamic_xinfo_on_rotation}{NULL (default) chooses dynamic for rotation ON without global/custom grids and fixed otherwise. Explicit FALSE permits fixed-grid rotation. TRUE requires rotation ON and no custom/global grid. Dynamic proposal grids are committed only on acceptance.}
\item{rotation_xinfo_policy}{auto resolves uniform/quantile using usequants. Uniform and floor-index quantile use the initial rotated training design; fixed mode retains that grid afterwards. Custom locations override the generator and are reported as fixed_custom. global_uniform always uses a fixed covering grid and rejects custom input.}
\item{freeze_sigma_after_warmup}{Freeze the Gaussian noise SD at the rotation warm-up boundary. With warm_up=0 the freeze applies from the first update.}
\item{freeze_sigma_value}{Positive SD to freeze at, or NA to use the current SD at the freeze boundary. It is a standard deviation, not a variance.}
\item{save_state_iters}{One-based full-chain iterations at which to store tree, per-tree Q, per-tree grid and sigma snapshots. NULL or an empty numeric vector requests none. Values must be finite positive integers not exceeding nskip+ndpost*keepevery; invalid requests are rejected before sampling. Valid requests are sorted and deduplicated.}
\item{compute_lpml}{Compute the inherited CPO/LPML estimate after sampling. Gaussian evaluation uses the original response, retained sigma at \code{nskip + s * keepevery}, and observation SD \code{sigma * w}. The default remains TRUE.}
\item{rotation_enabled}{Enable rotation proposals. FALSE also makes the automatic grid mode fixed but does not override initial_rotation. Use initial_rotation=FALSE for an axis-aligned BART control.}
}
\details{
The discrete prior is generated by a depth-dependent split attempt, a
coordinate draw, and a uniform admissible cutpoint draw. A failed coordinate
choice makes the node terminal. With fixed coordinate probabilities s,
internal factors are rho*s[v]/a and terminal factors are
1-rho*sum(s[G]). The manuscript case uses s[v]=1/p.
Rotation and birth/death use this same full prior, including cutpoint counts
and terminal factors. Invalid states have zero prior mass. Bounds are strict
in cutpoint values; zero training-coordinate range means no split.
There is no minimum leaf occupancy constraint or additional hard depth cap.
Empty leaves have zero collapsed log-likelihood contribution and their leaf
posterior equals the Gaussian prior. All leaf means are redrawn after the
retained collapsed structural state is chosen.

Joint cutpoint proposals use adjacent-order-statistic midpoints, the smallest
nearest index on a tie, and a single discrete Gaussian with scale max(1,0.02W).
Empty, all-left and all-right reference nodes hold their source index when
admissible. Invalid holds or empty intervals fail the whole proposal without
resampling. Forward generation and reverse evaluation share one recursion.

DART and depth-hyperparameter learning are extensions beyond the manuscript's
fixed uniform probabilities and fixed hyperparameters. The former retains
its concentration model and finite-grid concentration update, but adds the
terminal-factor MH correction. Depth-hyperparameter acceptance uses the same
full discrete prior. Omega remains inactive and grouped selection is not added.
The snapshot coordinate_log_prob and depth fields identify the conditional
prior used at that iteration. Whole posterior diagnostics remain conditional
on coordinate/depth hyperparameters; they do not include every hyperprior term.


Quantile coordinate evaluation uses the versioned ordered\_dot\_v1 rule:
separately rounded multiplication and addition in increasing input-column order.
Initialization, proposed endpoints, current caches and test/snapshot evaluation
use the same rule. An arbitrary BLAS product is not an exact substitute at
training knots. Old snapshots without this rule are not retroactively repaired.
The split rule, quantile order statistics and single-scale cutpoint proposal
are unchanged. Uniform/global coordinate arithmetic is unchanged.

The common logit path conditions the marginalized scale step on the complete
signed link predictor, including offset, then redraws the latent response at the
updated scale. It retains the existing 1001-term finite scale-prior approximation.
The legacy lbart()/clbart path is not changed by this correction.

Grid candidates use training predictors only. Dynamic uniform grids use
min + k*(max-min)/(m+1). Dynamic quantile grids use floor(k*(n-1)/(m+1))
order-statistic indices (0-based), without interpolation or tie removal.
Constant-coordinate duplicate candidates are retained.

Fixed uniform/quantile grids are generated from each tree's initial rotated
training design, once. Custom xinfo is copied unchanged to every initial frame;
its locations override the generator, and grid_mode reports fixed_custom.
Use initial_rotation=FALSE to interpret candidates in the original predictor axes.
NULL dynamic mode selects FALSE for rotation OFF, global_uniform or custom input,
and TRUE otherwise. Explicit FALSE permits fixed-grid rotation. Contradictory
TRUE requests are errors. Fixed refers to candidates, not tree or leaf updates.

Global uniform uses the padded maximum training-row norm as its covering
radius, with no user multiplier. This covers orthogonally rotated training
rows, not arbitrary future observations. No grid is rebuilt after a rotation
has been accepted. Dynamic endpoint rebuilding remains part of the joint
proposal and retains forward/reverse cutpoint masses and reject rollback.

Retuning uses a single discrete Gaussian over feasible cutpoint indices with
scale max(1,0.02*W). This scale is distinct from grid spacing and the angle
scale epsilon. Multiple affected nodes can be retuned in one proposal.
Birth proposals still require at least five observations in each proposed
child; rotation does not impose this as a new global leaf-occupancy condition.

Call set.seed() before gbart(). The seed formal does not reset R's RNG.
Saved predictions have retained-draw rows and observation columns. Rotated
prediction replay requires the same draw's tree, Q, per-tree grid and offset;
tree strings or a common static grid alone are insufficient.
}
\value{Posterior draws, prediction summaries, optional tree/rotation/grid snapshots and effective settings.}
\seealso{\code{\link{robart}}, \code{\link{robart_control}}, \code{\link{wbart}}}

\section{Input safety and DART controls}{
Iteration counts are checked in R and again at the native boundary. Invalid raw
native scalar counts are rejected before conversion to unsigned integers.
The returned \code{dart_theta_trace} contains post-tree-update concentration values
at every iteration when \code{sparse=TRUE}; initialization before DART activation
is included. \code{dart_theta_fixed}, \code{dart_theta_requested} and
\code{dart_theta_last} distinguish fixed and learned settings.
\code{dart_omega_active=FALSE} records the current nonoperative omega control.
Gaussian CPO/LPML inputs are corrected without altering sampling. Binary
postprocessing and the known DART augmentation numeric limits are unchanged.
}

\section{Gaussian postprocessing}{
Gaussian \code{sigma.mean} uses the same retained iterations with LPML on or off.
For \code{lambda=0}, the existing native omission of a sampled sigma trace is
preserved. Evaluation uses \code{sigest} until any configured warm-up switch,
then its fixed value; an unspecified switch value keeps the current known sigma.
No synthetic full sigma trace is added. A Gaussian density of zero represented
by log density \code{-Inf} gives log CPO and LPML \code{-Inf}, not \code{NaN}.
Invalid \code{NA}, \code{NaN} and positive-infinite log densities are errors.
The finite-value CPO formula and all binary postprocessing remain unchanged.
}

\seealso{\code{\link{robart}}, \code{\link{robart-package}}}
\examples{
set.seed(8)
x <- matrix(rnorm(120), 40, 3)
y <- as.numeric(x[, 1] + 0.4*x[, 2] > 0)
f <- gbart(x, y, x.test = x[1:4, , drop = FALSE], type = "pbart",
           offset = 0, ntree = 5L, nskip = 2L, ndpost = 3L,
           keepevery = 1L, initial_rotation = FALSE, warm_up = 0L,
           compute_lpml = FALSE, verbose = FALSE)
f$prob.test.mean
}
