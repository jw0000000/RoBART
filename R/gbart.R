
## BART: Bayesian Additive Regression Trees
## Copyright (C) 2018 Robert McCulloch and Rodney Sparapani

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, a copy is available at
## https://www.R-project.org/Licenses/GPL-2

#' General BART entry with tree-specific centered rotations
#'
#' Shares response preprocessing, BART/DART updates and native sampling with
#' robart(). Gaussian, probit and logit paths are retained. The rotation/grid
#' contracts below describe this implementation, not a convergence guarantee.
#' @param x.train Training predictors: rows are observations unless transposed=TRUE. The inherited model-matrix preprocessing is retained.
#' @param y.train Finite numeric response vector of length n. Binary values are 0 or 1. A constant Gaussian response needs valid explicit scales if automatic calibration is degenerate.
#' @param x.test Optional test predictors with the training column order. An empty matrix requests no test predictions.
#' @param type Response model: "wbart" (Gaussian), "pbart" (probit), or "lbart" (logit). The common logit path uses the corrected scale-then-latent update and a finite 1001-term mixture approximation.
#' @param ntype Internal response code (1, 2, or 3), normally derived from type. Do not set inconsistently with type.
#' @param sparse Learn coordinate probabilities with a Dirichlet independence-MH proposal and the full terminal-prior correction.
#' @param theta DART concentration: zero requests learning when sparse=TRUE; a finite positive value is fixed from initialization. This value now reaches the native setter.
#' @param omega Positive legacy DART value forwarded and recorded for compatibility. It has no effect in the current active predictor-level DART update; grouped selection is not implemented here.
#' @param a First shape parameter used by the inherited DART concentration update.
#' @param b Second shape parameter used by the inherited DART concentration update.
#' @param augment FALSE proposes split coordinates from all probabilities (invalid draws self-loop); TRUE normalizes over admissible coordinates. No virtual augmentation counts are used.
#' @param rho DART concentration scaling parameter; NULL resolves to the number of processed predictors.
#' @param xinfo Fixed candidate locations supplied as a numeric matrix (predictors in rows and candidate indices in columns, trailing NA padding allowed) or list. The automatic grid mode is fixed for custom input. Values are copied unchanged to every initial tree frame; use initial_rotation=FALSE for raw predictor axes. Explicit dynamic or global mode is rejected.
#' @param usequants Initial model-matrix cutpoint rule. Also resolves rotation_xinfo_policy="auto". This R preprocessing differs from the native dynamic-quantile rule.
#' @param rm.const Remove constant predictor columns during the inherited model-matrix preprocessing.
#' @param sigest Initial or calibration noise standard deviation. If unspecified, the inherited Gaussian path uses the linear-model residual estimate when p<n, otherwise sd(y). Not by itself a sigma-freezing option.
#' @param sigdf Degrees of freedom of the Gaussian noise-variance prior.
#' @param sigquant Calibration probability used in lambda=sigest^2*qchisq(1-sigquant,sigdf)/sigdf when lambda is unspecified.
#' @param k Leaf shrinkage parameter. Lowercase k is distinct from the rotation plane count K.
#' @param power Power in the polynomial tree split prior base/(1+depth)^power; also used by the tilted-polynomial alternative.
#' @param base Root split probability in the polynomial and tilted-polynomial depth priors.
#' @param lambda Gaussian noise-prior scale. With finite lambda the inherited initialization uses sqrt(lambda); the known-sigma lambda=0 convention is retained.
#' @param tau.num Numerator used for the Gaussian leaf standard deviation tau=tau.num/(k*sqrt(ntree)). When missing for Gaussian regression, use half the centered response range.
#' @param offset Optional finite constant: response-scale for Gaussian and link-scale for probit/logit. When omitted, binary models use the inverse link of mean(y); single-class binary data require an explicit offset. The common logit path includes offset in both scale and latent-variable updates.
#' @param w One finite positive observation SD multiplier per response. Zero, negative, missing and infinite values are rejected.
#' @param ntree Number of trees; default 200 for Gaussian and 50 for probit or logit.
#' @param numcut Positive requested candidate count; scalar for the inherited non-transposed preprocessing. Transposed input accepts a scalar or one count per retained predictor. Custom xinfo supplies candidate positions, not a proposal width.
#' @param ndpost Number of retained prediction draws.
#' @param nskip Number of burn-in iterations before retained prediction draws. Distinct from rotation warm_up.
#' @param keepevery Keep one prediction draw per this many post-burn-in iterations.
#' @param printevery Positive integer progress-print interval. Zero is rejected before native modulo arithmetic; it is not an output-OFF convention in this version.
#' @param transposed If TRUE, supply native-oriented predictor matrices (predictors by observations) and bypass model-matrix preprocessing.
#' @param hostname Whether to record the hostname on Unix. Disabled on other platforms.
#' @param mc.cores Inherited compatibility argument; this gbart body does not create parallel workers.
#' @param nice Inherited compatibility argument; this gbart body does not change process priority.
#' @param seed Inherited compatibility argument. It does not call set.seed in gbart; call set.seed before fitting, or use robart(seed=...).
#' @param K Number of ordered Givens planes in one centered rotation proposal. This is the sole active plane-count control.
#' @param epsilon Positive scale of the normal angle distribution before truncation.
#' @param theta_max Positive truncation bound for centered angles, strictly below pi-1e-8.
#' @param retune_cutpoints Jointly propose affected node cutpoint indices after a rotation. Uses one discrete Gaussian with internal scale max(1,0.02*W), where W is the feasible index count.
#' @param warm_up Number of initial iterations with rotation updates disabled. Does not delay posterior storage independently of nskip.
#' @param initial_rotation Use random initial tree frames when TRUE, otherwise identity frames. Independent of whether later rotation proposals are enabled.
#' @param tree_update_reps Number of birth/death and leaf updates per tree after rotation warm-up.
#' @param support_expand_prob Probability of trying a used/unused coordinate pair when support expansion is available. Not a second rotation kernel.
#' @param depth_prior Depth prior: polynomial (poly), exponentially tilted polynomial (exptilt), or nu^(depth+1) (nu). There is no statistical hard depth cap.
#' @param depth_tilt Nonnegative exponential tilt for the exptilt depth prior.
#' @param depth_nu Split-decay parameter for the nu depth prior; the existing validation keeps it in (0,0.5).
#' @param sample_depth_tilt Update depth tilt with its existing hyperprior and log-scale proposal, evaluating the full discrete tree prior.
#' @param depth_tilt_shape Shape of the Gamma prior used for the depth tilt.
#' @param depth_tilt_rate Rate of the Gamma prior used for the depth tilt.
#' @param depth_tilt_rw_sd Random-walk scale on the log depth-tilt parameter.
#' @param sample_depth_nu Update the nu-depth parameter with its existing hyperprior and logit-scale proposal, evaluating the full discrete tree prior.
#' @param depth_nu_a First Beta shape for u=2*depth_nu.
#' @param depth_nu_b Second Beta shape for u=2*depth_nu.
#' @param depth_nu_rw_sd Random-walk scale on logit(2*depth_nu).
#' @param save_depth_hyper Save depth-hyperparameter traces. NA resolves to TRUE if either depth hyperparameter is sampled.
#' @param debug Enable additional native debug checks and trace collection.
#' @param debug_every Cadence for the native debug output/checks.
#' @param verbose Enable ordinary native progress and argument output. FALSE suppresses these messages; explicitly requested debug output, errors and warnings remain independent.
#' @param save_alignment Save the inherited fixed 45-degree reference diagnostic. It is not a general signal-recovery or convergence measure.
#' @param save_rot_summary Save per-tree rotation proposal and acceptance summaries.
#' @param save_rot_trace Save sampled rotation acceptance and tree-size traces.
#' @param save_rot_ll_trace Save leaf-integrated likelihood terms. The log acceptance also includes the full discrete-prior and cutpoint-proposal ratios.
#' @param save_rot_part_trace Save partition-change traces. A nonpositive rot_diag_every is raised to rot_trace_every when required.
#' @param save_rot_angles Save canonical frame-angle summaries; these are not the chronological proposal-angle sequence.
#' @param rot_trace_every Storage interval for rotation trace records. Use the returned rot_trace_iter, not row number, as time.
#' @param rot_diag_every Cadence for expensive partition diagnostics; zero disables their calculation unless another requested diagnostic requires it.
#' @param rot_timing_every Cadence for rotation timing collection; zero disables timing. Positive values require at least one of save_rot_trace, save_rot_ll_trace, save_rot_part_trace or save_rot_global_ll_trace, and must be a multiple of rot_trace_every. Timing is recorded only after rotation warm-up; it is not made independent of trace storage.
#' @param save_rot_global_ll_trace Save the global Gaussian likelihood diagnostic around rotation proposals.
#' @param save_iteration_trace Save iteration-level elapsed-time and available target components.
#' @param true_sigma_for_trace Optional reference SD for diagnostics only. Does not fix or calibrate the sampled sigma.
#' @param rotation_only_after_warmup Freeze structural/DART/depth-hyper updates after rotation warm-up while continuing rotation and the final leaf-mean redraw. This conditions on the retained topology.
#' @param dynamic_xinfo_on_rotation NULL (default) chooses dynamic for rotation ON without global/custom grids and fixed otherwise. Explicit FALSE permits fixed-grid rotation. TRUE requires rotation ON and no custom/global grid. Dynamic proposal grids are committed only on acceptance.
#' @param rotation_xinfo_policy auto resolves uniform/quantile using usequants. Uniform and floor-index quantile use the initial rotated training design; fixed mode retains that grid afterwards. Custom locations override the generator and are reported as fixed_custom. global_uniform always uses a fixed covering grid and rejects custom input.
#' @param freeze_sigma_after_warmup Freeze the Gaussian noise SD at the rotation warm-up boundary. With warm_up=0 the freeze applies from the first update.
#' @param freeze_sigma_value Positive SD to freeze at, or NA to use the current SD at the freeze boundary. It is a standard deviation, not a variance.
#' @param save_state_iters One-based full-chain iterations at which to store tree, per-tree Q, per-tree grid and sigma snapshots. NULL or an empty numeric vector requests none. Values must be finite positive integers not exceeding nskip+ndpost*keepevery; invalid requests are rejected before sampling. Valid requests are sorted and deduplicated.
#' @param compute_lpml Compute the inherited CPO/LPML estimate after sampling. Gaussian evaluation uses the original response, matched retained sigma iterations and observation SD sigma*w. The default remains TRUE; robart() disables this calculation.
#' @param rotation_enabled Enable rotation proposals. FALSE also makes the automatic grid mode fixed but does not override initial_rotation. Use initial_rotation=FALSE for an axis-aligned BART control.
#' @details
#' Grid candidates use training predictors only. Dynamic uniform grids use
#' min + k*(max-min)/(m+1). Dynamic quantile grids use floor(k*(n-1)/(m+1))
#' order-statistic indices (0-based), without interpolation or tie removal.
#' Constant-coordinate duplicate candidates are retained.
#'
#' Fixed uniform/quantile grids are generated from each tree's initial rotated
#' training design, once. Custom xinfo is copied unchanged to every initial frame;
#' its locations override the generator, and grid_mode reports fixed_custom.
#' Use initial_rotation=FALSE to interpret candidates in the original predictor axes.
#' NULL dynamic mode selects FALSE for rotation OFF, global_uniform or custom input,
#' and TRUE otherwise. Explicit FALSE permits fixed-grid rotation. Contradictory
#' TRUE requests are errors. Fixed refers to candidates, not tree or leaf updates.
#'
#' Global uniform uses the padded maximum training-row norm as its covering
#' radius, with no user multiplier. This covers orthogonally rotated training
#' rows, not arbitrary future observations. No grid is rebuilt after a rotation
#' has been accepted. Dynamic endpoint rebuilding remains part of the joint
#' proposal and retains forward/reverse cutpoint masses and reject rollback.
#'
#' Retuning uses a single discrete Gaussian over feasible cutpoint indices with
#' scale max(1,0.02*W). This scale is distinct from grid spacing and the angle
#' scale epsilon. Multiple affected nodes can be retuned in one proposal.
#' Birth proposals still require at least five observations in each proposed
#' child; rotation does not impose this as a new global leaf-occupancy condition.
#'
#' Call set.seed() before gbart(). The seed formal does not reset R's RNG.
#' Saved predictions have retained-draw rows and observation columns. Rotated
#' prediction replay requires the same draw's tree, Q, per-tree grid and offset;
#' tree strings or a common static grid alone are insufficient.
#' Invalid iteration/count/weight inputs are rejected before native sampling. Entirely missing predictor columns are rejected instead of entering a nonterminating hot-deck loop. Partial-missing and factor-schema policies are not redesigned here.
#' Gaussian sigma.mean uses retained prediction iterations in either compute_lpml mode.
#' For lambda=0 no synthetic full sigma trace is added: the known sigest and any
#' configured warm-up switch define evaluation SDs. A Gaussian log density of
#' -Inf yields log CPO and LPML of -Inf; invalid NA/NaN/+Inf densities are rejected.
#' Probit/logit likelihood postprocessing is unchanged in this patch.
#' @details Quantile states use the versioned ordered_dot_v1 coordinate rule for initialization, proposal endpoints, and replay. Replay must use that rule rather than an arbitrary BLAS reduction at training knots; historical snapshots are not relabelled. Uniform/global coordinate arithmetic is unchanged. The common logit path uses collapsed scale MH followed by a latent-response update conditional on the new scale.
#' @return Posterior draws, predictions, optional tree/rotation/grid snapshots, and effective settings. Binary outputs keep the inherited latent/probability conventions.
#' @export
gbart=function(
    x.train, y.train,
    x.test=matrix(0,0,0), type='wbart',
    ntype=as.integer(
      factor(type, levels=c('wbart', 'pbart', 'lbart'))),
    sparse=FALSE, theta=0, omega=1,
    a=0.5, b=1, augment=FALSE, rho=NULL,
    xinfo=matrix(0,0,0), usequants=FALSE,
    rm.const=TRUE,
    sigest=NA, sigdf=3, sigquant=0.90,
    k=2, power=2, base=0.95,
    lambda=NA, tau.num=c(NA, 3, 6)[ntype],
    offset=NULL, w=rep(1, length(y.train)),
    ntree=c(200L, 50L, 50L)[ntype], numcut=100L,
    ndpost=1000L, nskip=100L,
    keepevery=c(1L, 10L, 10L)[ntype],
    printevery=100L, transposed=FALSE,
    hostname=FALSE,
    mc.cores = 1L, nice = 19L, seed = 99L, K = 5L, epsilon = 0.08,
    theta_max = 0.35 * epsilon / 0.15, retune_cutpoints = TRUE, warm_up=1000,
    initial_rotation = TRUE, tree_update_reps = 1,
    support_expand_prob = 0.5,
    # --- depth prior options  ---
    # depth_prior:
    #   - "poly"    : alpha/(1+d)^beta (original BART)
    #   - "exptilt" : alpha/(1+d)^beta * exp(-depth_tilt * d)
    #   - "nu"      : depth_nu^(d+1)
    depth_prior = c("poly", "exptilt", "nu"),
    depth_tilt = 0,
    depth_nu = 0.25,
    # --- optional hyperpriors for depth prior parameters  ---
    sample_depth_tilt = FALSE,
    depth_tilt_shape = 2,
    depth_tilt_rate  = 8,
    depth_tilt_rw_sd = 0.15,
    sample_depth_nu = FALSE,
    depth_nu_a = 2,
    depth_nu_b = 2,
    depth_nu_rw_sd = 0.35,
    # If NA: defaults to TRUE when any sample_depth_* is TRUE
    save_depth_hyper = NA,
    debug = FALSE, debug_every = 50L,
    # --- performance / output controls  ---
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
{
  .robart_validate_gbart_arguments(environment())
  if (is.null(x.test)) x.test <- matrix(0, 0L, 0L)

  if (length(rotation_enabled) != 1L || !is.logical(rotation_enabled) || is.na(rotation_enabled))
    stop("rotation_enabled must be TRUE or FALSE.", call. = FALSE)

  # gbart remains the common BART entry. These are the only angle controls.
  if (!is.numeric(K) || length(K) != 1L || !is.finite(K) ||
      K < 1 || K != floor(K) || K > .Machine$integer.max)
    stop("K must be one positive integer (the number of Givens planes).", call. = FALSE)
  K <- as.integer(K)
  if (!is.numeric(epsilon) || length(epsilon) != 1L ||
      !is.finite(epsilon) || epsilon <= 0)
    stop("epsilon must be one finite positive angle scale.", call. = FALSE)
  if (!is.numeric(theta_max) || length(theta_max) != 1L ||
      !is.finite(theta_max) || theta_max <= 0 || theta_max >= pi - 1e-8)
    stop("theta_max must be finite and strictly between 0 and pi - 1e-8.", call. = FALSE)
  if (!is.logical(retune_cutpoints) || length(retune_cutpoints) != 1L || is.na(retune_cutpoints))
    stop("retune_cutpoints must be TRUE or FALSE.", call. = FALSE)

  if (length(compute_lpml) != 1L || is.na(compute_lpml) || !is.logical(compute_lpml))
    stop("compute_lpml must be TRUE or FALSE.", call. = FALSE)

  if(is.na(ntype))
    stop("type argument must be set to either 'wbart', 'pbart' or 'lbart'")

  n = length(y.train)

  xinfo <- .robart_prepare_xinfo(xinfo)
  user_xinfo_supplied <- length(xinfo) > 0L
  grid_contract <- .robart_resolve_grid(rotation_enabled,
    dynamic_xinfo_on_rotation, match.arg(rotation_xinfo_policy), usequants,
    user_xinfo_supplied)
  dynamic_xinfo_on_rotation <- grid_contract$dynamic
  rotation_xinfo_policy <- grid_contract$policy
  if (user_xinfo_supplied) {
    input_p <- if (transposed) nrow(x.train) else ncol(bartModelMatrix(x.train))
    if (nrow(xinfo) != input_p)
      stop("Custom xinfo must have one row for each expanded predictor before constant-column removal.", call. = FALSE)
    if (transposed) numcut <- as.integer(rowSums(!is.na(xinfo)))
  }

  if(!transposed) {
    temp = bartModelMatrix(x.train, numcut, usequants=usequants,
                           xinfo=xinfo, rm.const=rm.const)
    x.train = t(temp$X)
    numcut = temp$numcut
    xinfo = temp$xinfo
    if(length(x.test)>0) {
      x.test = bartModelMatrix(x.test)
      x.test = t(x.test[ , temp$rm.const, drop = FALSE])
    }
    rm.const <- temp$rm.const
    grp <- temp$grp
    rm(temp)
  }
  else {
    rm.const <- NULL
    grp <- NULL
  }

  if(n!=ncol(x.train))
    stop('The length of y.train and the number of rows in x.train must be identical')

  p = nrow(x.train)
  np = ncol(x.test)
  if (transposed && length(numcut) == 1L) numcut <- rep(as.integer(numcut), p)
  .robart_validate_processed_design(x.train, x.test, y.train, w, numcut)
  if(length(rho)==0) rho=p
  if(length(rm.const)==0) rm.const <- 1:p
  if(length(grp)==0) grp <- 1:p

  check <- unique(sort(y.train))

  if(length(check)==2) {
    if(!all(check==0:1))
      stop('Binary y.train must be coded as 0 and 1')
    if(type=='wbart')
      stop("The outcome is binary so set type to 'pbart' or 'lbart'")
  }



  if(length(offset)==0) {
    offset=mean(y.train)
    if(type=='pbart') offset=qnorm(offset)
    else if(type=='lbart') offset=qlogis(offset)
  }

  if(type=='wbart') {
    # Keep the observed response for density evaluation. Native predictions
    # already include offset; the sampler still receives the centered response.
    y.train.for_lpml <- y.train
    y.train = y.train-offset

    if(!is.na(sigest) && !is.na(lambda) && lambda==0) {
      ##no op: sigma is fixed and known at given sigest value
    }
    else if(is.na(lambda)) {
      if(is.na(sigest)) {
        if(p < n)
          sigest = summary(lm(y.train~.,
                              data.frame(t(x.train),y.train)))$sigma
        else sigest = sd(y.train)
      }
      qchi = qchisq(1-sigquant, sigdf)
      lambda = (sigest^2)*qchi/sigdf #lambda parameter for sigma prior
    } else {
      sigest=sqrt(lambda)
    }

    if(is.na(tau.num)) {
      tau=(max(y.train)-min(y.train))/(2*k*sqrt(ntree))
    } else {
      tau=tau.num/(k*sqrt(ntree))
    }
  } else {
    lambda=1
    sigest=1
    tau=tau.num/(k*sqrt(ntree))

  }

  ## hot deck missing imputation
  ## must be conducted here since it would
  ## cause trouble with multi-threading on the C++ side

  check=(np>0 && np==n)

  for(i in 1:n)
    for(j in 1:p) {
      if(check) check=((is.na(x.train[j, i]) && is.na(x.test[j, i])) ||
                         (!is.na(x.train[j, i]) && !is.na(x.test[j, i]) &&
                            x.train[j, i]==x.test[j, i]))

      while(is.na(x.train[j, i])) {
        h=sample.int(n, 1)
        x.train[j, i]=x.train[j, h]
      }
    }

  if(check) x.test=x.train
  else if(np>0) {
    for(i in 1:np)
      for(j in 1:p)
        while(is.na(x.test[j, i])) {
          h=sample.int(np, 1)
          x.test[j, i]=x.test[j, h]
        }
  }


  .robart_validate_effective_scales(tau, sigest, lambda, offset, y.train, w, x.train, x.test)

  if(.Platform$OS.type!='unix') hostname <- FALSE
  else if(hostname)
    hostname <- system('hostname', intern=TRUE)

  ptm <- proc.time()

  # Normalize trace cadence; these controls do not choose a sampler kernel.
  if (is.null(rot_trace_every) || is.na(rot_trace_every) || rot_trace_every < 1L)
    rot_trace_every <- 1L

  save_rot_global_ll_trace <- isTRUE(save_rot_global_ll_trace)
  save_iteration_trace <- isTRUE(save_iteration_trace)
  true_sigma_for_trace <- as.numeric(true_sigma_for_trace)[1]
  rotation_only_after_warmup <- isTRUE(rotation_only_after_warmup)
  # Grid mode was resolved before preprocessing. No contradictory setting is
  # silently changed, and custom candidate values never enter a dynamic grid.
  support_expand_prob <- as.numeric(support_expand_prob)[1]
  if (is.na(support_expand_prob)) support_expand_prob <- 0.5
  support_expand_prob <- max(0, min(1, support_expand_prob))

  if (dynamic_xinfo_on_rotation && user_xinfo_supplied)
    stop("dynamic_xinfo_on_rotation=TRUE is incompatible with a user-supplied fixed xinfo matrix")
  freeze_sigma_after_warmup <- isTRUE(freeze_sigma_after_warmup)
  freeze_sigma_value <- as.numeric(freeze_sigma_value)[1]
  # The shared input check has already rejected fractions, NA and out-of-range
  # requests. Only order/duplicates are normalized here, never invalid values.
  save_state_iters <- sort(unique(as.integer(save_state_iters)))

  # Bundle experimental rotation/sigma diagnostics into one object before
  # calling cgbart().  This keeps the generated .Call wrapper below R's
  # top-level argument limit.
  # The native boundary rejects unknown names. The resolved grid mode and the
  # user-supplied-grid flag are explicit; initial_rotation is a separate choice.
  extra_ctrl <- list(
    rotation_enabled = rotation_enabled,
    save_rot_global_ll_trace = save_rot_global_ll_trace,
    save_iteration_trace = save_iteration_trace,
    true_sigma_for_trace = true_sigma_for_trace,
    rotation_only_after_warmup = rotation_only_after_warmup,
    dynamic_xinfo_on_rotation = dynamic_xinfo_on_rotation,
    rotation_xinfo_policy = rotation_xinfo_policy,
    user_xinfo_supplied = user_xinfo_supplied,
    support_expand_prob = support_expand_prob,
    rotation_angle_bound = theta_max,
    rotation_retune_cutpoints = retune_cutpoints,
    freeze_sigma_after_warmup = freeze_sigma_after_warmup,
    freeze_sigma_value = freeze_sigma_value,
    save_state_iters = save_state_iters
  )

  # --- depth prior option mapping  ---
  depth_prior <- match.arg(depth_prior)
  depth_prior_type <- switch(depth_prior,
                             poly = 0L,
                             exptilt = 1L,
                             nu = 2L)
  if (is.null(depth_tilt) || is.na(depth_tilt) || depth_tilt < 0) depth_tilt <- 0
  if (is.null(depth_nu) || is.na(depth_nu)) depth_nu <- 0.25

  # --- optional hyperprior sanitation  ---
  sample_depth_tilt <- isTRUE(sample_depth_tilt)
  sample_depth_nu   <- isTRUE(sample_depth_nu)

  if (is.na(save_depth_hyper))
    save_depth_hyper <- (sample_depth_tilt || sample_depth_nu)
  save_depth_hyper <- isTRUE(save_depth_hyper)

  depth_tilt_shape <- as.numeric(depth_tilt_shape)
  depth_tilt_rate  <- as.numeric(depth_tilt_rate)
  depth_tilt_rw_sd <- as.numeric(depth_tilt_rw_sd)
  if (!is.finite(depth_tilt_shape) || depth_tilt_shape <= 0) depth_tilt_shape <- 2
  if (!is.finite(depth_tilt_rate)  || depth_tilt_rate  <= 0) depth_tilt_rate  <- 8
  if (!is.finite(depth_tilt_rw_sd) || depth_tilt_rw_sd <= 0) depth_tilt_rw_sd <- 0.15

  depth_nu_a     <- as.numeric(depth_nu_a)
  depth_nu_b     <- as.numeric(depth_nu_b)
  depth_nu_rw_sd <- as.numeric(depth_nu_rw_sd)
  if (!is.finite(depth_nu_a) || depth_nu_a <= 0) depth_nu_a <- 2
  if (!is.finite(depth_nu_b) || depth_nu_b <= 0) depth_nu_b <- 2
  if (!is.finite(depth_nu_rw_sd) || depth_nu_rw_sd <= 0) depth_nu_rw_sd <- 0.35

  # keep nu strictly inside (0, 1/2) when using the nu-depth prior
  if (depth_prior_type == 2L) {
    if (!is.finite(depth_nu)) depth_nu <- 0.25
    depth_nu <- max(min(depth_nu, 0.499999), 1e-8)
  }

  # partition diagnostics are only meaningful if diagnostics are computed
  if (save_rot_part_trace && (is.null(rot_diag_every) || is.na(rot_diag_every) || rot_diag_every <= 0L))
    rot_diag_every <- rot_trace_every

  # res = .Call("cgbart",
  res = cgbart(ntype, ##as.integer(factor(type, levels=check))-1,
               n,  #number of observations in training data
               p,  #dimension of x
               np, #number of observations in test data
               x.train,   #pxn training data x
               y.train,   # length-n response (centered when required)
               x.test,    #p*np test data x
               ntree,
               numcut,
               ndpost*keepevery,
               nskip,
               keepevery,
               power,
               base,
               offset,
               tau,
               sigdf,
               lambda,
               sigest,
               w,
               sparse,
               theta,
               omega,
               epsilon,
               K,
               grp,
               a,
               b,
               rho,
               augment,
               printevery,
               xinfo,
               warm_up,
               initial_rotation,
               tree_update_reps,
               debug,
               debug_every,
               # IO/throttling options
               verbose,
               save_alignment,
               save_rot_summary,
               save_rot_trace,
               save_rot_ll_trace,
               save_rot_part_trace,
               save_rot_angles,
               rot_trace_every,
               rot_diag_every,
               rot_timing_every,
               # depth prior options
               depth_prior_type,
               depth_tilt,
               depth_nu,
               # depth prior hyperpriors
               sample_depth_tilt,
               depth_tilt_shape,
               depth_tilt_rate,
               depth_tilt_rw_sd,
               sample_depth_nu,
               depth_nu_a,
               depth_nu_b,
               depth_nu_rw_sd,
               save_depth_hyper,
               extra_ctrl
  )

  res$proc.time <- proc.time()-ptm
  res$hostname <- hostname
  res$w <- w
  res$true_sigma_for_trace <- true_sigma_for_trace
  res$rotation_kernel_label <- "centered"
  res$rotation_retune_cutpoints <- retune_cutpoints
  res$grid_mode <- grid_contract$mode
  res$grid_mode_selection <- grid_contract$selection
  res$grid_mode_automatic <- grid_contract$automatic
  res$grid_policy_requested <- grid_contract$requested_policy

  # Evaluation uses the same iteration as each retained prediction row.
  # Do not add a synthetic posterior sigma trace for a known-sigma fit.
  if (type == "wbart") {
    sigma.kept <- .robart_retained_sigma_for_gaussian(
      res, ndpost, nskip, keepevery, lambda, sigest, warm_up,
      freeze_sigma_after_warmup, freeze_sigma_value)
  }

  # Keep the inherited CPO definition and binary postprocessing unchanged.
  if (compute_lpml) {
    Y=t(matrix(if (type == "wbart") y.train.for_lpml else y.train,
               nrow=n, ncol=ndpost))
  
    if(type=='wbart') {
      res$yhat.train.mean <- apply(res$yhat.train, 2, mean)
      SD <- outer(sigma.kept, w, "*")
      if (any(!is.finite(SD)) || any(SD <= 0))
        stop("Retained observation SDs overflow or underflow; check sigma and w.", call. = FALSE)
      log.pdf=dnorm(Y, res$yhat.train, SD, TRUE)
      res$sigma.mean=mean(sigma.kept)
    }
    else {
      if(type=='pbart') res$prob.train = pnorm(res$yhat.train)
      else if(type=='lbart') res$prob.train = plogis(res$yhat.train)
  
      ##CPO=1/apply(1/dbinom(Y, 1, res$prob.train), 2, mean)
      log.pdf=dbinom(Y, 1, res$prob.train, TRUE)
  
      res$prob.train.mean <- apply(res$prob.train, 2, mean)
    }
  
    if (type == "wbart") {
      log.CPO <- .robart_log_cpo_for_gaussian(log.pdf)
    } else {
      min.log.pdf=t(matrix(apply(log.pdf, 2, min), nrow=n, ncol=ndpost))
      log.CPO=log(ndpost)+min.log.pdf[1, ]-
        log(apply(exp(min.log.pdf-log.pdf), 2, sum))
    }
    res$LPML=sum(log.CPO)
    ##res$CPO=exp(log.CPO)
    ##res$LPML=sum(log(CPO))
  } else {
    res$yhat.train.mean <- colMeans(res$yhat.train)
    if (type == "wbart") {
      res$sigma.mean <- mean(sigma.kept)
    } else {
      res$prob.train <- if (type == "pbart") pnorm(res$yhat.train) else plogis(res$yhat.train)
      res$prob.train.mean <- colMeans(res$prob.train)
    }
  }

  if(np>0) {
    if(type=='wbart')
      res$yhat.test.mean <- apply(res$yhat.test, 2, mean)
    else {
      if(type=='pbart') res$prob.test = pnorm(res$yhat.test)
      else if(type=='lbart') res$prob.test = plogis(res$yhat.test)

      res$prob.test.mean <- apply(res$prob.test, 2, mean)
    }
  }

  res$ndpost = ndpost
  res$offset = offset
  if (!is.null(res$treedraws$cutpoints))
    names(res$treedraws$cutpoints) = dimnames(x.train)[[1]]
  dimnames(res$varcount)[[2]] = as.list(dimnames(x.train)[[1]])
  dimnames(res$varprob)[[2]] = as.list(dimnames(x.train)[[1]])
  res$varcount.mean <- apply(res$varcount, 2, mean)
  res$varprob.mean <- apply(res$varprob, 2, mean)
  res$rm.const <- rm.const
  attr(res, 'class') <- type
  return(res)
}


# Internal Gaussian evaluation helpers; no model fitting, new draws, or RNG calls.
# The trace is full-chain, whereas yhat.train rows are burn + s*thin.
.robart_retained_sigma_for_gaussian <- function(res, ndpost, nskip, keepevery,
                                                lambda, sigest, warm_up,
                                                freeze_after_warmup, freeze_value) {
  idx <- nskip + seq_len(ndpost) * keepevery
  total <- nskip + ndpost * keepevery
  if (lambda == 0) {
    # Known sigma is not sampled or returned by the native routine. Reconstruct
    # only the deterministic evaluation schedule, including a requested switch.
    if (length(sigest) != 1L || !is.finite(sigest) || sigest <= 0)
      stop("Known-sigma evaluation needs a finite positive sigest.", call. = FALSE)
    out <- rep(sigest, ndpost)
    final.sigma <- sigest
    if (isTRUE(freeze_after_warmup) &&
        length(freeze_value) == 1L && is.finite(freeze_value) && freeze_value > 0) {
      first.frozen <- warm_up + 1
      out[idx >= first.frozen] <- freeze_value
      if (total >= first.frozen) final.sigma <- freeze_value
    }
    # This check detects an unsupported future native schedule rather than
    # silently substituting the final sigma for every retained iteration.
    last <- res[["sigma_last", exact = TRUE]]
    if (length(last) != 1L || !is.finite(last) || last != final.sigma)
      stop("Known-sigma evaluation schedule disagrees with native sigma_last.", call. = FALSE)
  } else {
    # Exact extraction matters: sigma_last must not partially match sigma.
    trace <- res[["sigma", exact = TRUE]]
    if (!is.numeric(trace) || !is.null(dim(trace)) || length(trace) != total)
      stop("Gaussian sigma trace does not match the full iteration count.", call. = FALSE)
    out <- trace[idx]
  }
  if (length(out) != ndpost || any(!is.finite(out)) || any(out <= 0))
    stop("Retained Gaussian sigma values must be finite and positive.", call. = FALSE)
  out
}

.robart_log_cpo_for_gaussian <- function(log.pdf) {
  if (!is.matrix(log.pdf) || !is.numeric(log.pdf) ||
      nrow(log.pdf) < 1L || ncol(log.pdf) < 1L)
    stop("Gaussian log densities must be a nonempty draw-by-observation matrix.", call. = FALSE)
  if (anyNA(log.pdf) || any(log.pdf == Inf))
    stop("Gaussian log densities contain NA, NaN, or positive infinity.", call. = FALSE)

  # Keep the original finite-value sum and reduction order. A -Inf density
  # makes this harmonic-mean estimate of log CPO equal to -Inf, not NaN.
  valid <- colSums(log.pdf == -Inf) == 0L
  out <- rep(-Inf, ncol(log.pdf))
  if (any(valid)) {
    ell <- log.pdf[, valid, drop = FALSE]
    S <- nrow(ell)
    m <- ncol(ell)
    min.log.pdf <- t(matrix(apply(ell, 2L, min), nrow = m, ncol = S))
    out[valid] <- log(S) + min.log.pdf[1L, ] -
      log(apply(exp(min.log.pdf - ell), 2L, sum))
  }
  out
}

#' Summarize accepted rotation proposals using global log-likelihood differences
#' Internal diagnostic; not exported.
rot_accepted_global_effect <- function(fit, aggregate = c("sum", "mean"), cumulative = FALSE) {
  aggregate <- match.arg(aggregate)
  if (is.null(fit$global_rot_ll_diff_trace_full) || is.null(fit$rot_accept_trace_full) || is.null(fit$rot_trace_iter)) {
    stop("fit must contain global rotation ll trace and accept trace. Rerun gbart with save_rot_trace=TRUE and save_rot_global_ll_trace=TRUE.")
  }
  d <- fit$global_rot_ll_diff_trace_full
  a <- fit$rot_accept_trace_full
  if (!all(dim(d) == dim(a))) stop("trace dimensions do not match")
  d[a != 1L] <- NA_real_
  nacc <- rowSums(a == 1L, na.rm = TRUE)
  by_iter <- if (aggregate == "sum") rowSums(d, na.rm = TRUE) else rowMeans(d, na.rm = TRUE)
  by_iter[nacc == 0L] <- NA_real_
  cum <- cumsum(replace(by_iter, is.na(by_iter), 0))
  out <- data.frame(iter = fit$rot_trace_iter, accepted = nacc, value = by_iter, cumulative = cum)
  if (isTRUE(cumulative)) out$cumulative else out
}

#' Profile one tree around a saved rotation-BART state
#' Internal diagnostic; not a supported public prediction interface.
rot_profile_snapshot <- function(fit, x.train, y.train,
                                 snapshot = 1L,
                                 tree_index = 1L,
                                 plane = c(1L, 2L),
                                 theta_grid = seq(-0.4, 0.4, length.out = 81),
                                 w = NULL,
                                 transposed = FALSE,
                                 sigma_true = NA_real_) {
  if (is.null(fit$saved_states) || length(fit$saved_states) == 0L) {
    stop("fit$saved_states is empty. Rerun gbart with save_state_iters set.")
  }
  snapshot <- as.integer(snapshot)[1]
  if (is.na(snapshot) || snapshot < 1L || snapshot > length(fit$saved_states)) {
    stop("snapshot index is out of range")
  }
  plane <- as.integer(plane)
  if (length(plane) != 2L || anyNA(plane) || plane[1] == plane[2]) stop("plane must have two distinct indices")

  st <- fit$saved_states[[snapshot]]
  x.train <- as.matrix(x.train)
  if (!transposed) {
    keep <- fit$rm.const
    if (!is.null(keep)) x.train <- x.train[, keep, drop = FALSE]
    x.use <- t(x.train)
  } else {
    x.use <- x.train
  }
  y.use <- as.numeric(y.train) - fit$offset
  if (is.null(w)) {
    if (!is.null(fit$w)) w <- fit$w else w <- rep(1, length(y.use))
  }
  if (length(w) != length(y.use)) stop("w must have length n")
  if (is.na(sigma_true) && !is.null(fit$true_sigma_for_trace) && is.finite(fit$true_sigma_for_trace)) {
    sigma_true <- fit$true_sigma_for_trace
  }

  cgbart_profile_snapshot(
    as.numeric(x.use),
    as.numeric(y.use),
    as.numeric(w),
    st,
    fit$tau,
    as.numeric(theta_grid),
    as.integer(tree_index)[1],
    plane[1],
    plane[2],
    as.numeric(st$sigma),
    as.numeric(sigma_true)[1]
  )
}
