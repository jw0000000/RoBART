# RoBART quick start

This guide uses RoBART 1.0.1 for Gaussian regression and Bernoulli-probit
classification. Both examples use small sampling budgets to demonstrate the
API. A substantive analysis needs its own mixing and convergence assessment.

## Install and load

Install an R-compatible C++17 toolchain; Windows source installation requires
the Rtools version appropriate for your R version. Then install dependencies
and the source archive:

```r
install.packages(c("Rcpp", "RcppEigen", "nnet", "survival"))
install.packages("robart_1.0.1.tar.gz", repos = NULL, type = "source")
library(robart)
```

## Prepare numeric inputs

`robart()` requires a finite numeric training matrix and a matching numeric
response vector. Test inputs must have the same number and order of columns.
Missing and infinite values are rejected.

```r
set.seed(1)
x <- matrix(rnorm(80 * 3), nrow = 80, ncol = 3)
y <- sin(x[, 1] + x[, 2]) + rnorm(80, sd = 0.5)
x_test <- matrix(rnorm(12 * 3), nrow = 12, ncol = 3)
```

The test inputs are generated separately from the training inputs. Supply all
inputs for which you need predictions using `x.test` when fitting. Arbitrary
new-data prediction after fitting is not implemented.

## Fit Gaussian regression

```r
fit <- robart(
  x, y, x.test = x_test,
  ntree = 20L, nskip = 20L, ndpost = 30L, seed = 2L
)
summary(fit)
```

`nskip` is the number of burn-in iterations, and `ndpost` is the number of
retained prediction draws. The defaults are 200 trees, 9000 burn-in iterations
and 1000 retained draws. `seed` calls `set.seed()` before preprocessing and
leaves the caller's RNG algorithm unchanged.

## Retrieve means and posterior draws

```r
test_mean <- predict(fit)
train_mean <- predict(fit, data = "train")
test_draws <- predict(fit, type = "draws")

# Columns correspond to test observations; rows are retained draws.
test_interval <- apply(test_draws, 2L, quantile,
                       probs = c(0.025, 0.975))
```

`test_interval` contains pointwise posterior intervals for the regression
function at the supplied test inputs. These intervals do not add observation
noise, and they are not simultaneous bands across test inputs.

`predict(fit, newdata = x_test)` is not the retrieval interface and raises an
error even when those rows were supplied during fitting. Use `predict(fit)`
to access the saved test predictions.

## Inspect traces and stored fields

```r
sigma_trace <- robart_trace(fit, variable = "sigma")
plot(sigma_trace$iteration, sigma_trace$value, type = "l",
     xlab = "MCMC iteration", ylab = "Sigma")

function_trace <- robart_trace(fit, variable = "latent", index = 1L)
flat <- robart_flat(fit)
```

For Gaussian fits, `variable = "latent"` retrieves the regression-mean trace.
`robart_flat()` restores the inherited flat layout without discarding fields.
The compact result also groups data in `fit$diagnostics`, `fit$settings`,
`fit$states` and `fit$metadata`. Use `?robart_trace` to select a supported trace
and `?robart_control` to configure storage. Examine function traces and
multiple chains on the same data; rotation acceptance alone does not assess
forest convergence.

## Fit Bernoulli-probit classification

```r
set.seed(4)
z <- as.numeric(runif(nrow(x)) < pnorm(x[, 1] + 0.5 * x[, 2]))

fit_b <- robart(
  x, z, x.test = x_test, type = "pbart", offset = 0,
  ntree = 20L, nskip = 20L, ndpost = 30L, seed = 3L
)

test_probability <- predict(fit_b)
probability_draws <- predict(fit_b, type = "draws")
link_draws <- predict(fit_b, type = "draws", scale = "link")
probability_trace <- robart_trace(fit_b, variable = "probability", index = 1L)
```

The response must contain numeric `0`/`1`. Response-scale predictions apply
the probit transformation to each draw before averaging. `offset` is on the
link scale, and the probit latent standard deviation is fixed at one; the
Gaussian noise controls do not apply.

## Compare rotation settings

For a paired axis-aligned fit under the same convenience-profile prior, use
the same inputs and relevant controls with `rotation = FALSE`:

```r
fit_axis <- robart(
  x, y, x.test = x_test, rotation = FALSE,
  ntree = 20L, nskip = 20L, ndpost = 30L, seed = 2L
)
axis_mean <- predict(fit_axis)
```

This uses identity frames and fixed uniform grids. It is a comparison within
this implementation and does not imply identical fits to the upstream BART
package. Matching seeds do not imply matching random-number streams or
convergence of either chain.

## Advanced interface and reproducibility

`gbart()` exposes additional grid, DART, depth-prior and response-model
settings. Its inherited `seed` argument does not call `set.seed()`; seed the
caller before using it. The common `gbart(type = "lbart")` logit route uses
a finite 1001-term normal-scale-mixture approximation. `robart()` itself
supports only Gaussian and probit models.

Static `predict()` rejects flat `gbart()` fits, including fits with rotations
disabled. Retrieve the saved `yhat.test` or `prob.test` fields instead.
General rotated new-data prediction cannot be reconstructed from only the
final rotation or sparse state snapshots.

For reproducibility, record the package version, R and dependency versions,
input data, seed/RNG settings and all fitting arguments. Native ABI
`2026101001`, snapshot schema `9` and compact layout `1` describe separate
compatibility contracts; older saved fits are not automatically upgraded.

```r
packageVersion("robart")
sessionInfo()
citation("robart")
```

Further help is available through `?robart`, `?robart_control`, `?robart_trace`
and `?gbart`. Source provenance is available with
`system.file("COPYRIGHTS", package = "robart")`.
