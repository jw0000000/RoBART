# RoBART

**Version 1.0.1**

RoBART fits Bayesian additive regression trees with a separate orthogonal
rotation for each tree. Its main interface supports Gaussian regression and
Bernoulli-probit classification, with posterior draws and sampling diagnostics.
The package is maintained by Jeongung Heo and incorporates code from the
[BART R package](https://CRAN.R-project.org/package=BART).

**Prediction inputs must be supplied when fitting.** Pass a numeric matrix as
`x.test`; `predict(fit)` then returns the saved test predictions. Prediction on
arbitrary new data after fitting is not implemented in this version.

## Installation

This is an R source package with C++ code. Install its external R dependencies:

```r
install.packages(c("Rcpp", "RcppEigen", "nnet", "survival"))
```

Install the source from GitHub:
```r
install.packages("remotes")
remotes::install_github("jw0000000/RoBART")
```

Install the source from R:

```r
install.packages("robart_1.0.1.tar.gz", repos = NULL, type = "source")
```

Alternatively, from a terminal:

```sh
R CMD INSTALL robart_1.0.1.tar.gz
```

Compilation requires an R-compatible C++ compiler with C++17 support. On
Windows, install the Rtools version appropriate for your R installation. No
previous RoBART installation, private data or generated development bindings
are required. Install into a fresh library when comparing package versions.

## Gaussian regression

```r
library(robart)

set.seed(1)
x <- matrix(rnorm(80 * 3), nrow = 80, ncol = 3)
y <- sin(x[, 1] + x[, 2]) + rnorm(80, sd = 0.5)
x_test <- matrix(rnorm(12 * 3), nrow = 12, ncol = 3)

fit <- robart(
  x, y, x.test = x_test,
  ntree = 20L, nskip = 20L, ndpost = 30L, seed = 2L
)

summary(fit)
test_mean <- predict(fit)
test_draws <- predict(fit, type = "draws")
train_mean <- predict(fit, data = "train")

sigma_trace <- robart_trace(fit, variable = "sigma")
plot(sigma_trace$iteration, sigma_trace$value, type = "l",
     xlab = "MCMC iteration", ylab = "Sigma")
```

Rows of `test_draws` are retained draws; columns correspond to rows of
`x_test`. The test inputs above are generated separately from the training
inputs. Gaussian prediction draws describe the regression function and do not
add a new observation-noise draw.

These small iteration counts demonstrate the interface. For substantive
analysis, assess mixing and stability across chains and choose the sampling
budget accordingly. The default profile uses 200 trees, 9000 burn-in
iterations, 1000 retained prediction draws, `K = 5` and `epsilon = 0.08`.

## Bernoulli-probit classification

```r
set.seed(4)
z <- as.numeric(runif(nrow(x)) < pnorm(x[, 1] + 0.5 * x[, 2]))

fit_b <- robart(
  x, z, x.test = x_test, type = "pbart", offset = 0,
  ntree = 20L, nskip = 20L, ndpost = 30L, seed = 3L
)

test_probability <- predict(fit_b)
probability_draws <- predict(fit_b, type = "draws")
robart_trace(fit_b, variable = "probability", index = 1L)
```

The response must be numeric `0`/`1`. Probit probabilities are transformed
draw by draw before averaging. The latent standard deviation is fixed at one,
and `offset` is on the probit link scale.

## Interfaces and scope

| Interface | Supported use |
|---|---|
| `robart()` | Gaussian/probit convenience interface; finite numeric matrices; uniform grids; compact or flat output |
| `robart_control()` | Storage and diagnostic controls |
| `robart_trace()` | Function, probability and available sampler traces |
| `robart_flat()` | Lossless access to the inherited flat result fields |
| `gbart()` | Advanced Gaussian/probit/logit interface, grid options and DART settings |
| `wbart()` | Retained axis-aligned engine with separate inherited contracts |
| `predict()` on a `robart` result | Saved training or test predictions |
| `predict()` on a static `wbart` result | Guarded legacy serial new-data prediction |

`robart()` does not accept missing or infinite inputs, formulas or factor
predictors. Supply numeric matrices with matching predictor columns. Its
`seed` argument calls `set.seed()` before preprocessing and does not change
`RNGkind()`. For `gbart()`, call `set.seed()` yourself: its inherited `seed`
argument does not seed that fitting path.

`gbart(type = "lbart")` uses a finite 1001-term normal-scale-mixture
approximation for Bernoulli-logit fitting. `robart(type = "lbart")` is not
exposed. The separate legacy logit implementation is retained internally and
does not receive the same correction. Inherited factor and parallel wrappers
are not part of the convenience interface's supported contract.

Static `predict()` rejects flat `gbart()` results, including fits with rotation
disabled. Read their saved `yhat.test` or `prob.test` fields. Arbitrary rotated
new-data prediction requires each retained draw's tree, rotation and grid;
using only the final rotation is not a valid substitute.

See the [quick start](inst/doc/quick-start.md), `?robart`, `?robart_control`,
`?gbart` and `help(package = "robart")` for details.

## Model and version compatibility

The uniform-coordinate model uses the full recursive discrete prior in both
rotation and birth/death updates, including terminal-node probabilities.
Ancestor value bounds are strict; there is no minimum leaf occupancy or
additional hard depth cap. Cutpoint proposals use adjacent-order-statistic
midpoints, admissible holds for degenerate reference bags, and failure of the
whole proposal without retries.

For a paired axis-aligned comparison under the same prior, use
`robart(rotation = FALSE)`. The implementation changes inherited structural
updates, so this is not a claim of identical fits to upstream BART or older
RoBART implementations.

Advanced DART settings retain weighted internal and terminal factors and a
full-prior correction to the Dirichlet proposal. With `augment = TRUE`, birth
coordinates are normalized over admissible coordinates; no virtual
failed-split counts are used. The fixed-uniform model's theoretical guarantees
do not automatically extend to DART. Its inherited finite-grid concentration
update, floating-point limits and binary extreme-tail LPML limits remain.
Fixed polynomial, nu and tilted depth priors are available through `gbart()`;
learned depth hyperparameters use full-prior Metropolis-Hastings ratios.

Version 1.0.1 optimizes full-target evaluation while retaining the public R
arguments and defaults. Native ABI is `2026101001`, state snapshot schema is
`9`, and compact result layout is `1`. Clean compilation is required when
changing native versions. Older snapshots and experimental results are not
automatically upgraded. Quantile snapshots require the `ordered_dot_v1`
coordinate rule; arbitrary BLAS multiplication may evaluate boundary points
differently. See [NEWS.md](NEWS.md) for the change history.

## Development and checks

From the directory containing the `robart` source folder:

```sh
R CMD build robart
R CMD check --no-manual robart_1.0.1.tar.gz
```

Remove `--no-manual` to also check the PDF manual when LaTeX is available.
These are verification commands, not a recorded passing result. Record the
actual R version, platform and ERROR/WARNING/NOTE output for each release.
The tests under `tests/` run during standard `R CMD check`. Passing package
tests does not establish posterior convergence or reproduce every research
experiment.

The native bridge is maintained in `src/native_api.cpp`, `R/native_api.R`
and `src/init.cpp`. It is not generated by `Rcpp::compileAttributes()`.
`NAMESPACE` and `man/*.Rd` are also maintained explicitly. Public API changes
must keep implementation, help, namespace registration and tests consistent;
do not regenerate exports from the inherited source comments.

## Contact

The package author and maintainer is Jeongung Heo
(<jwheo@yonsei.ac.kr>). Include a small reproducible example and
`sessionInfo()` when reporting an installation or usage problem.

## License and citation

RoBART is distributed under **GPL version 2 or any later version**. Original
BART contributors retain their copyright and credit. See
[inst/COPYRIGHTS](inst/COPYRIGHTS) for source provenance and
[LICENSE.md](LICENSE.md) for the license text, also distributed in
`inst/LICENSES/GPL-2.txt`.
The exact historical BART fork commit has not been established; later upstream
metadata is used for attribution, not as a claim about that origin.

Use `citation("robart")` for the software citation and acknowledge the BART
implementation:

Sparapani, R., Spanbauer, C. and McCulloch, R. (2021).
*Nonparametric Machine Learning and Efficient Computation with Bayesian
Additive Regression Trees: The BART R Package.* Journal of Statistical
Software, 97(1), 1-66. [doi:10.18637/jss.v097.i01](https://doi.org/10.18637/jss.v097.i01).
