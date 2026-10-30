# RoBART 1.0.1

## Public distribution documentation

* Prepare public-facing package documentation and author information under
  the name Jeongung Heo.
* Set the maintainer contact to jwheo@yonsei.ac.kr, retain original BART
  attribution, and add machine-readable software citation metadata.
* Select C++17 explicitly in both build configurations and update the
  declared system requirement; the sampler source expressions are unchanged.
* Add GitHub Actions package checks for Windows, macOS and Linux, plus
  repository and package-build exclusion rules.
* Provide Gaussian and probit examples with separately generated test inputs,
  describe the saved-input prediction interface, and distinguish its seed
  behavior from the advanced `gbart()` interface.
* Keep the package version, sampler, public R arguments and defaults unchanged
  while preparing this source for public distribution. Installation and check
  commands describe how to verify a release; they are not a passing check record.

## Full-target evaluation optimization

* Reuse validated per-tree geometry and structural-prior values. The cache key
  checks tree topology/split labels/cutpoints, coordinate weights, and depth
  parameters; it does not cache partial-residual sufficient statistics.
* Reuse existing old/new contexts in sequential cutpoint generation/evaluation.
  Refresh only touched coordinate support, fused with dynamic-uniform grid min/max.
* Reuse current birth/death prior and proposal pools. Compute the two collapsed
  scores with one routing pass, retaining observation-order parent accumulation
  and the original left-to-right full-score sums to limit floating-point changes.
* Preserve iteration-level prior traces by reading valid cached priors; no trace,
  MH prior factor, failure self-transition, empty leaf, or RNG draw is removed.
* Preserve midpoint and first-index ties while using binary search, and remove
  unused reverse-design buffer arithmetic; reverse angle-density sums remain.
* Public R arguments/defaults, registered .Call arities and snapshot schema 9
  are unchanged. Native class ABI is 2026101001; clean compilation is required.
* The unoptimized 1.0.0 full evaluators remain as reference/shadow test oracles.
  Do not resume a live 1.0.0 sampler or reuse frozen build signatures across
  these installations.

# RoBART 1.0.0

* Record the supplied implementation in the 1.0.0 source snapshot. The
  version-label change does not alter its fitting algorithm or defaults.
* Retain the full recursive discrete prior, shared midpoint/hold/failure
  cutpoint law, full-prior rotation and birth/death updates, no minimum-child
  occupancy, DART terminal correction and full-prior depth updates.
* Retain the static wbart initialization/grid/serialization correction,
  ordered_dot_v1 quantile evaluation and common finite-mixture logit update.
* Retain the package tests and finite-state reference files. Native ABI is
  2026100701, state snapshot schema is 9 and compact R result layout is 1.
* Preserve GPL version 2 or any later version terms, third-party copyright
  notices and original BART citations.
* General rotated new-data prediction remains unimplemented. Earlier fitted
  results and snapshots are not upgraded or made equivalent by changing a
  package version label.
