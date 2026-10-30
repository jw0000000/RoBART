# Contributing to robart

Report a reproducible problem through this repository's Issues page. Include
the package version, `sessionInfo()`, the operating system, the random seed,
and a small example using data that can be shared publicly. For an installation
failure, include the relevant compiler output and the first error message.

## Development conventions

The public API is declared in `NAMESPACE`, and help files in `man/` are
maintained explicitly. Update the function, namespace, help, and relevant
tests together when changing a public interface. Do not regenerate all
documentation from the inherited roxygen comments without reviewing the
resulting API changes.

The native bridge is package-owned: `R/native_api.R`, `src/native_api.cpp`,
and `src/init.cpp` are maintained together. Their input guards, registered
names, and argument counts are part of the interface contract. These are not
generated `RcppExports` files; running `Rcpp::compileAttributes()` is not a
required installation or development step.

Preserve the GPL notices and upstream credits recorded in `inst/COPYRIGHTS`.
Describe user-visible changes in `NEWS.md` and explain changes to the
statistical target or saved-state compatibility explicitly.

## Checks

The package uses base-R tests under `tests/`, with fixtures and helpers under
`tests/support/`. They run as part of the standard package check. No separate
testing framework or external validation harness is required.

From the parent directory of a clean package source tree, run:

```sh
R CMD build robart
R CMD check --no-manual robart_1.0.1.tar.gz
```

Use the archive name matching `Version` in `DESCRIPTION` if the version has
changed. With a working LaTeX installation, omit `--no-manual` to check the PDF
manual too. Review errors, warnings, and notes, and retain the actual check
output. The GitHub Actions workflow checks release R on Windows, macOS, and
Linux, plus the preceding and development R versions on Linux.

Keep generated binaries, build archives, check directories, and local validation
output out of commits. Short package tests check implementation contracts;
scientific benchmark or convergence claims require their own evidence.
