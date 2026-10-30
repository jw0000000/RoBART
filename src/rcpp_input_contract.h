#ifndef ROBART_RCPP_INPUT_CONTRACT_H
#define ROBART_RCPP_INPUT_CONTRACT_H
#include <Rcpp.h>
#include <cmath>
#include <climits>
#include "recording_contract.h"

// Run before Rcpp converts raw SEXPs to unsigned counts. Keep this call in the
// cgbart registration boundary until generated and manual bindings are separated.
namespace robart_contract {
inline void sexp_count(SEXP x, const char* name, bool positive = true) {
  if ((TYPEOF(x) != INTSXP && TYPEOF(x) != REALSXP) || Rf_xlength(x) != 1)
    Rcpp::stop("%s must be one numeric integer.", name);
  const double value = Rcpp::as<double>(x);
  if (!R_FINITE(value) || value != std::floor(value) ||
      value < (positive ? 1.0 : 0.0) || value > static_cast<double>(INT_MAX))
    Rcpp::stop("%s is outside the supported integer range.", name);
}
inline void sexp_flag(SEXP x, const char* name) {
  const int tx = TYPEOF(x);
  if ((tx != LGLSXP && tx != INTSXP && tx != REALSXP) || Rf_xlength(x) != 1)
    Rcpp::stop("%s must be one TRUE/FALSE or 0/1 value.", name);
  if (tx == LGLSXP && LOGICAL(x)[0] == NA_LOGICAL)
    Rcpp::stop("%s must not be NA.", name);
  const double value = tx == LGLSXP ? static_cast<double>(LOGICAL(x)[0]) : Rcpp::as<double>(x);
  if (!R_FINITE(value) || (value != 0.0 && value != 1.0))
    Rcpp::stop("%s must be TRUE/FALSE or 0/1.", name);
}
inline Rcpp::IntegerVector sexp_snapshot_iterations(SEXP x, std::size_t total) {
  if (x == R_NilValue) return Rcpp::IntegerVector(0);
  if ((TYPEOF(x) != INTSXP && TYPEOF(x) != REALSXP) ||
      Rf_getAttrib(x, R_DimSymbol) != R_NilValue || Rf_isFactor(x))
    Rcpp::stop("save_state_iters must be a numeric vector or NULL.");
  const R_xlen_t len = Rf_xlength(x);
  if (len > INT_MAX) Rcpp::stop("Too many requested snapshot iterations.");
  Rcpp::IntegerVector out(len);
  for (R_xlen_t k = 0; k < len; ++k) {
    const double value = TYPEOF(x) == REALSXP ? REAL(x)[k] :
      (INTEGER(x)[k] == NA_INTEGER ? NA_REAL : static_cast<double>(INTEGER(x)[k]));
    out[k] = snapshot_iteration(value, total);
  }
  return out;
}
} // namespace robart_contract
#endif
