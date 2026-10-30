// Package-owned native registration. Keep names/arities stable for saved callers.
// No Rcpp attribute owns this file. Tests check every registered entry and guard.
#include <Rcpp.h>
#include <R_ext/Rdynload.h>

RcppExport SEXP _robart_cgbart(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP _robart_cgbart_profile_snapshot(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP _robart_cwbart(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP _robart_cgbart_canonical_coordinates(SEXP, SEXP);

RcppExport SEXP cabart(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP cdraw_lambda_i(SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP clbart(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP cpbart(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP cpwbart(SEXP, SEXP, SEXP);
RcppExport SEXP crtgamma(SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP crtnorm(SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP mc_cores_openmp(void);

RcppExport SEXP _robart_formal_tree_audit(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
RcppExport SEXP _robart_formal_cut_audit(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_robart_formal_tree_audit", (DL_FUNC) &_robart_formal_tree_audit, 6},
    {"_robart_formal_cut_audit", (DL_FUNC) &_robart_formal_cut_audit, 8},
    {"_robart_cgbart_canonical_coordinates", (DL_FUNC) &_robart_cgbart_canonical_coordinates, 2},
    {"_robart_cgbart", (DL_FUNC) &_robart_cgbart, 60},
    {"_robart_cgbart_profile_snapshot", (DL_FUNC) &_robart_cgbart_profile_snapshot, 11},
    {"_robart_cwbart", (DL_FUNC) &_robart_cwbart, 31},
    {"cabart",          (DL_FUNC) &cabart,          31},
    {"cdraw_lambda_i",  (DL_FUNC) &cdraw_lambda_i,   4},
    {"clbart",          (DL_FUNC) &clbart,          24},
    {"cpbart",          (DL_FUNC) &cpbart,          27},
    {"cpwbart",         (DL_FUNC) &cpwbart,          3},
    {"crtgamma",        (DL_FUNC) &crtgamma,         4},
    {"crtnorm",         (DL_FUNC) &crtnorm,          4},
    {"mc_cores_openmp", (DL_FUNC) &mc_cores_openmp,  0},
    {NULL, NULL, 0}
};

RcppExport void R_init_robart(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
