// Internal deterministic evaluator. Author: Jeongung Heo
#include "canonical_coordinates.h"
// Registered through the package-owned native API in native_api.cpp.
Rcpp::NumericMatrix cgbart_canonical_coordinates(Rcpp::NumericMatrix x,Rcpp::NumericMatrix Q){
  const int n=x.nrow(),p=x.ncol();
  if(p<1 || Q.nrow()!=p || Q.ncol()!=p)Rcpp::stop("Q must match the columns of x.");
  Eigen::MatrixXd q(p,p);
  for(int b=0;b<p;++b)for(int a=0;a<p;++a){
    if(!R_FINITE(Q(a,b)))Rcpp::stop("Nonfinite Q.");q(a,b)=Q(a,b);}
  std::vector<double>raw(static_cast<size_t>(n)*p),rotated(raw.size());
  for(int i=0;i<n;++i)for(int a=0;a<p;++a){
    if(!R_FINITE(x(i,a)))Rcpp::stop("Nonfinite x.");raw[a+static_cast<size_t>(p)*i]=x(i,a);}
  robart::canonical_project(q,raw.data(),p,n,rotated.data());
  Rcpp::NumericMatrix out(n,p);
  for(int i=0;i<n;++i)for(int a=0;a<p;++a)out(i,a)=rotated[a+static_cast<size_t>(p)*i];
  return out;
}
