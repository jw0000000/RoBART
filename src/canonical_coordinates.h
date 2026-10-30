// Quantile evaluation contract. Author: Jeongung Heo
#ifndef ROBART_CANONICAL_COORDINATES_H
#define ROBART_CANONICAL_COORDINATES_H
#include <RcppEigen.h>
#include <vector>
#include <stdexcept>
#include <cmath>
namespace robart {
// Separately rounded multiply/add in increasing input-column order. A BLAS
// reduction or a fused multiply-add is not interchangeable at tree boundaries.
// This changes coordinate evaluation, NOT the strict x < cut split convention.
inline double canonical_dot(const Eigen::MatrixXd& Q, size_t row,
                            const double* x, size_t p) {
  volatile double sum=0.0;
  for(size_t k=0;k<p;++k){
    volatile double product=Q(static_cast<int>(row),static_cast<int>(k))*x[k];
    sum=sum+product;
  }
  return sum;
}
inline void canonical_project(const Eigen::MatrixXd& Q,const double* raw,
                              size_t p,size_t n,double* out,
                              const std::vector<char>* touched=nullptr){
  if(p==0 || Q.rows()!=static_cast<int>(p) || Q.cols()!=static_cast<int>(p) ||
     (n && (!raw || !out)) || (touched && touched->size()!=p))
    throw std::invalid_argument("Invalid canonical coordinate input.");
  if(n && raw==out)throw std::invalid_argument("Do not overwrite canonical raw input.");
  for(size_t i=0;i<n;++i)for(size_t a=0;a<p;++a){
    if(touched && !(*touched)[a])continue;
    out[a+p*i]=canonical_dot(Q,a,raw+p*i,p);
    if(!std::isfinite(out[a+p*i]))throw std::runtime_error("Nonfinite canonical coordinate.");
  }
}
}
#endif
