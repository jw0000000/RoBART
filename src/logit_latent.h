// Partially collapsed logit update. Author: Jeongung Heo
#ifndef ROBART_LOGIT_LATENT_H
#define ROBART_LOGIT_LATENT_H
#include "lambda.h"
#include "rtnorm.h"
#include <cmath>
#include <stdexcept>
namespace robart {
// Scale kernel leaves p(v|y,f,c) proportional to prior(v)*Phi(sign*(f+c)/sqrt(v))
// invariant with Z integrated out. Restore Z AFTER the scale MH step, including
// rejection. kmax=1000 retains the inherited 1001-term variance-prior approximation.
inline void update_logit_latent(double f,double offset,double sign,
    double&sd,double&z,rn&gen,int kmax=1000){
  if(!std::isfinite(f)||!std::isfinite(offset)||(sign!=1.&&sign!=-1.)||
     !std::isfinite(sd)||sd<=0||kmax<0)
    throw std::invalid_argument("Invalid logit latent input.");
  const double eta=sign*(f+offset);
  if(!std::isfinite(eta)||!std::isfinite(sd*sd))throw std::invalid_argument("Nonfinite logit latent state.");
  const double v=draw_lambda_i(sd*sd,eta,kmax,1,gen);
  if(!std::isfinite(v)||v<=0)throw std::runtime_error("Nonpositive logit latent variance.");
  const double new_sd=std::sqrt(v);
  const double new_z=sign*rtnorm(sign*f,-sign*offset,new_sd,gen);
  if(!std::isfinite(new_z))throw std::runtime_error("Nonfinite logit latent response.");
  sd=new_sd;z=new_z;
}
}
#endif
