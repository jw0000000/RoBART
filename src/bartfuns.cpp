/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include "bartfuns.h"
#include "input_contract.h"

//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t numcut)
{
   int* nc = new int[p];
   for(size_t i=0; i<p; ++i) nc[i]=numcut;
   makexinfo(p, n, x, xi, nc);
   delete [] nc;
}

void makexinfo(size_t p, size_t n, double *x, xinfo& xi, int *nc)
{

   double xinc;

   //compute min and max for each x
   std::vector<double> minx(p,INFINITY);
   std::vector<double> maxx(p,-INFINITY);
   double xx;
   for(size_t i=0;i<p;i++) {
      for(size_t j=0;j<n;j++) {
         xx = *(x+p*j+i);
         if(xx < minx[i]) minx[i]=xx;
         if(xx > maxx[i]) maxx[i]=xx;
      }
   }
   //make grid of nc cutpoints between min and max for each x.
   xi.resize(p);
   for(size_t i=0;i<p;i++) {
      xinc = (maxx[i]-minx[i])/(nc[i]+1.0);
      xi[i].resize(nc[i]);
      for(size_t j=0;j<nc[i];j++) xi[i][j] = minx[i] + (j+1)*xinc;
   }
}


// Training-only covering radius, with deterministic roundoff padding.
// Orthogonal rotations preserve row norms; no tunable radius multiplier is used.
double global_covering_radius_from_design(const double* x,
                                             size_t p,
                                             size_t n)
{
   if (x == nullptr || p == 0 || n == 0) return 0.0;

   long double max_norm_sq = 0.0L;
   for (size_t i = 0; i < n; ++i) {
      long double norm_sq = 0.0L;
      for (size_t j = 0; j < p; ++j) {
         const long double z = static_cast<long double>(*(x + p*i + j));
         norm_sq += z*z;
      }
      if (norm_sq > max_norm_sq) max_norm_sq = norm_sq;
   }

   // Deterministic numerical padding for accumulated Givens roundoff.
   return static_cast<double>(std::sqrt(max_norm_sq)) * (1.0 + 1e-10);
}

// nc specifies the actual per-coordinate candidate counts after preprocessing.
// Uniform: min + k*(max-min)/(nc+1). Quantile: floor-index order statistics
// at k/(nc+1), without interpolation; ties are retained. No responses are read.
void build_xinfo_from_design(const double* x,
                             size_t p,
                             size_t n,
                             const std::vector<int>& nc,
                             XInfoPolicy policy,
                             xinfo& xi,
                             double* global_radius_out)
{
   xi.clear();
   xi.resize(p);
   if (global_radius_out)
      *global_radius_out = std::numeric_limits<double>::quiet_NaN();
   if (n == 0 || p == 0) return;

   if (policy == XInfoPolicy::GlobalCoveringUniform) {
      const double radius = global_covering_radius_from_design(x,p,n);
      if (global_radius_out) *global_radius_out = radius;
      for (size_t j = 0; j < p; ++j) {
         const size_t ncj = (j < nc.size() && nc[j] > 0) ? (size_t)nc[j] : 0;
         xi[j].assign(ncj,0.0);
         if (ncj == 0) continue;
         const double inc = 2.0*radius/((double)ncj+1.0);
         for (size_t k=0;k<ncj;++k)
            xi[j][k] = -radius + (double)(k+1)*inc;
      }
      return;
   }

   for (size_t j = 0; j < p; ++j) {
      const size_t ncj = (j < nc.size() && nc[j] > 0) ? (size_t)nc[j] : 0;
      xi[j].assign(ncj,0.0);
      if (ncj == 0) continue;

      std::vector<double> vals;
      vals.reserve(n);
      for (size_t i=0;i<n;++i) vals.push_back(*(x+p*i+j));
      if (vals.empty()) continue;

      if (policy == XInfoPolicy::QuantileFixedNC) {
         std::sort(vals.begin(),vals.end());
         const size_t last=vals.size()-1;
         for (size_t k=0;k<ncj;++k) {
            const double q=((double)(k+1))/((double)(ncj+1));
            size_t idx=(size_t)std::floor(q*(double)last);
            if (idx>last) idx=last;
            xi[j][k]=vals[idx];
         }
      } else {
         auto mm=std::minmax_element(vals.begin(),vals.end());
         const double minv=*mm.first, maxv=*mm.second;
         const double inc=(maxv-minv)/((double)ncj+1.0);
         for (size_t k=0;k<ncj;++k) xi[j][k]=minv+(double)(k+1)*inc;
      }
   }
}
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
static inline bool cansplit_available(tree::tree_p n, xinfo& xi)
{
   if (!n) return false;
   return cansplit(n, xi);
}

// Depth prior split_prob_depth() is declared inline in bartfuns.h.

double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots)
{
   double pb;  //prob of birth to be returned
   tree::npv bnv; //all the bottom nodes
   t.getbots(bnv);
   for(size_t i=0;i!=bnv.size();i++)
      if(cansplit_available(bnv[i],xi)) goodbots.push_back(bnv[i]);
   if(goodbots.size()==0) { //are there any bottom nodes you can split on?
      pb=0.0;
   } else {
      if(t.treesize()==1) pb=1.0; //is there just one node?
      else pb=pi.pb;
   }
   return pb;
}
//--------------------------------------------------
//compute n and \sum y_i for left and right give bot and v,c
void getsuff(tree& x, tree::tree_p nx, size_t v, size_t c, xinfo& xi, dinfo* di_ptr, size_t& nl, double& syl, size_t& nr, double& syr)
{
   double *xx;//current x
   nl=0; syl=0.0;
   nr=0; syr=0.0;

   size_t n = (di_ptr -> n);
   for(size_t i=0;i<n;i++) {
      xx = (di_ptr -> x) + i*(di_ptr -> p);
      if(nx==x.bn(xx,xi)) { //does the bottom node = xx's bottom node
         if(xx[v] < xi[v][c]) {
               nl++;
               syl += ((di_ptr -> y)[i]);
          } else {
               nr++;
               syr += ((di_ptr -> y)[i]);
          }
      }
   }

}
//lh, replacement for lil that only depends on sum y.
double lh(size_t n, double sy, double sigma, double tau)
{
   double s2 = sigma*sigma;
   double t2 = tau*tau;
   double k = n*t2+s2;
   return -.5*log(k) + ((t2*sy*sy)/(2.0*s2*k));
}
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else depth-prior probability
double pgrow(tree::tree_p n, xinfo& xi, pinfo& pi, const DepthPriorSpec& dp)
{
   if(!cansplit_available(n, xi)) return 0.0;
   return split_prob_depth(n->depth(), pi, dp);
}
//--------------------------------------------------
//compute n and \sum y_i for left and right bots
void getsuff(tree& x, tree::tree_p l, tree::tree_p r, xinfo& xi, dinfo* di_ptr, size_t& nl, double& syl, size_t& nr, double& syr)
{
   double *xx;//current x
   nl=0; syl=0.0;
   nr=0; syr=0.0;
   
   size_t n = (di_ptr -> n);
   for(size_t i=0;i<n;i++) {
      xx = (di_ptr -> x) + i*(di_ptr -> p);
      tree::tree_cp bn = x.bn(xx,xi);
      if(bn==l) {
         nl++;
         syl += ((di_ptr->y)[i]);
      }
      if(bn==r) {
         nr++;
         syr += ((di_ptr->y)[i]);
      }
   }
}
//--------------------------------------------------
//get sufficients stats for all bottom nodes, this way just loop through all the data once.
void allsuff(tree& x, xinfo& xi, dinfo* di_ptr, tree::npv& bnv, std::vector<size_t>& nv, std::vector<double>& syv)
{
   tree::tree_cp tbn; //the pointer to the bottom node for the current observations
   size_t ni;         //the  index into vector of the current bottom node
   double *xx;        //current x

   bnv.clear();
   x.getbots(bnv);

   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   nv.resize(nb);
   syv.resize(nb);

   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz i=0;i!=bnv.size();i++) {bnmap[bnv[i]]=i;nv[i]=0;syv[i]=0.0;}
   
   size_t n = (di_ptr -> n);
   for(size_t i=0;i<n;i++) {
      xx = (di_ptr -> x) + i*(di_ptr -> p);
      tbn = x.bn(xx,xi);
      ni = bnmap[tbn];

      ++(nv[ni]);
      syv[ni] += ((di_ptr->y)[i]);
   }
}
//--------------------------------------------------
// draw all the bottom node mu's
void drmu(tree& t, xinfo& xi, dinfo* di_ptr, pinfo& pi, double sigma, rn& gen)
{
   tree::npv bnv;
   std::vector<size_t> nv;
   std::vector<double> syv;
   allsuff(t,xi,di_ptr,bnv,nv,syv);

   for(tree::npv::size_type i=0;i!=bnv.size();i++) 
      bnv[i]->settheta(drawnodemu(nv[i],syv[i],pi.tau,sigma,gen));
}
//--------------------------------------------------
//draw one mu from post 
double drawnodemu(size_t n, double sy, double tau, double sigma, rn& gen)
{
   double s2 = sigma*sigma;
   double b = n/s2;
   double a = 1.0/(tau*tau);
   return (sy/s2)/(a+b) + gen.normal()/sqrt(a+b);
}

double log_sum_exp(std::vector<double>& v){
    double mx=v[0],sm=0.;
    for(size_t i=0;i<v.size();i++) if(v[i]>mx) mx=v[i];
    for(size_t i=0;i<v.size();i++){
      sm += exp(v[i]-mx);
    }
    return mx+log(sm);
}

//--------------------------------------------------
//draw variable splitting probabilities from Dirichlet (Linero, 2018)
void draw_s(std::vector<size_t>& nv, std::vector<double>& lpv, double& theta, rn& gen){
  size_t p=nv.size();
// Now draw s, the vector of splitting probabilities
  std::vector<double> _theta(p);
  for(size_t j=0;j<p;j++) _theta[j]=theta/(double)p+(double)nv[j];
  //gen.set_alpha(_theta);
  lpv=gen.log_dirichlet(_theta);
}

//--------------------------------------------------
//draw Dirichlet sparsity parameter from posterior using grid
void draw_theta0(bool const_theta, double& theta, std::vector<double>& lpv,
		 double a, double b, double rho, rn& gen){
  // Draw sparsity parameter theta_0 (Linero calls it alpha); see Linero, 2018
  // theta / (theta + rho ) ~ Beta(a,b)
  // Set (a=0.5, b=1) for sparsity
  // Set (a=1, b=1) for non-sparsity
  // rho = p usually, but making rho < p increases sparsity
  if(!const_theta){
    size_t p=lpv.size();
    double sumlpv=0.,lse;
    
    std::vector<double> lambda_g (1000,0.);
    std::vector<double> theta_g (1000,0.);
    std::vector<double> lwt_g (1000,0.);
    for(size_t j=0;j<p;j++) sumlpv+=lpv[j];
    for(size_t k=0;k<1000;k++){
      lambda_g[k]=(double)(k+1)/1001.;
      theta_g[k]=(lambda_g[k]*rho)/(1.-lambda_g[k]);
      double theta_log_lik=lgamma(theta_g[k])-(double)p*lgamma(theta_g[k]/(double)p)+(theta_g[k]/(double)p)*sumlpv;
      double beta_log_prior=(a-1.)*log(lambda_g[k])+(b-1.)*log(1.-lambda_g[k]);
      lwt_g[k]=theta_log_lik+beta_log_prior;      
    }
    lse=log_sum_exp(lwt_g);
    for(size_t k=0;k<1000;k++) {
      lwt_g[k]=exp(lwt_g[k]-lse);
    }
    gen.set_wts(lwt_g);    
    theta=theta_g[gen.discrete()];
  } 
}

/**
 * @brief code from the paper <<Random Rotation Ensembles>>
 */
void generateRandomRotationMatrix(MatrixXd &M, int n, rn& gen, bool debug_mode){

   if(debug_mode){
      return;
   }

   // MTRand mtrand;

   Eigen::MatrixXd A(n,n);
   const Eigen::VectorXd ones(Eigen::VectorXd::Ones(n));

   for(int i=0;i<n;++i){
      for(int j=0;j<n;++j){
         // original source code
         A(i,j) = gen.normal();
      }
   }

   const Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
   const Eigen::MatrixXd Q = qr.householderQ();
   M = Q * (qr.matrixQR().diagonal().array() < 0).select(-ones, ones).asDiagonal();

   if(M.determinant()<0){
      for(int i = 0; i<n;++i){
         M(i,0) = -M(i,0);
      }
   }


}
void printPtrToMat(double *x, size_t p, size_t n){


  Eigen::VectorXd newVec(p*n);
  size_t count=0;
  for(size_t i=0;i<p;i++) {
    for(size_t j=0;j<n;j++) {
      double xx = *(x+p*j+i);
      newVec[count] = xx;
      count+=1;
    }
  }
  
  Eigen::MatrixXd A = Eigen::Map<const Eigen::MatrixXd>(newVec.data(), n, p);
  
  cout << "A =" << std::endl << A << std::endl;
  // }else{

}
