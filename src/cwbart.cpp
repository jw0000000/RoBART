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

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "heterbart.h"

#include "math.h"
#include <limits>
#include <stdexcept>

#include <Rcpp.h>
// RcppEigen is declared through DESCRIPTION LinkingTo.
#include <RcppEigen.h>

using namespace Rcpp;

#ifndef NoRcpp

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)

#endif
// Registered through the package-owned native API in native_api.cpp.
Rcpp::List cwbart(
   size_t n,            //number of observations in training data
   size_t p,		//dimension of x
   size_t np,		//number of observations in test data
   Rcpp::NumericVector  xv,		//x, train,  pxn (transposed so rows are contiguous in memory)
   Rcpp::NumericVector  yv,		//y, train,  nx1
   Rcpp::NumericVector  xpv,		//x, test, pxnp (transposed so rows are contiguous in memory)
   size_t m,		//number of trees
   Rcpp::IntegerVector nc_vec,		//number of cut points
   size_t nd,		//number of kept draws (except for thinnning ..)
   size_t burn,		//number of burn-in draws skipped
   double mybeta,
   double alpha,
   double tau,
   double nu,
   double lambda,
   double sigma,
   Rcpp::NumericVector  wv,
   size_t dart_i,
   double theta,
   double omega,
   Rcpp::IntegerVector grp_vec,
   double a,
   double b,
   double rho,
   size_t aug_i,
   size_t nkeeptrain,
   size_t nkeeptest,
   size_t nkeeptestme,
   size_t nkeeptreedraws,
   size_t printevery,
   Rcpp::NumericMatrix Xinfo
)
{

   printf("*****Start process!!!!!*******\n");

   //--------------------------------------------------
   //process args
   double *ix = &xv[0];
   double *iy = &yv[0];
   double *ixp = &xpv[0];
   int *numcut = &nc_vec[0];
   double *iw = &wv[0];
   bool dart;
   if(dart_i==1) dart=true;
   else dart=false;
   bool aug;
   if(aug_i==1) aug=true;
   else aug=false;
   int *grp = &grp_vec[0];
   

   Rcpp::NumericVector trmean(n); //train
   Rcpp::NumericVector temean(np);
   Rcpp::NumericVector sdraw(nd+burn);
   Rcpp::NumericMatrix trdraw(nkeeptrain,n);
   Rcpp::NumericMatrix tedraw(nkeeptest,np);
   Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
   Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);

   //random number generation
   arn gen;

   heterbart bm(m);

   if(Xinfo.size()>0) {
      cout << "Xinfo.size() > 0" << std::endl;
      xinfo _xi;
      _xi.resize(p);
      for(size_t i=0;i<p;i++) {
         _xi[i].resize(numcut[i]);
         for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
      }
      bm.setxinfo(_xi);
      cout << "eof Xinfo.size() > 0" <<std::endl << std::endl;
   }


   for(size_t i=0;i<n;i++) trmean[i]=0.0;
   for(size_t i=0;i<np;i++) temean[i]=0.0;

   printf("*****Into main of wbart\n");
   //-----------------------------------------------------------

   size_t skiptr,skipte,skipteme,skiptreedraws;
   if(nkeeptrain) {skiptr=nd/nkeeptrain;}
   else skiptr = nd+1;
   if(nkeeptest) {skipte=nd/nkeeptest;}
   else skipte=nd+1;
   if(nkeeptestme) {skipteme=nd/nkeeptestme;}
   else skipteme=nd+1;
   if(nkeeptreedraws) {skiptreedraws = nd/nkeeptreedraws;}
   else skiptreedraws=nd+1;

   //--------------------------------------------------
   //print args
   printf("*****Data:\n");
   printf("data:n,p,np: %zu, %zu, %zu\n",n,p,np);
   printf("y1,yn: %lf, %lf\n",iy[0],iy[n-1]);
   printf("x1,x[n*p]: %lf, %lf\n",ix[0],ix[n*p-1]);
   if(np) printf("xp1,xp[np*p]: %lf, %lf\n",ixp[0],ixp[np*p-1]);
   printf("*****Number of Trees: %zu\n",m);
   printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
   printf("*****burn and ndpost: %zu, %zu\n",burn,nd);
   printf("*****Prior:beta,alpha,tau,nu,lambda: %lf,%lf,%lf,%lf,%lf\n",
                   mybeta,alpha,tau,nu,lambda);
   printf("*****sigma: %lf\n",sigma);
   printf("*****w (weights): %lf ... %lf\n",iw[0],iw[n-1]);
   cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: "
	<< dart << ',' << theta << ',' << omega << ',' << a << ','
	<< b << ',' << rho << ',' << aug << endl;
   printf("*****nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws: %zu,%zu,%zu,%zu\n",
               nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws);
   printf("*****printevery: %zu\n",printevery);
   printf("*****skiptr,skipte,skipteme,skiptreedraws: %zu,%zu,%zu,%zu\n",skiptr,skipte,skipteme,skiptreedraws);

   //--------------------------------------------------
   //heterbart bm(m);
   bm.setprior(alpha,mybeta,tau);

   // Preserve the inherited input preparation for this separate BART entry.
   bm.setdata(p,n,ix,iy,numcut);
   bm.setdart(a,b,rho,aug,dart,theta,omega);

   // This legacy entry serializes only trees and one common cutpoint grid.
   // Its live forest must therefore remain axis-aligned on that exact grid.
   // setdata() has already initialized xi when no explicit Xinfo was supplied.
   bm.set_rotation_mh_start_iter(std::numeric_limits<size_t>::max());
   bm.set_dynamic_xinfo_on_rotation(false);
   bm.set_rotation_xinfo_policy(XInfoPolicy::UniformFixedNC);
   bm.set_user_xinfo_supplied(true);

   //--------------------------------------------------
   // Initialize identity frames without replacing the common cutpoint values.
   bm.applyrandrotation(ix, ixp, p, n, np, numcut, gen, false);
   for (size_t j = 0; j < m; ++j) {
      if (bm.get_tree_xinfo(j) != bm.getxinfo())
         throw std::runtime_error("cwbart: live and serialized cutpoint grids differ");
      Eigen::MatrixXd frame;
      bm.get_tree_rotation_matrix(j, frame);
      if (frame.rows() != static_cast<Eigen::Index>(p) ||
          frame.cols() != static_cast<Eigen::Index>(p) ||
          !(frame.array() == Eigen::MatrixXd::Identity(p, p).array()).all())
         throw std::runtime_error("cwbart: static prediction requires identity frames");
   }

   bm.setallfit(p,n,ix); //Initialize the ensemble fit from its per-tree grids and designs.

   bool is_debug_randrot = false;
   if(is_debug_randrot){
     bm.israndrotationvalid(ix);
   }


   //--------------------------------------------------
   //sigma
   //gen.set_df(n+nu);
   double *svec = new double[n];
   for(size_t i=0;i<n;i++) svec[i]=iw[i]*sigma;

   //--------------------------------------------------

   std::stringstream treess;  //string stream to write trees to
   // Preserve double leaf means in the text used by predict.wbart().
   treess.precision(std::numeric_limits<double>::max_digits10);
   treess << nkeeptreedraws << " " << m << " " << p << endl;
   // dart iterations
   std::vector<double> ivarprb (p,0.);
   std::vector<size_t> ivarcnt (p,0);

   //--------------------------------------------------
   //temporary storage
   //out of sample fit
   double* fhattest=0; //posterior mean for prediction
   if(np) { fhattest = new double[np]; }
   double restemp=0.0,rss=0.0;

   //--------------------------------------------------
   //mcmc
   printf("\nMCMC\n");
   size_t trcnt=0; //count kept train draws
   size_t tecnt=0; //count kept test draws
   size_t temecnt=0; //count test draws into posterior mean
   size_t treedrawscnt=0; //count kept bart draws
   bool keeptest,keeptestme,keeptreedraw;

   time_t tp;
   int time1 = time(&tp);
   xinfo& xi = bm.getxinfo();

   bool is_na_bm_f_found_bf_burn_in = false;
   bool is_na_bm_f_found_after_burn_in = false;
   bool is_na_bm_f_found2 = false;
   for(size_t i=0;i<(nd+burn);i++) {
      if(i%printevery==0) printf("done %zu (out of %lu)\n",i,nd+burn);
      if(i==(burn/2)&&dart) bm.startdart();
      //draw bart
      bm.draw(svec,gen);
      //draw sigma
      rss=0.0;
      for(size_t k=0;k<n;k++) {
        restemp=(iy[k]-bm.f(k))/(iw[k]);
        rss += restemp*restemp;
        if(std::isnan(bm.f(k)) and not is_na_bm_f_found_bf_burn_in){
          cout << "std::isnan(bm.f(k)) is true when i = " << i << " and k = " << k << std::endl;
          is_na_bm_f_found_bf_burn_in = true;
        }
      }
      sigma = sqrt((nu*lambda + rss)/gen.chi_square(n+nu));
      for(size_t k=0;k<n;k++) svec[k]=iw[k]*sigma;
      sdraw[i]=sigma;
      if(i>=burn) {
         for(size_t k=0;k<n;k++) {
            trmean[k]+=bm.f(k);
           if(std::isnan(bm.f(k)) and not is_na_bm_f_found_after_burn_in){
             cout << "std::isnan(bm.f(k)) is true (after the burn-in period) when i = " << i << " and k = " << k << std::endl;
             is_na_bm_f_found2 = true;
             is_na_bm_f_found_after_burn_in = true;
           }
         }
         if(is_na_bm_f_found2){
           is_na_bm_f_found_after_burn_in = true;
         }

         if(i%printevery==0) {
            cout << "bm.f(0)" << bm.f(0) << std::endl;
            cout << "bm.f(1)" << bm.f(1) << std::endl;
            cout << "bm.f(2)" << bm.f(2) << std::endl;
            cout << "bm.f(3)" << bm.f(3) << std::endl;
            cout << "bm.f(4)" << bm.f(4) << std::endl << std::endl;

            cout << "trmean[0]" << trmean[0] << std::endl;
            cout << "trmean[1]" << trmean[1] << std::endl;
            cout << "trmean[2]" << trmean[2] << std::endl;
            cout << "trmean[3]" << trmean[3] << std::endl;
            cout << "trmean[4]" << trmean[4] << std::endl << std::endl;
         }
         if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
            for(size_t k=0;k<n;k++) TRDRAW(trcnt,k)=bm.f(k);
            trcnt+=1;
         }
         keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
         keeptestme = nkeeptestme && (((i-burn+1) % skipteme) ==0) && np;
         if(keeptest || keeptestme) bm.predict(p,np,ixp,fhattest,true);
         if(keeptest) {
            for(size_t k=0;k<np;k++) TEDRAW(tecnt,k)=fhattest[k];
            tecnt+=1;
         }
         if(keeptestme) {
            for(size_t k=0;k<np;k++) temean[k]+=fhattest[k];
            temecnt+=1;
         }
         keeptreedraw = nkeeptreedraws && (((i-burn+1) % skiptreedraws) ==0);
         if(keeptreedraw) {
//	   #ifndef NoRcpp
//	   #endif

            for(size_t j=0;j<m;j++) {
	      treess << bm.gettree(j);

	    }
            #ifndef NoRcpp
	    ivarcnt=bm.getnv();
	    ivarprb=bm.getpv();
	    size_t k=(i-burn)/skiptreedraws;
	    for(size_t j=0;j<p;j++){
	      varcnt(k,j)=ivarcnt[j];
	      varprb(k,j)=ivarprb[j];
	    }
            #else
	    varcnt.push_back(bm.getnv());
	    varprb.push_back(bm.getpv());
	    #endif

            treedrawscnt +=1;
         }
      }
   }
   int time2 = time(&tp);
   printf("time: %ds\n",time2-time1);
   for(size_t k=0;k<n;k++) trmean[k]/=nd;
   for(size_t k=0;k<np;k++) temean[k]/=temecnt;
   printf("check counts\n");
   printf("trcnt,tecnt,temecnt,treedrawscnt: %zu,%zu,%zu,%zu\n",trcnt,tecnt,temecnt,treedrawscnt);
   //--------------------------------------------------
   //PutRNGstate();

   if(fhattest) delete[] fhattest;
   if(svec) delete [] svec;

   //--------------------------------------------------
#ifndef NoRcpp
   Rcpp::List ret;
   ret["sigma"]=sdraw;
   ret["yhat.train.mean"]=trmean;
   ret["yhat.train"]=trdraw;
   ret["yhat.test.mean"]=temean;
   ret["yhat.test"]=tedraw;
   ret["varcount"]=varcnt;
   ret["varprob"]=varprb;

    //  bm.gettree(i).pr();

   Rcpp::List xiret(xi.size());
   for(size_t i=0;i<xi.size();i++) {
      Rcpp::NumericVector vtemp(xi[i].size());
      std::copy(xi[i].begin(),xi[i].end(),vtemp.begin());
      xiret[i] = Rcpp::NumericVector(vtemp);
   }

   Rcpp::List treesL;
   //treesL["nkeeptreedraws"] = Rcpp::wrap<int>(nkeeptreedraws); //in trees
   //treesL["ntree"] = Rcpp::wrap<int>(m); //in trees
   //treesL["numx"] = Rcpp::wrap<int>(p); //in cutpoints
   treesL["cutpoints"] = xiret;
   treesL["trees"]=Rcpp::CharacterVector(treess.str());
   ret["treedraws"] = treesL;

   return ret;
#else

#endif

}
