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

#include <ctime>

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "rtnorm.h"
//#include "rtruncnorm.h"

#ifndef NoRcpp

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)

RcppExport SEXP cpbart(
   SEXP _in,            //number of observations in training data
   SEXP _ip,		//dimension of x
   SEXP _inp,		//number of observations in test data
   SEXP _ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
   SEXP _iy,		//y, train,  nx1
   SEXP _ixp,		//x, test, pxnp (transposed so rows are contiguous in memory)
   SEXP _im,		//number of trees
   SEXP _inc,		//number of cut points
   SEXP _ind,		//number of kept draws (except for thinnning ..)
   SEXP _iburn,		//number of burn-in draws skipped
   SEXP _ipower,
   SEXP _ibase,
   SEXP _binaryOffset,
   SEXP _itau,
//   SEXP _iM, // number of shards for Modified LISA MCMC
   SEXP _idart,
   SEXP _itheta,
   SEXP _iomega,
   SEXP _igrp,
   SEXP _ia,
   SEXP _ib,
   SEXP _irho,
   SEXP _iaug,
   SEXP _inkeeptrain,
   SEXP _inkeeptest,
//   SEXP _inkeeptestme,
   SEXP _inkeeptreedraws,
   SEXP _inprintevery,
//   SEXP _treesaslists,
   SEXP _Xinfo
)
{

   //--------------------------------------------------
   //process args
   size_t n = Rcpp::as<int>(_in);
   size_t p = Rcpp::as<int>(_ip);
   size_t np = Rcpp::as<int>(_inp);
   Rcpp::NumericVector  xv(_ix);
   double *ix = &xv[0];
   Rcpp::IntegerVector  yv(_iy); // binary
   int *iy = &yv[0];
   Rcpp::NumericVector  xpv(_ixp);
   double *ixp = &xpv[0];
   size_t m = Rcpp::as<int>(_im);
   Rcpp::IntegerVector _nc(_inc);
   int *numcut = &_nc[0];
   size_t nd = Rcpp::as<int>(_ind);
   size_t burn = Rcpp::as<int>(_iburn);
   double mybeta = Rcpp::as<double>(_ipower);
   double alpha = Rcpp::as<double>(_ibase);
   double binaryOffset = Rcpp::as<double>(_binaryOffset);
   double tau = Rcpp::as<double>(_itau);
   bool dart;
   if(Rcpp::as<int>(_idart)==1) dart=true;
   else dart=false;
   double a = Rcpp::as<double>(_ia);
   double b = Rcpp::as<double>(_ib);
   double rho = Rcpp::as<double>(_irho);
   bool aug;
   if(Rcpp::as<int>(_iaug)==1) aug=true;
   else aug=false;
   double theta = Rcpp::as<double>(_itheta);
   double omega = Rcpp::as<double>(_iomega);
   Rcpp::IntegerVector _grp(_igrp);
   int *grp = &_grp[0];
   size_t nkeeptrain = Rcpp::as<int>(_inkeeptrain);
   size_t nkeeptest = Rcpp::as<int>(_inkeeptest);
   size_t nkeeptreedraws = Rcpp::as<int>(_inkeeptreedraws);
   size_t printevery = Rcpp::as<int>(_inprintevery);
   Rcpp::NumericMatrix Xinfo(_Xinfo);


   Rcpp::NumericMatrix trdraw(nkeeptrain,n);
   Rcpp::NumericMatrix tedraw(nkeeptest,np);
   Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
   Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);

   //random number generation
   arn gen;

   bart bm(m);

   if(Xinfo.size()>0) {
     xinfo _xi;
     _xi.resize(p);
     for(size_t i=0;i<p;i++) {
       _xi[i].resize(numcut[i]);
       for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
     }
     bm.setxinfo(_xi);
   }
#else

#define TRDRAW(a, b) trdraw[a][b]
#define TEDRAW(a, b) tedraw[a][b]

void cpbart(
   size_t n,            //number of observations in training data
   size_t p,		//dimension of x
   size_t np,		//number of observations in test data
   double* ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
   int* iy,		//y, train,  nx1
   double* ixp,		//x, test, pxnp (transposed so rows are contiguous in memory)
   size_t m,		//number of trees
   int *numcut, //size_t nc,		//number of cut points
   size_t nd,		//number of kept draws (except for thinnning ..)
   size_t burn,		//number of burn-in draws skipped
   double mybeta,
   double alpha,
   double binaryOffset,
   double tau,
   bool dart,
   double theta,
   double omega,
   int *grp,
   double a,
   double b,
   double rho,
   bool aug,
   size_t nkeeptrain,
   size_t nkeeptest,
   size_t nkeeptreedraws,
   size_t printevery,
   unsigned int n1, // additional parameters needed to call from C++
   unsigned int n2,

   double* _trdraw,
   double* _tedraw
)
{

   std::vector<double*> trdraw(nkeeptrain);
   std::vector<double*> tedraw(nkeeptest);

   for(size_t i=0; i<nkeeptrain; ++i) trdraw[i]=&_trdraw[i*n];
   for(size_t i=0; i<nkeeptest; ++i) tedraw[i]=&_tedraw[i*np];

   std::vector< std::vector<size_t> > varcnt;
   std::vector< std::vector<double> > varprb;

   //random number generation
   arn gen(n1, n2);

   bart bm(m);
#endif



   double* iz = new double[n];

   std::stringstream treess;  //string stream to write trees to
   treess.precision(10);
   treess << nkeeptreedraws << " " << m << " " << p << endl;
   // dart iterations
   std::vector<double> ivarprb (p,0.);
   std::vector<size_t> ivarcnt (p,0);

   printf("*****Into main of pbart\n");

   size_t skiptr,skipte,skiptreedraws;
   if(nkeeptrain) {skiptr=nd/nkeeptrain;}
   else skiptr = nd+1;
   if(nkeeptest) {skipte=nd/nkeeptest;}
   else skipte=nd+1;

   if(nkeeptreedraws) {skiptreedraws = nd/nkeeptreedraws;}
   else skiptreedraws=nd+1;

   //--------------------------------------------------
   //print args
   printf("*****Data:\n");
   printf("data:n,p,np: %zu, %zu, %zu\n",n,p,np);
   printf("y1,yn: %d, %d\n",iy[0],iy[n-1]);
   printf("x1,x[n*p]: %lf, %lf\n",ix[0],ix[n*p-1]);
   if(np) printf("xp1,xp[np*p]: %lf, %lf\n",ixp[0],ixp[np*p-1]);
   printf("*****Number of Trees: %zu\n",m);
   printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
   printf("*****burn and ndpost: %zu, %zu\n",burn,nd);
   printf("*****Prior:mybeta,alpha,tau: %lf,%lf,%lf\n",
                   mybeta,alpha,tau);
   printf("*****binaryOffset: %lf\n",binaryOffset);
   cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: " 
	<< dart << ',' << theta << ',' << omega << ',' << a << ',' 
	<< b << ',' << rho << ',' << aug << endl;
   printf("*****nkeeptrain,nkeeptest,nkeeptreedraws: %zu,%zu,%zu\n",
               nkeeptrain,nkeeptest,nkeeptreedraws);

   printf("*****printevery: %zu\n",printevery);
   printf("*****skiptr,skipte,skiptreedraws: %zu,%zu,%zu\n",skiptr,skipte,skiptreedraws);

   //--------------------------------------------------
   //bart bm(m);
   bm.setprior(alpha,mybeta,tau);
   bm.setdata(p,n,ix,iz,numcut);
   bm.setdart(a,b,rho,aug,dart,theta,omega);
   //--------------------------------------------------
//init
  for(size_t k=0; k<n; k++) {
    if(iy[k]==0) iz[k]= -rtnorm(0., binaryOffset, 1., gen);
    else iz[k]=rtnorm(0., -binaryOffset, 1., gen);

  }

   //--------------------------------------------------

   //--------------------------------------------------
   //temporary storage
   //out of sample fit
   double* fhattest=0; 
   if(np) { fhattest = new double[np]; }

   //--------------------------------------------------
   //mcmc
   printf("\nMCMC\n");
   size_t trcnt=0; //count kept train draws
   size_t tecnt=0; //count kept test draws
   bool keeptest,/*keeptestme,*/keeptreedraw;

   time_t tp;
   int time1 = time(&tp);
   xinfo& xi = bm.getxinfo();

   for(size_t i=0;i<(nd+burn);i++) {
      if(i%printevery==0) printf("done %zu (out of %lu)\n",i,nd+burn);
      if(i==(burn/2)&&dart) bm.startdart();
      //draw bart
      bm.draw(1., gen);
      //bm.draw(rootM, gen);

  for(size_t k=0; k<n; k++) {
    if(iy[k]==0) iz[k]= -rtnorm(-bm.f(k), binaryOffset, 1., gen);
    else iz[k]=rtnorm(bm.f(k), -binaryOffset, 1., gen);

  }

      if(i>=burn) {

         if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
            for(size_t k=0;k<n;k++) TRDRAW(trcnt,k)=bm.f(k);
            trcnt+=1;
         }
         keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
         if(keeptest) {
	   bm.predict(p,np,ixp,fhattest);
            for(size_t k=0;k<np;k++) TEDRAW(tecnt,k)=fhattest[k];
            tecnt+=1;
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

         }
      }
   }
   int time2 = time(&tp);
   printf("time: %ds\n",time2-time1);
/*
   for(size_t k=0;k<n;k++) trmean[k]/=nd;
   for(size_t k=0;k<np;k++) temean[k]/=temecnt;
*/
   printf("check counts\n");
   printf("trcnt,tecnt: %zu,%zu\n",trcnt,tecnt);
   //--------------------------------------------------
   //PutRNGstate();

   if(fhattest) delete[] fhattest;
   delete[] iz;

#ifndef NoRcpp
   //--------------------------------------------------
   Rcpp::List ret;
   ret["yhat.train"]=trdraw;
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
