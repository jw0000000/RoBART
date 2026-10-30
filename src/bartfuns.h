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

#ifndef GUARD_bartfuns_h
#define GUARD_bartfuns_h

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include <algorithm>
#include <cmath>
#include <limits>

#include <RcppEigen.h>
// RcppEigen is declared through DESCRIPTION LinkingTo.
// ------------------------------------------------------------------
// Depth prior options (tree structure prior)
//
// Original BART prior (Chipman, George, McCulloch):
//   P(split at depth d) = alpha / (1 + d)^beta
//
// Extensions implemented here:
//   - Exponentially tilted polynomial prior:
//       P(split at depth d) = alpha / (1 + d)^beta * exp(-tilt * d)
//   - Pure exponential depth prior (used in some contraction-rate proofs):
//       P(split at depth d) = nu^(d+1)
//
// NOTE: these affect only the tree structure prior (and thus the BD MH ratio),
// not the (mu | tree) leaf parameter prior.
// ------------------------------------------------------------------
enum DepthPriorType {
  DEPTH_PRIOR_POLY = 0,
  DEPTH_PRIOR_EXPTILT = 1,
  DEPTH_PRIOR_NU = 2
};

struct DepthPriorSpec {
  int type;
  // for DEPTH_PRIOR_EXPTILT
  double tilt;
  // for DEPTH_PRIOR_NU
  double nu;

  DepthPriorSpec(int type_ = DEPTH_PRIOR_POLY,
                 double tilt_ = 0.0,
                 double nu_ = 0.25)
      : type(type_), tilt(tilt_), nu(nu_) {}
};

// ------------------------------------------------------------------
// Split probability as a function of depth (tree structure prior)
//
// Returns P(split at depth d) under the chosen depth prior, clamped to [0,1].
// No statistical hard cap is imposed; availability is checked by callers.
// ------------------------------------------------------------------
inline double split_prob_depth(size_t d, const pinfo& pi,
                               const DepthPriorSpec& dp)
{

  double pg = 0.0;
  if (dp.type == DEPTH_PRIOR_NU) {
    // Root has depth d=0 => pg = nu^(1).
    const double nu = dp.nu;
    pg = (nu <= 0.0) ? 0.0 : std::pow(nu, (double)(d + 1));
  } else if (dp.type == DEPTH_PRIOR_EXPTILT) {
    const double base = pi.alpha / std::pow(1.0 + (double)d, pi.mybeta);
    const double tilt = (dp.tilt <= 0.0 ? 0.0 : dp.tilt);
    pg = base * std::exp(-tilt * (double)d);
  } else {
    pg = pi.alpha / std::pow(1.0 + (double)d, pi.mybeta);
  }

  if (pg < 0.0) pg = 0.0;
  if (pg > 1.0) pg = 1.0;
  return pg;
}


//--------------------------------------------------
// use Eigen MatrixXd
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix;
using Eigen::HouseholderQR;

//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc);
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, int* nc);


enum class XInfoPolicy {
  UniformFixedNC = 0,
  QuantileFixedNC = 1,
  // Rotation-invariant fixed grid on [-B,B], B=max_i ||x_i||_2.
  GlobalCoveringUniform = 2
};

double global_covering_radius_from_design(const double* x,
                                             size_t p,
                                             size_t n);

void build_xinfo_from_design(const double* x,
                             size_t p,
                             size_t n,
                             const std::vector<int>& nc,
                             XInfoPolicy policy,
                             xinfo& xi,
                             double* global_radius_out = nullptr);
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots);
//--------------------------------------------------
//compute n and \sum y_i for left and right give bot and v,c
void getsuff(tree& x, tree::tree_p nx, size_t v, size_t c, xinfo& xi, dinfo* di_ptr, size_t& nl, double& syl, size_t& nr, double& syr);
//--------------------------------------------------
//lh, replacement for lil that only depends on sum y.
double lh(size_t n, double sy, double sigma, double tau);
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else a/(1+d)^b
double pgrow(tree::tree_p n, xinfo& xi, pinfo& pi, const DepthPriorSpec& dp);

// Backward-compatible wrapper: if called without an explicit DepthPriorSpec,
// default to the original polynomial depth prior.
inline double pgrow(tree::tree_p n, xinfo& xi, pinfo& pi)
{
  return pgrow(n, xi, pi, DepthPriorSpec());
}
//--------------------------------------------------
//compute n and \sum y_i for left and right bots
void getsuff(tree& x, tree::tree_p l, tree::tree_p r, xinfo& xi, dinfo* di_ptr, size_t& nl, double& syl, size_t& nr, double& syr);
//--------------------------------------------------
//get sufficients stats for all bottom nodes, this way just loop through all the data once.
void allsuff(tree& x, xinfo& xi, dinfo* di_ptr, tree::npv& bnv, std::vector<size_t>& nv, std::vector<double>& syv);
//--------------------------------------------------
// draw all the bottom node mu's
void drmu(tree& t, xinfo& xi, dinfo* di_ptr, pinfo& pi, double sigma, rn& gen);
//--------------------------------------------------
//draw one mu from post 
double drawnodemu(size_t n, double sy, double tau, double sigma, rn& gen);
//--------------------------------------------------
//draw variable splitting probabilities from Dirichlet (Linero, 2018)
void draw_s(std::vector<size_t>& nv, std::vector<double>& lpv, double& theta, rn& gen);
//--------------------------------------------------
//draw Dirichlet sparsity parameter from posterior using grid
void draw_theta0(bool const_theta, double& theta, std::vector<double>& lpv,
		 double a, double b, double rho, rn& gen);

// Generate random rotation matrix
void generateRandomRotationMatrix(MatrixXd &M, int n, rn& gen, bool debug_mode=false);

void printPtrToMat(double *x, size_t p, size_t n);

#endif
