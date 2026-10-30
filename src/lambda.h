/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017-2018 Robert McCulloch, Rodney Sparapani
 *                          and Robert Gramacy
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

#ifndef GUARD_lambda
#define GUARD_lambda

#include "common.h"

#ifdef NoRcpp
using ::pnorm;
#else
using R::pnorm;
RcppExport SEXP cdraw_lambda_i(SEXP lambda, SEXP mean, SEXP kmax, SEXP thin);
#endif

/* draw lambda from its (infinite mixture) prior */

double draw_lambda_prior(double *psii, int kmax, rn& gen);

/* Prior-proposal MH for the scale with continuous latent Z integrated out.
 * xbeta must include sign*(function + link-scale offset). Restore Z afterwards.
 * Legacy callers are not changed merely by this documentation. */

double draw_lambda_i(double lambda_old, double xbeta,
                     int kmax, int thin, rn& gen);

#endif
