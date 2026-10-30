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

#include "bd.h"
#include "formal_birth_death.h"

bool bd(tree& x, xinfo& xi, dinfo* di_ptr, pinfo& pi, double sigma,
             std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen,
             const DepthPriorSpec& dp, const std::vector<double>* log_weights,
             robart::formal::TreeCache* state_cache)
{
   std::vector<double> sd(di_ptr->n,sigma);
   return robart::formal::birth_death(x,xi,di_ptr,pi,sd.data(),nv,pv,aug,gen,dp,log_weights,nullptr,state_cache);
}
