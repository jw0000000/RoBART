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

#ifndef GUARD_heterbart_h
#define GUARD_heterbart_h
#include "bart.h"
#include "heterbartfuns.h"
#include "heterbd.h"

// Build/link ABI guard.  bart is a non-PIMPL base class, so adding data
// members to bart changes the offsets used by heterbart::draw().  R package
// builds do not always track transitive header dependencies for existing .o
// files.  cgbart.cpp calls the two out-of-line functions below so that a stale
// heterbart.o fails at link time (or is rejected by the size check) instead of
// silently causing memory corruption at runtime.
static constexpr int ROBART_HETERBART_ABI_VERSION = 2026101001;
int robart_heterbart_compiled_abi_version() noexcept;
std::size_t robart_heterbart_compiled_sizeof() noexcept;

class heterbart : public bart
{
public:
   heterbart():bart() {}
   heterbart(size_t m):bart(m) {}
   void pr();
   void draw(double *sigma, rn& gen);
};

#endif
