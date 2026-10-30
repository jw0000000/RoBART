/*
 * Full recursive discrete-tree prior and its shared support.
 * Copyright (C) 2026 Jeongung Heo
 * GPL version 2 or, at your option, any later version.
 *
 * Uniform-coordinate case implements paper_body.tex (discrete-prior).
 * Nonuniform log weights define an explicitly documented DART extension:
 * internal rho_d*s_v/a, terminal 1-rho_d*sum_{v in G}s_v.
 * No observation-occupancy constraint is imposed. Strict value bounds, not
 * index arithmetic, handle repeated grid values and degenerate coordinates.
 */
#ifndef ROBART_FORMAL_TREE_PRIOR_H
#define ROBART_FORMAL_TREE_PRIOR_H
#include "bartfuns.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace robart { namespace formal {
inline double neginf() { return -std::numeric_limits<double>::infinity(); }
struct Interval {
  int lo, hi;
  Interval(int l=0,int u=-1):lo(l),hi(u){}
  bool empty() const { return hi<lo; }
  size_t size() const { return empty()?0:static_cast<size_t>(hi-lo)+1; }
  bool contains(size_t c) const { return !empty() && c>=static_cast<size_t>(lo) && c<=static_cast<size_t>(hi); }
};

inline std::vector<unsigned char> design_support(const double* X,size_t p,size_t n) {
  if(!p || !n || !X) throw std::invalid_argument("Formal tree support requires a nonempty training design.");
  std::vector<unsigned char> active(p,0);
  for(size_t j=0;j<p;++j) {
    double lo=X[j],hi=lo;
    for(size_t i=0;i<n;++i) {
      double z=X[j+p*i];
      if(!std::isfinite(z)) throw std::invalid_argument("Formal tree support received nonfinite coordinates.");
      lo=std::min(lo,z); hi=std::max(hi,z);
    }
    active[j]=(lo<hi);
  }
  return active;
}

inline std::vector<double> normalized_log_weights(size_t p,const std::vector<double>* logs=nullptr) {
  if(!p) throw std::invalid_argument("At least one coordinate is required.");
  if(!logs) return std::vector<double>(p,-std::log(static_cast<double>(p)));
  if(logs->size()!=p) throw std::invalid_argument("Coordinate log-probability length mismatch.");
  double mx=neginf();
  for(double z:*logs) {
    if(std::isnan(z) || z==std::numeric_limits<double>::infinity())
      throw std::invalid_argument("Invalid coordinate log probabilities.");
    mx=std::max(mx,z);
  }
  if(!std::isfinite(mx)) throw std::invalid_argument("Coordinate probabilities have zero total mass.");
  double sum=0.;for(double z:*logs)sum+=std::exp(z-mx);
  const double normalizer=mx+std::log(sum);
  std::vector<double> ans=*logs;for(double& z:ans)z-=normalizer;
  return ans;
}
inline std::vector<double> logs_from_weights(const std::vector<double>& w,size_t p) {
  if(w.size()!=p)throw std::invalid_argument("Coordinate probability length mismatch.");
  std::vector<double> logs(p);
  for(size_t j=0;j<p;++j) {
    if(!std::isfinite(w[j])||w[j]<0.)throw std::invalid_argument("Invalid coordinate probability.");
    logs[j]=w[j]>0.?std::log(w[j]):neginf();
  }
  return normalized_log_weights(p,&logs);
}

// Only the owning sampler/cache may use this tag after validating geometry.
struct ValidatedGeometryTag {};
inline void validate_grid(const xinfo& g,size_t p) {
  if(g.size()!=p)throw std::invalid_argument("Formal grid dimension mismatch.");
  for(const auto& col:g) {
    if(col.size()>static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Grid exceeds supported integer indices.");
    for(size_t k=0;k<col.size();++k)
      if(!std::isfinite(col[k])||(k&&col[k]<col[k-1]))
        throw std::invalid_argument("Formal grid must be finite and nondecreasing.");
  }
}

struct Context {
  const xinfo& grid;
  std::vector<unsigned char> active;
  std::vector<double> logw;
  Context(const xinfo& g,const double* X,size_t p,size_t n,const std::vector<double>* logs=nullptr)
    :grid(g),active(design_support(X,p,n)),logw(normalized_log_weights(p,logs)) {
    validate_grid(g,p);
  }
  Context(const xinfo& g,const std::vector<unsigned char>& validated_active,
          const std::vector<double>* logs,ValidatedGeometryTag)
    :grid(g),active(validated_active),logw(normalized_log_weights(g.size(),logs)) {
    if(active.size()!=g.size())throw std::invalid_argument("Cached support dimension mismatch.");
  }
  size_t p() const { return grid.size(); }
  Interval allowed(const tree* node,size_t v) const {
    if(!node||v>=p()||!active[v]||grid[v].empty())return Interval();
    double lo=-std::numeric_limits<double>::infinity(),hi=-lo;
    const tree* child=node;
    for(const tree* par=node->getp();par;child=par,par=par->getp()) {
      if(par->getv()>=p()||par->getc()>=grid[par->getv()].size())return Interval();
      if(par->getv()==v) {
        const double c=grid[v][par->getc()];
        if(par->getl()==child)hi=std::min(hi,c);
        else if(par->getr()==child)lo=std::max(lo,c);
        else return Interval();
      }
    }
    if(!(lo<hi))return Interval();
    const auto& col=grid[v];
    const auto a=std::upper_bound(col.begin(),col.end(),lo);
    const auto b=std::lower_bound(col.begin(),col.end(),hi);
    if(a>=b)return Interval();
    return Interval(static_cast<int>(a-col.begin()),static_cast<int>(b-col.begin())-1);
  }
  std::vector<size_t> good(const tree* node) const {
    std::vector<size_t> out;
    for(size_t j=0;j<p();++j)if(!allowed(node,j).empty())out.push_back(j);
    return out;
  }
  double log_mass(const std::vector<size_t>& vars) const {
    if(vars.empty())return neginf();
    if(vars.size()==p())return 0.;
    double mx=neginf();for(size_t j:vars)mx=std::max(mx,logw[j]);
    if(!std::isfinite(mx))return neginf();
    double sum=0.;for(size_t j:vars)sum+=std::exp(logw[j]-mx);
    return std::min(0.,mx+std::log(sum)); // normalization-roundoff only
  }
};

struct PriorTerms {
  double internal=0.,terminal=0.;
  bool valid=true;
  double total() const {return valid?internal+terminal:neginf();}
};
inline PriorTerms prior(const tree& tr,const Context& ctx,const pinfo& pi,const DepthPriorSpec& dp) {
  PriorTerms out;
  struct Item{const tree* node;size_t depth;};
  std::vector<Item> stack(1,Item{&tr,0});
  while(!stack.empty()) {
    const Item it=stack.back();stack.pop_back();
    const tree* node=it.node;
    const double rho=split_prob_depth(it.depth,pi,dp);
    if(!std::isfinite(rho)||rho<0.||rho>1.) {out.valid=false;return out;}
    if(node->getl()) {
      if(!node->getr()) {out.valid=false;return out;}
      const size_t v=node->getv();
      const Interval a=ctx.allowed(node,v);
      if(!a.contains(node->getc())||rho<=0.||!std::isfinite(ctx.logw[v])) {out.valid=false;return out;}
      out.internal+=std::log(rho)+ctx.logw[v]-std::log(static_cast<double>(a.size()));
      stack.push_back(Item{node->getr(),it.depth+1});
      stack.push_back(Item{node->getl(),it.depth+1});
    } else {
      if(node->getr()) {out.valid=false;return out;}
      const double lm=ctx.log_mass(ctx.good(node));
      const double mass=std::isfinite(lm)?std::exp(lm):0.;
      if(rho*mass>=1.) {out.valid=false;return out;}
      out.terminal+=std::log1p(-rho*mass);
    }
  }
  return out;
}

// Complete normalized N(0,tau^2) leaf integral, dropping only factors common
// to every partition. Empty leaves contribute exactly zero.
inline double leaf_score(double b,double M,double tau) {
  const double tau2=tau*tau,k=1.+tau2*b;
  return -.5*std::log1p(tau2*b)+.5*tau2*M*M/k;
}
inline double collapsed_score(tree& tr,xinfo& grid,dinfo* di,const double* sd,double tau) {
  if(!di||!di->x||!di->y||!sd||!(tau>0.)||!std::isfinite(tau))
    throw std::invalid_argument("Invalid collapsed-likelihood inputs.");
  tree::npv leaves;tr.getbots(leaves);
  std::map<tree::tree_p,size_t> index;
  for(size_t j=0;j<leaves.size();++j)index[leaves[j]]=j;
  std::vector<double> b(leaves.size(),0.),M(leaves.size(),0.);
  for(size_t i=0;i<di->n;++i) {
    if(!(sd[i]>0.)||!std::isfinite(sd[i])||!std::isfinite(di->y[i]))
      throw std::invalid_argument("Nonfinite residual or invalid conditional SD.");
    const size_t j=index.at(tr.bn(di->x+di->p*i,grid));
    const double w=1./(sd[i]*sd[i]); b[j]+=w;M[j]+=di->y[i]*w;
  }
  double ans=0.;for(size_t j=0;j<b.size();++j)ans+=leaf_score(b[j],M[j],tau);
  return ans;
}
}} // namespace robart::formal
#endif
