/* Exact-state caches for the full recursive prior.
 * Copyright (C) 2026 Jeongung Heo; GPL (>=2).
 * No RNG use. Geometry is owned by bart; tracked rotation commits are the only
 * in-place coordinate writes between resets. Public audit helpers use the
 * uncached constructor. Cached objects never retain a reference to a proposal
 * grid, tree node, residual vector, or conditional SD vector.
 */
#ifndef ROBART_FORMAL_TREE_CACHE_H
#define ROBART_FORMAL_TREE_CACHE_H
#include "formal_tree_prior.h"
#include <cstdint>
#include <cstring>
#include <utility>

namespace robart { namespace formal {
inline bool same_fp(double a,double b) {
  return std::memcmp(&a,&b,sizeof(double))==0;
}
inline bool same_terms(const PriorTerms& a,const PriorTerms& b) {
  return a.valid==b.valid && same_fp(a.internal,b.internal) && same_fp(a.terminal,b.terminal);
}
struct GeometryCache {
  bool valid=false;
  const double* coordinates=nullptr; // identity check only, never dereferenced on a hit
  size_t p=0,n=0;
  uint64_t revision=0;
  std::vector<unsigned char> active;
  xinfo checked_grid;
  void invalidate() { valid=false; }
  void reset(const xinfo& g,const double* X,size_t pp,size_t nn) {
    // Strong exception guarantee: publish only after all validation succeeds.
    const std::vector<unsigned char> a=design_support(X,pp,nn);
    validate_grid(g,pp);
    xinfo copy=g;
    active=a;checked_grid.swap(copy);coordinates=X;p=pp;n=nn;++revision;valid=true;
  }
  void ensure(const xinfo& g,const double* X,size_t pp,size_t nn) {
    if(!valid||coordinates!=X||p!=pp||n!=nn||checked_grid!=g)reset(g,X,pp,nn);
#ifdef ROBART_OPT_SHADOW
    if(active!=design_support(X,pp,nn))throw std::logic_error("Geometry cache shadow mismatch.");
    validate_grid(g,pp);
#endif
  }
  // 'touched' is the exact union of coordinates updated by the Givens path.
  // With an empty mask, all columns are refreshed. Unchanged columns/grid
  // entries are checked against the accepted geometry before reuse.
  GeometryCache proposal(const xinfo& g,const double* X,const std::vector<char>& touched,
                         const std::vector<unsigned char>* fused_active=nullptr) const {
    if(!valid||g.size()!=p||!X)throw std::logic_error("Invalid proposal cache base.");
    GeometryCache out=*this;
    for(size_t v=0;v<p;++v) {
      const bool changed=touched.empty()||(v<touched.size()&&touched[v]);
      if(!changed) {
        if(g[v]!=checked_grid[v])throw std::logic_error("Untouched proposal grid changed.");
        continue;
      }
      if(fused_active) {
        if(fused_active->size()!=p)throw std::logic_error("Fused support size mismatch.");
        out.active[v]=(*fused_active)[v];
      } else {
        double lo=X[v],hi=lo;
        for(size_t i=0;i<n;++i) {
          const double z=X[v+p*i];
          if(!std::isfinite(z))throw std::invalid_argument("Formal tree support received nonfinite coordinates.");
          lo=std::min(lo,z);hi=std::max(hi,z);
        }
        out.active[v]=(lo<hi);
      }
      // Keep the exact original finite/nondecreasing grid contract.
      const auto& col=g[v];
      if(col.size()>static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Grid exceeds supported integer indices.");
      for(size_t k=0;k<col.size();++k)
        if(!std::isfinite(col[k])||(k&&col[k]<col[k-1]))
          throw std::invalid_argument("Formal grid must be finite and nondecreasing.");
      out.checked_grid[v]=col;
    }
    out.coordinates=X;++out.revision;
#ifdef ROBART_OPT_SHADOW
    if(out.active!=design_support(X,p,n))throw std::logic_error("Proposal support shadow mismatch.");
    validate_grid(g,p);
#endif
    return out;
  }
};

struct PriorNodeKey {
  size_t v,c;unsigned char children;
  bool operator==(const PriorNodeKey& o)const {return v==o.v&&c==o.c&&children==o.children;}
};
inline std::vector<PriorNodeKey> prior_structure_key(const tree& tr) {
  tree::cnpv nodes;tr.getnodes(nodes);
  std::vector<PriorNodeKey> key;key.reserve(nodes.size());
  for(const tree* x:nodes) {
    const unsigned char flags=(x->getl()?1:0)|(x->getr()?2:0);
    key.push_back(PriorNodeKey{flags?x->getv():0,flags?x->getc():0,flags});
  }
  return key; // deterministic preorder; leaf means deliberately absent
}
struct TreeCache {
  GeometryCache geometry;
  bool prior_valid=false;
  uint64_t prior_revision=0;
  std::vector<PriorNodeKey> structure;
  std::vector<double> weights;
  double alpha=0.,beta=0.,tilt=0.,nu=0.;int depth_type=-1;
  PriorTerms value;
  void invalidate() {geometry.invalidate();prior_valid=false;}
  Context context(const xinfo& g,const double* X,size_t p,size_t n,const std::vector<double>* logs) {
    geometry.ensure(g,X,p,n);
    return Context(g,geometry.active,logs,ValidatedGeometryTag{});
  }
  bool parameters_match(const Context& ctx,const pinfo& pi,const DepthPriorSpec& dp)const {
    return prior_valid&&prior_revision==geometry.revision&&weights==ctx.logw&&
      same_fp(alpha,pi.alpha)&&same_fp(beta,pi.mybeta)&&depth_type==dp.type&&
      same_fp(tilt,dp.tilt)&&same_fp(nu,dp.nu);
  }
  PriorTerms get(const tree& tr,const Context& ctx,const pinfo& pi,const DepthPriorSpec& dp) {
    std::vector<PriorNodeKey> key=prior_structure_key(tr);
    if(!parameters_match(ctx,pi,dp)||structure!=key) {
      const PriorTerms z=prior(tr,ctx,pi,dp);
      remember_key(key,ctx,pi,dp,z);
    }
#ifdef ROBART_OPT_SHADOW
    if(!same_terms(value,prior(tr,ctx,pi,dp)))throw std::logic_error("Tree prior cache shadow mismatch.");
#endif
    return value;
  }
  void remember_key(const std::vector<PriorNodeKey>& key,const Context& ctx,const pinfo& pi,
                    const DepthPriorSpec& dp,const PriorTerms& z) {
    structure=key;weights=ctx.logw;prior_revision=geometry.revision;
    alpha=pi.alpha;beta=pi.mybeta;depth_type=dp.type;tilt=dp.tilt;nu=dp.nu;
    value=z;prior_valid=true;
  }
  void remember(const tree& tr,const Context& ctx,const pinfo& pi,const DepthPriorSpec& dp,const PriorTerms& z) {
    remember_key(prior_structure_key(tr),ctx,pi,dp,z);
  }
  void commit_geometry(GeometryCache& proposal,const double* accepted_X) {
    // Proposal owns numeric vectors only: safe after xi_cur.swap(xi_prop).
    geometry=std::move(proposal);geometry.coordinates=accepted_X;prior_valid=false;
  }
};
}} // robart::formal
#endif
