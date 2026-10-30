/*
 * Birth/death MH for the same full collapsed target as rotation.
 * Copyright (C) 2026 Jeongung Heo; GPL (>= 2).
 * No old leaf means enter an acceptance probability; callers redraw leaves
 * after the final retained structural state. No minimum occupancy gate.
 */
#ifndef ROBART_FORMAL_BIRTH_DEATH_H
#define ROBART_FORMAL_BIRTH_DEATH_H
#include "formal_tree_cache.h"
namespace robart { namespace formal {
struct Pools {
  tree::npv births,nogs;
  double pb=0.;
};
inline Pools pools(tree& tr,const Context& ctx,const pinfo& pi) {
  Pools z;tree::npv leaves;tr.getbots(leaves);tr.getnogs(z.nogs);
  for(auto node:leaves)if(std::isfinite(ctx.log_mass(ctx.good(node))))z.births.push_back(node);
  z.pb=z.births.empty()?0.:(z.nogs.empty()?1.:pi.pb);
  if(!std::isfinite(z.pb)||z.pb<0.||z.pb>1.)throw std::invalid_argument("Invalid birth mixture probability.");
  return z;
}
inline double log_birth_q_from_pools(tree& tr,const Context& ctx,const Pools& z,size_t id,size_t v,size_t c,bool available_only) {
  if(z.pb<=0.)return neginf();
  tree* node=tr.getptr(id);
  if(!node||node->getl()||std::find(z.births.begin(),z.births.end(),node)==z.births.end())return neginf();
  const Interval a=ctx.allowed(node,v);
  if(!a.contains(c)||!std::isfinite(ctx.logw[v]))return neginf();
  const double ln=available_only?ctx.log_mass(ctx.good(node)):0.;
  return std::log(z.pb)-std::log(static_cast<double>(z.births.size()))+ctx.logw[v]-ln-std::log(static_cast<double>(a.size()));
}
inline double log_death_q_from_pools(tree& tr,const Pools& z,size_t id) {
  if(z.pb>=1.||z.nogs.empty())return neginf();
  tree* node=tr.getptr(id);
  if(std::find(z.nogs.begin(),z.nogs.end(),node)==z.nogs.end())return neginf();
  return std::log1p(-z.pb)-std::log(static_cast<double>(z.nogs.size()));
}
// Public reference interfaces retain full validation/evaluation behavior.
inline double log_birth_q(tree& tr,const Context& ctx,const pinfo& pi,size_t id,size_t v,size_t c,bool available_only) {
  return log_birth_q_from_pools(tr,ctx,pools(tr,ctx,pi),id,v,c,available_only);
}
inline double log_death_q(tree& tr,const Context& ctx,const pinfo& pi,size_t id) {
  return log_death_q_from_pools(tr,pools(tr,ctx,pi),id);
}
inline size_t sample_index(size_t n,rn& gen) {
  if(!n)throw std::invalid_argument("Cannot sample from an empty set.");
  const double u=gen.uniform();
  if(!(u>=0.&&u<1.))throw std::runtime_error("Uniform RNG must return a value in [0,1).");
  return std::min(n-1,static_cast<size_t>(u*n));
}
inline size_t sample_coordinate(const Context& ctx,const std::vector<size_t>& vars,rn& gen) {
  const double lm=ctx.log_mass(vars);
  if(!std::isfinite(lm))throw std::runtime_error("Available coordinates have zero proposal mass.");
  const double u=gen.uniform();double acc=0.;size_t last=vars.front();
  for(size_t v:vars)if(std::isfinite(ctx.logw[v])) {
    last=v;acc+=std::exp(ctx.logw[v]-lm);if(u<acc)return v;
  }
  return last; // only a floating-point cumulative-sum endpoint
}
struct MoveTerms {
  double likelihood=0.,prior=0.,forward=neginf(),reverse=neginf(),logacc=neginf();
  bool birth=false,valid=false;
  size_t id=0,v=0,c=0;
};
inline MoveTerms evaluate_move(tree& old,tree& proposed,const Context& ctx,pinfo& pi,
    const DepthPriorSpec& dp,dinfo* di,const double* sd,bool available_only,
    bool birth,size_t id,size_t v,size_t c) {
  MoveTerms z;z.birth=birth;z.id=id;z.v=v;z.c=c;
  const double po=prior(old,ctx,pi,dp).total();
  if(!std::isfinite(po))throw std::runtime_error("Current tree is outside the formal prior support.");
  const double pn=prior(proposed,ctx,pi,dp).total();
  if(!std::isfinite(pn))return z;
  z.forward=birth?log_birth_q(old,ctx,pi,id,v,c,available_only):log_death_q(old,ctx,pi,id);
  z.reverse=birth?log_death_q(proposed,ctx,pi,id):log_birth_q(proposed,ctx,pi,id,v,c,available_only);
  if(!std::isfinite(z.forward)||!std::isfinite(z.reverse))return z;
  z.prior=pn-po;
  xinfo& grid=const_cast<xinfo&>(ctx.grid);
  z.likelihood=collapsed_score(proposed,grid,di,sd,pi.tau)-collapsed_score(old,grid,di,sd,pi.tau);
  z.logacc=z.likelihood+z.prior+z.reverse-z.forward;
  z.valid=std::isfinite(z.logacc);
  return z;
}
// One routing pass for a genuine local birth/death. Unchanged leaves share
// sufficient statistics. The merged parent is accumulated in observation order,
// NOT by adding child sums. Finally both full scores are summed in the original
// left-to-right leaf order: this deliberately preserves reference rounding.
// No residual/SD/tau statistics survive the current function call.
inline double paired_collapsed_difference(tree& old,tree& proposed,xinfo& grid,dinfo* di,
        const double* sd,double tau,bool birth,size_t id) {
  if(!di||!di->x||!di->y||!sd||!(tau>0.)||!std::isfinite(tau))
    throw std::invalid_argument("Invalid collapsed-likelihood inputs.");
  tree& fine=birth?proposed:old;
  tree* parent=fine.getptr(id);
  if(!parent||!parent->isnog())throw std::logic_error("Local score needs one nog.");
  tree::npv leaves;fine.getbots(leaves);
  std::map<tree::tree_p,size_t> index;
  for(size_t j=0;j<leaves.size();++j)index[leaves[j]]=j;
  const size_t left=index.at(parent->getl()),right=index.at(parent->getr());
  if(right!=left+1)throw std::logic_error("Nog children are not adjacent leaves.");
  std::vector<double> b(leaves.size(),0.),M(leaves.size(),0.);
  double bp=0.,Mp=0.;
  for(size_t i=0;i<di->n;++i) {
    if(!(sd[i]>0.)||!std::isfinite(sd[i])||!std::isfinite(di->y[i]))
      throw std::invalid_argument("Nonfinite residual or invalid conditional SD.");
    const size_t j=index.at(fine.bn(di->x+di->p*i,grid));
    const double w=1./(sd[i]*sd[i]);b[j]+=w;M[j]+=di->y[i]*w;
    if(j==left||j==right){bp+=w;Mp+=di->y[i]*w;}
  }
  double detailed=0.,merged=0.;
  for(size_t j=0;j<leaves.size();++j) {
    const double h=leaf_score(b[j],M[j],tau);
    detailed+=h;
    if(j==left)merged+=leaf_score(bp,Mp,tau);
    else if(j!=right)merged+=h;
  }
  const double delta=birth?detailed-merged:merged-detailed;
#ifdef ROBART_OPT_SHADOW
  const double ref=collapsed_score(proposed,grid,di,sd,tau)-collapsed_score(old,grid,di,sd,tau);
  if(!same_fp(delta,ref) && !(std::isnan(delta)&&std::isnan(ref)))
    throw std::logic_error("Paired collapsed likelihood differs from full evaluator.");
#endif
  return delta;
}
inline MoveTerms evaluate_move_reusing(tree& old,tree& proposed,const Context& ctx,pinfo& pi,
    const DepthPriorSpec& dp,dinfo* di,const double* sd,bool available_only,
    bool birth,size_t id,size_t v,size_t c,const PriorTerms& old_prior,
    const Pools& old_pools,PriorTerms& new_prior) {
  MoveTerms z;z.birth=birth;z.id=id;z.v=v;z.c=c;
  const double po=old_prior.total();
  if(!std::isfinite(po))throw std::runtime_error("Current tree is outside the formal prior support.");
  new_prior=prior(proposed,ctx,pi,dp);const double pn=new_prior.total();
  if(!std::isfinite(pn))return z;
  // Match reference order: only build the reverse pool after its forward term.
  z.forward=birth?log_birth_q_from_pools(old,ctx,old_pools,id,v,c,available_only)
                 :log_death_q_from_pools(old,old_pools,id);
  const Pools new_pools=pools(proposed,ctx,pi);
  z.reverse=birth?log_death_q_from_pools(proposed,new_pools,id)
                 :log_birth_q_from_pools(proposed,ctx,new_pools,id,v,c,available_only);
  if(!std::isfinite(z.forward)||!std::isfinite(z.reverse))return z;
  z.prior=pn-po;
  z.likelihood=paired_collapsed_difference(old,proposed,const_cast<xinfo&>(ctx.grid),di,sd,pi.tau,birth,id);
  z.logacc=z.likelihood+z.prior+z.reverse-z.forward;
  z.valid=std::isfinite(z.logacc);
#ifdef ROBART_OPT_SHADOW
  const MoveTerms ref=evaluate_move(old,proposed,ctx,pi,dp,di,sd,available_only,birth,id,v,c);
  if(z.valid!=ref.valid||!same_fp(z.likelihood,ref.likelihood)||!same_fp(z.prior,ref.prior)||
     !same_fp(z.forward,ref.forward)||!same_fp(z.reverse,ref.reverse)||!same_fp(z.logacc,ref.logacc))
    throw std::logic_error("Birth/death acceptance terms differ from full evaluator.");
#endif
  return z;
}
inline bool birth_death(tree& tr,xinfo& grid,dinfo* di,pinfo& pi,const double* sd,
    std::vector<size_t>& nv,const std::vector<double>& pv,bool available_only,rn& gen,
    const DepthPriorSpec& dp,const std::vector<double>* logs=nullptr,MoveTerms* diagnostics=nullptr,
    TreeCache* state_cache=nullptr) {
  std::vector<double> derived;
  if(!logs){derived=logs_from_weights(pv,di->p);logs=&derived;}
  const Context ctx=state_cache?state_cache->context(grid,di->x,di->p,di->n,logs)
                               :Context(grid,di->x,di->p,di->n,logs);
  const PriorTerms old_prior=state_cache?state_cache->get(tr,ctx,pi,dp):prior(tr,ctx,pi,dp);
  if(!std::isfinite(old_prior.total()))
    throw std::runtime_error("Current structural state violates formal tree prior support.");
  const Pools z=pools(tr,ctx,pi);
  if(z.births.empty()&&z.nogs.empty())return false;
  const bool birth=gen.uniform()<z.pb;
  size_t id=0,v=0,c=0;
  tree proposed(tr);
  if(birth) {
    if(z.births.empty())return false;
    const tree* node=z.births[sample_index(z.births.size(),gen)];id=node->nid();
    std::vector<size_t> vars;
    if(available_only)vars=ctx.good(node);
    else {vars.resize(di->p);for(size_t j=0;j<di->p;++j)vars[j]=j;}
    v=sample_coordinate(ctx,vars,gen);
    const Interval a=ctx.allowed(node,v);
    if(a.empty())return false; // failure mass, never resample to condition on success
    c=static_cast<size_t>(a.lo)+sample_index(a.size(),gen);
    proposed.birthp(proposed.getptr(id),v,c,0.,0.);
  } else {
    if(z.nogs.empty())return false;
    const tree* node=z.nogs[sample_index(z.nogs.size(),gen)];id=node->nid();v=node->getv();c=node->getc();
    proposed.deathp(proposed.getptr(id),0.);
  }
  PriorTerms new_prior;
  const MoveTerms terms=evaluate_move_reusing(tr,proposed,ctx,pi,dp,di,sd,available_only,
                                             birth,id,v,c,old_prior,z,new_prior);
  if(diagnostics)*diagnostics=terms;
  if(!terms.valid)return false;
  if(std::log(gen.uniform())>=terms.logacc)return false;
  if(nv.size()!=di->p || (!birth&&nv[v]==0) || (birth&&nv[v]==std::numeric_limits<size_t>::max()))
    throw std::runtime_error("Structural split counts are inconsistent.");
  tr=proposed;
  if(state_cache)state_cache->remember(tr,ctx,pi,dp,new_prior);
  if(birth)++nv[v];else --nv[v];
  return true;
}
}} // namespace robart::formal
#endif
