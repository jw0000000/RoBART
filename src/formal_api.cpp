/* Deterministic native audits of the production prior, cutpoint law and BD.
 * Copyright (C) 2026 Jeongung Heo. GPL (>= 2).
 * Internal registered entries: no fitting, latent update or RNG consumption.
 */
#include <Rcpp.h>
#include "formal_birth_death.h"
#include "formal_audit_io.h"
#include "rotation_mh.h"
namespace {
struct AuditInput {
  Rcpp::NumericMatrix X;
  size_t p,n;
  std::vector<double> x;
  xinfo grid;
  pinfo pi;
  DepthPriorSpec dp;
  std::vector<double> logs;
  bool available=false, enumerate=true;
  AuditInput(SEXP xs,SEXP gs,SEXP cs):X(xs),p(X.ncol()),n(X.nrow()) {
    if(!p||!n)Rcpp::stop("Audit design must be a nonempty n-by-p matrix.");
    x.resize(p*n);
    for(size_t i=0;i<n;++i)for(size_t j=0;j<p;++j)x[j+p*i]=X(i,j);
    Rcpp::List G(gs),C(cs);if(static_cast<size_t>(G.size())!=p)Rcpp::stop("Audit grid dimension mismatch.");
    for(size_t j=0;j<p;++j)grid.push_back(Rcpp::as<std::vector<double> >(G[j]));
    auto number=[&](const char* name,double value){return C.containsElementNamed(name)?Rcpp::as<double>(C[name]):value;};
    pi.alpha=number("base",.95);pi.mybeta=number("power",2.);pi.tau=number("tau",1.);pi.pb=number("pb",.5);
    const double typ=number("depth_type",0.);
    if(!std::isfinite(typ)||typ<0.||typ>2.||typ!=std::floor(typ))Rcpp::stop("Invalid audit depth type.");
    dp.type=static_cast<int>(typ);dp.tilt=number("tilt",0.);dp.nu=number("nu",.25);
    if(!std::isfinite(pi.alpha)||pi.alpha<=0||pi.alpha>=1||!std::isfinite(pi.mybeta)||pi.mybeta<0||
       !std::isfinite(pi.tau)||pi.tau<=0||!std::isfinite(pi.pb)||pi.pb<=0||pi.pb>=1||
       !std::isfinite(typ)||typ!=dp.type||dp.type<0||dp.type>2||!std::isfinite(dp.tilt)||dp.tilt<0||
       !std::isfinite(dp.nu)||dp.nu<=0||dp.nu>=1)Rcpp::stop("Invalid deterministic audit prior controls.");
    if(C.containsElementNamed("log_prob"))logs=Rcpp::as<std::vector<double> >(C["log_prob"]);
    else logs.assign(p,-std::log(static_cast<double>(p)));
    logs=robart::formal::normalized_log_weights(p,&logs);
    if(C.containsElementNamed("enumerate_moves")) {
      Rcpp::LogicalVector a=C["enumerate_moves"];if(a.size()!=1||a[0]==NA_LOGICAL)Rcpp::stop("enumerate_moves must be one logical value.");
      enumerate=a[0];
    }
    if(C.containsElementNamed("augment")) {
      Rcpp::LogicalVector a=C["augment"];if(a.size()!=1||a[0]==NA_LOGICAL)Rcpp::stop("augment must be one logical value.");
      available=a[0];
    }
  }
};
}
RcppExport SEXP _robart_formal_tree_audit(SEXP xs,SEXP ys,SEXP ss,SEXP ts,SEXP gs,SEXP cs) {
BEGIN_RCPP
  AuditInput a(xs,gs,cs);Rcpp::NumericVector y(ys),sd(ss);
  if(static_cast<size_t>(y.size())!=a.n||static_cast<size_t>(sd.size())!=a.n)Rcpp::stop("Audit residual/SD length mismatch.");
  tree tr=robart::formal::audit_tree(Rcpp::as<std::string>(ts));
  const robart::formal::Context ctx(a.grid,a.x.data(),a.p,a.n,&a.logs);
  const robart::formal::PriorTerms lp=robart::formal::prior(tr,ctx,a.pi,a.dp);
  dinfo di;di.p=a.p;di.n=a.n;di.x=a.x.data();di.y=y.begin();
  Rcpp::List ans;ans["valid"]=lp.valid;ans["log_prior"]=lp.total();ans["log_terminal"]=lp.terminal;
  if(!lp.valid){ans["log_marginal"]=R_NegInf;ans["moves"]=Rcpp::List();return ans;}
  ans["log_marginal"]=robart::formal::collapsed_score(tr,a.grid,&di,sd.begin(),a.pi.tau);
  const auto pool=robart::formal::pools(tr,ctx,a.pi);ans["birth_probability"]=pool.pb;
  Rcpp::List moves;
  if(!a.enumerate){ans["moves"]=moves;return ans;}
  auto append=[&](bool birth,size_t id,size_t v,size_t c){
    tree prop(tr);if(birth)prop.birthp(prop.getptr(id),v,c,0.,0.);else prop.deathp(prop.getptr(id),0.);
    const auto z=robart::formal::evaluate_move(tr,prop,ctx,a.pi,a.dp,&di,sd.begin(),a.available,birth,id,v,c);
    Rcpp::List r;r["birth"]=birth;r["node"]=static_cast<double>(id);r["variable"]=static_cast<double>(v);r["cut"]=static_cast<double>(c);
    r["tree"]=robart::formal::structure_string(prop);r["logq_forward"]=z.forward;r["logq_reverse"]=z.reverse;
    r["log_prior_ratio"]=z.prior;r["log_likelihood_ratio"]=z.likelihood;r["logacc"]=z.logacc;r["valid"]=z.valid;
    moves.push_back(r);
  };
  for(tree* node:pool.births)for(size_t v:ctx.good(node)) {
    const auto A=ctx.allowed(node,v);for(int c=A.lo;c<=A.hi;++c)append(true,node->nid(),v,c);
  }
  for(tree* node:pool.nogs)append(false,node->nid(),node->getv(),node->getc());
  ans["moves"]=moves;return ans;
END_RCPP
}
RcppExport SEXP _robart_formal_cut_audit(SEXP xo,SEXP xn,SEXP go,SEXP gn,SEXP to,SEXP tn,SEXP touch,SEXP control) {
BEGIN_RCPP
  AuditInput a(xo,go,control),b(xn,gn,control);
  if(a.p!=b.p||a.n!=b.n)Rcpp::stop("Cut audit designs disagree.");
  tree old=robart::formal::audit_tree(Rcpp::as<std::string>(to)),proposed=robart::formal::audit_tree(Rcpp::as<std::string>(tn));
  Rcpp::LogicalVector tv(touch);if(static_cast<size_t>(tv.size())!=a.p)Rcpp::stop("touched length mismatch.");
  std::vector<char> touched(a.p);for(size_t j=0;j<a.p;++j){if(tv[j]==NA_LOGICAL)Rcpp::stop("touched has NA.");touched[j]=tv[j];}
  robart::CutSnapshot A,B;robart::snapshot_internal_cuts(old,A);robart::snapshot_internal_cuts(proposed,B);
  if(A.nodes.size()!=B.nodes.size())Rcpp::stop("Cut audit topologies disagree.");
  for(size_t j=0;j<A.nodes.size();++j)if(A.nodes[j]->nid()!=B.nodes[j]->nid()||A.nodes[j]->getv()!=B.nodes[j]->getv())Rcpp::stop("Cut audit labels/topology disagree.");
  const robart::formal::Context ac(a.grid,a.x.data(),a.p,a.n,&a.logs),bc(b.grid,b.x.data(),b.p,b.n,&b.logs);
  const double lpold=robart::formal::prior(old,ac,a.pi,a.dp).total(),lpnew=robart::formal::prior(proposed,bc,b.pi,b.dp).total();
  double f=R_NegInf,r=R_NegInf;
  if(std::isfinite(lpold)) {
    std::vector<tree::tree_cp> leaves(a.n);for(size_t i=0;i<a.n;++i)leaves[i]=old.bn(a.x.data()+a.p*i,a.grid);
    robart::CutSnapshot target=A;target.cuts=B.cuts;
    f=robart::sequential_cutpoint_rule(old,bc,a.p,a.n,leaves,b.x.data(),touched,nullptr,&target);
  }
  if(std::isfinite(lpnew)) {
    std::vector<tree::tree_cp> leaves(b.n);for(size_t i=0;i<b.n;++i)leaves[i]=proposed.bn(b.x.data()+b.p*i,b.grid);
    robart::CutSnapshot target=B;target.cuts=A.cuts;
    r=robart::sequential_cutpoint_rule(proposed,ac,b.p,b.n,leaves,a.x.data(),touched,nullptr,&target);
  }
  Rcpp::List ans;ans["logq_forward"]=f;ans["logq_reverse"]=r;ans["log_prior_old"]=lpold;ans["log_prior_new"]=lpnew;
  ans["source_restored"]=robart::formal::structure_string(old)==robart::formal::structure_string(robart::formal::audit_tree(Rcpp::as<std::string>(to)));
  return ans;
END_RCPP
}
