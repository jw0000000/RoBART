// Validated tree interchange for deterministic audit entry points.
// Copyright (C) 2026 Jeongung Heo. GPL (>= 2).
#ifndef ROBART_FORMAL_AUDIT_IO_H
#define ROBART_FORMAL_AUDIT_IO_H
#include "tree.h"
#include <sstream>
#include <map>
#include <limits>
#include <stdexcept>
namespace robart { namespace formal {
inline tree audit_tree(const std::string& text) {
  std::istringstream in(text);long long count=0;
  if(!(in>>count)||count<1||count>100001||count%2==0)throw std::invalid_argument("Invalid audit tree node count.");
  std::map<size_t,node_info> nodes;
  for(long long i=0;i<count;++i) {
    unsigned long long id,v,c;double mu;
    if(!(in>>id>>v>>c>>mu)||!id||!std::isfinite(mu)||
       id>std::numeric_limits<size_t>::max()||v>std::numeric_limits<size_t>::max()||c>std::numeric_limits<size_t>::max())
      throw std::invalid_argument("Malformed audit tree row.");
    node_info nd;nd.id=id;nd.v=v;nd.c=c;nd.theta=mu;
    if(!nodes.emplace(nd.id,nd).second)throw std::invalid_argument("Duplicate audit node ID.");
  }
  std::string extra;if(in>>extra)throw std::invalid_argument("Trailing audit tree tokens.");
  if(!nodes.count(1))throw std::invalid_argument("Audit tree root must be node 1.");
  for(const auto& kv:nodes) {
    const size_t id=kv.first;
    if(id!=1&&!nodes.count(id/2))throw std::invalid_argument("Missing audit tree parent.");
    if(id<=std::numeric_limits<size_t>::max()/2-1) {
      if(nodes.count(2*id)!=nodes.count(2*id+1))throw std::invalid_argument("Audit tree needs two children per internal node.");
    }
  }
  tree tr;tr.settheta(nodes.at(1).theta);
  for(const auto& kv:nodes) {
    const size_t id=kv.first;
    if(id<=std::numeric_limits<size_t>::max()/2-1 && nodes.count(2*id)) {
      tree* nd=tr.getptr(id);if(!nd)throw std::invalid_argument("Disconnected audit tree.");
      tr.birthp(nd,kv.second.v,kv.second.c,nodes.at(2*id).theta,nodes.at(2*id+1).theta);
    }
  }
  return tr;
}
inline std::string structure_string(const tree& tr) {
  tree::cnpv nodes;tr.getnodes(nodes);
  std::ostringstream out;out<<nodes.size()<<'\n';
  for(const tree* nd:nodes)out<<nd->nid()<<' '<<(nd->getl()?nd->getv():0)<<' '<<(nd->getl()?nd->getc():0)<<" 0\n";
  return out.str();
}
}} // namespace
#endif
