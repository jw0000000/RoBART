/*
 * Tree-specific centered Givens proposals with joint cutpoint retuning.
 * Include from bart.cpp and heterbart.cpp. The accepted state contains the
 * tree, rotation, rotated design and per-tree cutpoint grid.
 * Sufficient statistics are recomputed with tree::bn and the supplied grid.
 * Define ROBART_ROTATION_MH_DEBUG for logging and consistency checks.
 * Author: Jeongung Heo
 */
#ifndef ROBART_ROTATION_MH_H
#define ROBART_ROTATION_MH_H

#include <RcppEigen.h>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <chrono>


#include "common.h"
#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "canonical_coordinates.h"
#include "formal_tree_cache.h"

namespace robart {

// -----------------------------------------------------------------------------
// Lightweight timing container (milliseconds). Filled only when requested.
// -----------------------------------------------------------------------------
struct RotTiming {
  double t_seq_ms      = 0.0; // sample K Givens + touched
  double t_xprop_ms    = 0.0; // clone X + apply K Givens
  double t_cut_fwd_ms  = 0.0; // forward cutpoint proposal
  double t_diag_ms     = 0.0; // optional partition diagnostics
  double t_ll_ms       = 0.0; // ll_old + ll_new
  double t_cut_bwd_ms  = 0.0; // backward logq for returning to old cutpoints
  double t_accept_ms   = 0.0; // accept/reject bookkeeping (R, buffers)
  double t_total_ms    = 0.0; // total inside the centered rotation block
};

inline double _ms_between_(const std::chrono::steady_clock::time_point& t0,
                           const std::chrono::steady_clock::time_point& t1)
{
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct RotState {
  Eigen::MatrixXd R;            // p x p
  // Native buffers deliberately use std::vector rather than Rcpp vectors.
  // These arrays live for the whole .Call and are referenced through raw
  // pointers by dinfo.  Keeping them outside the R heap prevents an R garbage
  // collection (for example while serialising saved states) from interacting
  // with the training/test design buffers.
  std::vector<double> Xtr;      // length p*n, observation-major: Xtr[a + p*i]
  std::vector<double> Xte;      // length p*np (optional)
  bool has_test = false;
  // Quantile caches are functions of the current Q and immutable .Call inputs.
  // Borrowed pointers do not own memory; those inputs outlive this native model.
  bool canonical_coordinates=false;
  const double* raw_train=nullptr;
  const double* raw_test=nullptr;
  size_t raw_p=0, raw_n=0, raw_np=0;
  bool canonical_test_dirty=false;

  struct Pending { size_t a, b; double c, s; }; // Plane indices and the cosine/sine of a pending rotation.
  std::vector<Pending> pending;

  // ---------------------------------------------------------------------------
  // Angle accumulator for accepted rotation moves (oracle comparison / plotting)
  //   - angle_sum(a,b): cumulative theta (radians) applied in plane (a,b)
  //   - angle_sum(b,a) is maintained as -angle_sum(a,b) (antisymmetric)
  //   - angle_count(a,b): how many times the plane (a,b) was updated (accepted)
  // Notes:
  //   - Many entries can remain zero if the plane is rarely proposed/accepted.
  //   - These are initialized lazily (on first accept) to avoid unnecessary alloc.
  // ---------------------------------------------------------------------------
  Eigen::MatrixXd angle_sum;    // p x p
  Eigen::MatrixXi angle_count;  // p x p
};


// Forward declarations (some helpers are defined later in the header)
inline void apply_givens_left_inplace(Eigen::MatrixXd& R,
                                      size_t a, size_t b,
                                      double c, double s);

// A single truncated discrete-Gaussian proposal on cutpoint indices.
// W is the CURRENT allowed interval length. This is one component, not one
// universal numeric width: s(W) = max(1, 0.02 W). No public scale parameter.
inline double cutpoint_index_scale(int L, int U)
{
  const double W = static_cast<double>(U) - static_cast<double>(L) + 1.0;
  return std::max(1.0, 0.02 * W);
}

inline double logZ_discrete_gauss(int L, int U, int center, double s)
{
  if (U < L) return -std::numeric_limits<double>::infinity();
  if (s <= 0.0 || !std::isfinite(s)) return -std::numeric_limits<double>::infinity();
  double sum = 0.0;
  for (int c = L; c <= U; ++c) {
    const double d = (double)c - (double)center;
    const double e = -0.5 * (d*d) / (s*s);
    sum += std::exp(e);
  }
  return std::log(sum);
}

inline double logphi_discrete_gauss(int c, int L, int U, int center, double s, double logZ)
{
  if (c < L || c > U) return -std::numeric_limits<double>::infinity();
  if (!std::isfinite(logZ)) return -std::numeric_limits<double>::infinity();
  const double d = (double)c - (double)center;
  const double e = -0.5 * (d*d) / (s*s);
  return e - logZ;
}

inline double logq_cutpoint_index(int c, int L, int U, int center)
{
  if (U < L || c < L || c > U || center < L || center > U)
    return -std::numeric_limits<double>::infinity();
  const double scale = cutpoint_index_scale(L, U);
  const double logZ = logZ_discrete_gauss(L, U, center, scale);
  return logphi_discrete_gauss(c, L, U, center, scale, logZ);
}

inline int sample_discrete_gauss_trunc(int L, int U, int center, double s, rn& gen)
{
  if (U <= L) return L;
  double maxlogw = -std::numeric_limits<double>::infinity();
  const int W = U - L + 1;
  std::vector<double> logw((size_t)W);
  for (int c = L; c <= U; ++c) {
    const double d = (double)c - (double)center;
    const double lw = -0.5 * (d*d) / (s*s);
    logw[(size_t)(c - L)] = lw;
    if (lw > maxlogw) maxlogw = lw;
  }
  double sumw = 0.0;
  for (int i = 0; i < W; ++i) sumw += std::exp(logw[(size_t)i] - maxlogw);
  const double u = gen.uniform() * sumw;
  double acc = 0.0;
  for (int i = 0; i < W; ++i) {
    acc += std::exp(logw[(size_t)i] - maxlogw);
    if (u <= acc) return L + i;
  }
  return U;
}

inline int sample_cutpoint_index(int L, int U, int center, rn& gen,
                                  double* out_logq)
{
  if (U < L || center < L || center > U)
    throw std::invalid_argument("Invalid cutpoint interval/center");
  // Keep singleton intervals deterministic; a non-singleton uses one uniform
  // draw, rather than a mixture-selector draw followed by an index draw.
  if (U == L) {
    if (out_logq) *out_logq = 0.0;
    return L;
  }
  const int c = sample_discrete_gauss_trunc(L, U, center,
                                           cutpoint_index_scale(L, U), gen);
  if (out_logq) *out_logq = logq_cutpoint_index(c, L, U, center);
  return c;
}


// Sample a rotation plane (a,b).
// cand_vars is the current tree's split-variable support S_t.
//
// For support_expand_prob = eta in [0,1]:
//   - eta = 0: use the existing within-support pair rule.
//   - 0 < eta <= 1:
//       * If |S_t| >= 2 and the complement is nonempty:
//           select one in-support and one out-of-support coordinate with probability eta;
//           otherwise select two in-support coordinates.
//       * If |S_t| = 1, select one in-support and one out-of-support coordinate when possible.
//       * If |S_t| = 0, sample from all coordinates.
//       * If |S_t| = p, select only within the support.
//
// The rule allows refinement of existing split directions and expansion
inline size_t draw_uniform_index(size_t m, rn& gen)
{
  if (m <= 1) return 0;
  double u = gen.uniform();
  size_t idx = (size_t)(u * (double)m);
  if (idx >= m) idx = m - 1;
  return idx;
}

inline void sample_rotation_pair_global(size_t p,
                                        rn& gen,
                                        size_t& a,
                                        size_t& b)
{
  if (p < 2) {
    a = 0; b = 0;
    return;
  }
  a = draw_uniform_index(p, gen);
  b = draw_uniform_index(p, gen);
  while (b == a) b = draw_uniform_index(p, gen);
}

inline void sample_rotation_pair_inside(const std::vector<size_t>& pool,
                                        size_t p,
                                        rn& gen,
                                        size_t& a,
                                        size_t& b)
{
  if (pool.empty()) {
    sample_rotation_pair_global(p, gen, a, b);
    return;
  }
  if (pool.size() == 1) {
    a = pool[0];
    if (a >= p) a %= p;
    if (p == 2) {
      b = (a == 0 ? 1 : 0);
      return;
    }
    b = draw_uniform_index(p, gen);
    while (b == a) b = draw_uniform_index(p, gen);
    return;
  }

  const size_t ia = draw_uniform_index(pool.size(), gen);
  size_t ib = draw_uniform_index(pool.size(), gen);
  while (ib == ia) ib = draw_uniform_index(pool.size(), gen);
  a = pool[ia];
  b = pool[ib];
  if (a >= p) a %= p;
  if (b >= p) b %= p;
  if (a == b) b = (a + 1) % p;
}

inline void build_support_complement(size_t p,
                                     const std::vector<size_t>& support,
                                     std::vector<size_t>& outside)
{
  outside.clear();
  if (p == 0) return;
  std::vector<char> in_support(p, 0);
  for (size_t v : support) {
    if (v < p) in_support[v] = 1;
  }
  outside.reserve(p > support.size() ? p - support.size() : 0);
  for (size_t v = 0; v < p; ++v) {
    if (!in_support[v]) outside.push_back(v);
  }
}

inline void sample_rotation_pair(size_t p,
                                 rn& gen,
                                 const std::vector<size_t>* cand_vars,
                                 size_t& a,
                                 size_t& b,
                                 double support_expand_prob = 0.0)
{
  if (p < 2) {
    a = 0; b = 0;
    return;
  }

  double eta = support_expand_prob;
  if (!std::isfinite(eta)) eta = 0.0;
  if (eta < 0.0) eta = 0.0;
  if (eta > 1.0) eta = 1.0;

  if (!cand_vars || cand_vars->empty()) {
    sample_rotation_pair_global(p, gen, a, b);
    return;
  }

  // sanitize support (in-range, unique)
  std::vector<size_t> support;
  support.reserve(cand_vars->size());
  std::vector<char> seen(p, 0);
  for (size_t v : *cand_vars) {
    if (v < p && !seen[v]) {
      support.push_back(v);
      seen[v] = 1;
    }
  }

  if (support.empty()) {
    sample_rotation_pair_global(p, gen, a, b);
    return;
  }

  // At eta=0, retain the within-support pair proposal.
  if (eta <= 0.0) {
    sample_rotation_pair_inside(support, p, gen, a, b);
    return;
  }

  // Coordinates outside the current split-variable support.
  std::vector<size_t> outside;
  build_support_complement(p, support, outside);

  // A full support cannot be expanded; use the within-support rule.
  if (outside.empty()) {
    sample_rotation_pair_inside(support, p, gen, a, b);
    return;
  }

  // A singleton support requires one in-support and one out-of-support coordinate.
  if (support.size() == 1) {
    a = support[0];
    b = outside[draw_uniform_index(outside.size(), gen)];
    if (gen.uniform() < 0.5) std::swap(a, b);
    return;
  }

  // For larger supports, choose between refinement and expansion.
  if (gen.uniform() < eta) {
    a = support[draw_uniform_index(support.size(), gen)];
    b = outside[draw_uniform_index(outside.size(), gen)];
    if (gen.uniform() < 0.5) std::swap(a, b);
  } else {
    sample_rotation_pair_inside(support, p, gen, a, b);
  }
}


// Collect the coordinate indices used by internal splits.
// Return sorted, unique indices.
inline void collect_split_vars(tree& tr, size_t p, std::vector<size_t>& out)
{
  tree::npv nodes;
  tr.getnodes(nodes);
  out.clear();
  out.reserve(nodes.size());
  for (auto nd : nodes) {
    if (!nd) continue;
    if (nd->getl()) { // Internal split node.
      size_t v = nd->getv();
      if (v < p) out.push_back(v);
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}


inline Eigen::MatrixXd givens_rotation(int p, size_t a, size_t b, double theta)
{
  Eigen::MatrixXd G = Eigen::MatrixXd::Identity(p,p);
  double c = std::cos(theta), s = std::sin(theta);
  G(a,a)=c; G(b,b)=c; G(a,b)=-s; G(b,a)=s;
  return G;
}

inline Eigen::VectorXd signal_dir(size_t p){
  Eigen::VectorXd v = Eigen::VectorXd::Zero((int)p);
  if (p>=1) v(0)=1.0; if (p>=2) v(1)=1.0;
  double norm = std::sqrt(p>=2 ? 2.0 : 1.0);
  v /= norm;
  return v;
}

inline void tree_alignment_from_R(const Eigen::MatrixXd& R,
                                  double& angle_deg,
                                  double& delta_deg,
                                  double& align_score)
{
  const int p = (int)R.rows();
  Eigen::VectorXd v = signal_dir((size_t)p);
  Eigen::VectorXd w = R * v; // Rotated reference direction.
  int kstar; align_score = w.cwiseAbs().maxCoeff(&kstar);
  double ux = R(kstar, 0);
  double uy = (p>=2 ? R(kstar, 1) : 0.0);
  angle_deg = std::atan2(uy, ux) * 180.0 / M_PI;
  double d = angle_deg - 45.0;             // Angular difference from 45 degrees.
  d = std::remainder(d, 180.0);            // Fold using a 180-degree period, with results in [-90,90].
  delta_deg = d;
}

// -----------------------------------------------------------------------------
// Canonical (deterministic) Givens decomposition angles from an orthogonal matrix
// -----------------------------------------------------------------------------
//
// We apply left-multiplications to A <- G * A to eliminate the strict lower
// triangle. The corresponding "forward" angles for x' = G x are
// theta_use = -theta_elim.
//
// Output:
//   ang: p x p antisymmetric, ang(a,b) = -ang(b,a).
//        In this elimination order, each plane (a,b) is used at most once
//        (0 if no elimination needed).
//
inline void givens_plane_angles_from_R(const Eigen::MatrixXd& R,
                                       Eigen::MatrixXd& ang,
                                       double tol = 1e-12)
{
  const int p = (int)R.rows();
  ang = Eigen::MatrixXd::Zero(p, p);
  if (p < 2) return;

  Eigen::MatrixXd A = R;
  for (int j = 0; j < p - 1; ++j) {
    for (int i = p - 1; i >= j + 1; --i) {
      const double aa = A(j, j);
      const double bb = A(i, j);
      if (std::fabs(bb) < tol) continue;
      const double r = std::sqrt(aa * aa + bb * bb);
      if (r < tol) continue;
      const double c = aa / r;
      const double s = -bb / r;

      // Apply to rows (j,i). Only columns k>=j can be nonzero under this
      // elimination order.
      for (int k = j; k < p; ++k) {
        const double rowj = A(j, k);
        const double rowi = A(i, k);
        A(j, k) = c * rowj - s * rowi;
        A(i, k) = s * rowj + c * rowi;
      }

      const double theta_elim = std::atan2(s, c);
      const double theta_use  = -theta_elim;
      ang(j, i) += theta_use;
      ang(i, j) -= theta_use;
    }
  }
}


// Find the nearest grid index to value within [L,U].
inline size_t nearest_index_in_range(const std::vector<double>& grid,
                                       int L, int U, double value)
{
  if (L < 0) L = 0;
  if (U >= (int)grid.size()) U = (int)grid.size()-1;
  if (U < L) return (size_t)L;
  auto b = grid.begin()+L, e = grid.begin()+U+1;
  auto it = std::lower_bound(b, e, value);
  if (it == b) return (size_t)L;
  if (it == e) return (size_t)U;
  size_t i1 = (size_t)(it - grid.begin());
  size_t i0 = i1 - 1;
  return ( std::fabs(grid[i0]-value) <= std::fabs(grid[i1]-value) ) ? i0 : i1;
}

// Apply all pending rotations to the test buffer in place.
inline void flush_pending_test_rotations(RotState& st, size_t p, size_t np)
{
  if(st.canonical_coordinates){
    if(!st.has_test || np==0)return;
    if(st.raw_p!=p || st.raw_np!=np || !st.raw_test || st.Xte.size()!=p*np)
      throw std::invalid_argument("Canonical test design registration mismatch.");
    if(st.canonical_test_dirty){
      canonical_project(st.R,st.raw_test,p,np,st.Xte.data());
      st.canonical_test_dirty=false;
    }
    st.pending.clear();return;
  }
  if (!st.has_test || np==0 || st.pending.empty()) return;
  if ((size_t)st.Xte.size() != p*np) return; // Skip if the test buffer has not been initialized to the expected size.
  for (const auto& u : st.pending) {
    const size_t a = u.a, b = u.b; const double c = u.c, s = u.s;
    for (size_t i=0; i<np; ++i) {
      double xa = st.Xte[a + p*i];
      double xb = st.Xte[b + p*i];
      double na =  c*xa - s*xb;
      double nb =  s*xa + c*xb;
      st.Xte[a + p*i] = na;
      st.Xte[b + p*i] = nb;
    }
  }
  st.pending.clear();
}


// Build xinfo from raw X (p x n), keeping the same number of cuts as xi_old per variable
inline void make_xinfo_from_x(double *x, size_t p, size_t n, const xinfo& xi_old, xinfo& xi_new)
{
  xi_new = xinfo(p);
  for (size_t j=0; j<p; ++j) {
    size_t nc = xi_old[j].size();
    std::vector<double> vals; vals.reserve(n);
    for (size_t i=0; i<n; ++i) {
      vals.push_back(*(x + p*i + j));
    }
    std::sort(vals.begin(), vals.end());
    if (vals.size() == 0) {
      xi_new[j] = std::vector<double>(nc, 0.0);
      continue;
    }
    // Equally spaced quantile-based cutpoints between min and max (exclusive)
    xi_new[j].resize(nc);
    for (size_t k=0; k<nc; ++k) {
      double q = ( (double) (k+1) ) / ( (double) (nc+1) );
      size_t idx = std::min((size_t) std::floor(q * (vals.size()-1)), vals.size()-1);
      xi_new[j][k] = vals[idx];
    }
  }
}

// Retuning utilities targeting previous membership; ties and grid snapping may change it.

inline bool is_descendant(const tree* node, const tree* ancestor)
{
  if (!node || !ancestor) return false;
  const tree* p = node;
  while (p) {
    if (p == ancestor) return true;
    p = p->getp();
  }
  return false;
}

inline void precompute_bottom_nodes(tree& t, xinfo& xi_old, size_t p, size_t n, double* x_old,
                                    std::vector<tree::tree_cp>& bn_old)
{
  bn_old.resize(n);
  for (size_t i=0; i<n; ++i) {
    double* xi = x_old + p*i;
    bn_old[i] = t.bn(xi, xi_old);
  }
}

// Recursively retune all internal nodes of tr using new xinfo / X to preserve L/R
inline void retune_node(tree::tree_p nd,
                        xinfo& xi_old, xinfo& xi_new,
                        size_t p, size_t n,
                        double* x_old, double* x_new,
                        const std::vector<tree::tree_cp>& bn_old)
{
  if (!nd || !nd->getl()) return; // bottom -> nothing to retune

  size_t v = nd->getv();          // split variable index
  // Collect indices that flowed through this node originally + L/R labels
  std::vector<size_t> idx;  idx.reserve(n);
  std::vector<int>     lab;  lab.reserve(n); // 1: left, 0: right

  for (size_t i=0; i<n; ++i) {
    tree::tree_cp b = bn_old[i];
    if (is_descendant(b, nd)) {
      int is_left = is_descendant(b, nd->getl()) ? 1 : 0;
      idx.push_back(i);
      lab.push_back(is_left);
    }
  }

  int L=0, U = (int)xi_new[v].size()-1;
  nd->rg(v, &L, &U); // allowable cut index range per ancestors

  if (U >= L && !idx.empty()) {
    size_t best_c = (size_t)L;
    size_t best_match = 0;

#ifdef ROBART_ROTATION_MH_DEBUG
    size_t iter = 0;
#endif
    for (int c=L; c<=U; ++c) {
      double thr = xi_new[v][c];
      size_t match = 0;
      for (size_t k=0; k<idx.size(); ++k) {
        size_t i = idx[k];
        double val_new = *(x_new + p*i + v);
        int pred_left = (val_new < thr) ? 1 : 0;
        if (pred_left == lab[k]) ++match;
      }
      if (match > best_match) { best_match = match; best_c = (size_t)c; }
#ifdef ROBART_ROTATION_MH_DEBUG
      ++iter;
#endif
    }
    nd->setc(best_c);
  }

  // recursive
  retune_node(nd->getl(), xi_old, xi_new, p, n, x_old, x_new, bn_old);
  retune_node(nd->getr(), xi_old, xi_new, p, n, x_old, x_new, bn_old);
}

inline void retune_cutpoints_preserve_membership(tree& tr,
                                          xinfo& xi_old, xinfo& xi_new,
                                          size_t p, size_t n,
                                          double* x_old, double* x_new)
{
  std::vector<tree::tree_cp> bn_old;
  precompute_bottom_nodes(tr, xi_old, p, n, x_old, bn_old);
  retune_node(&tr, xi_old, xi_new, p, n, x_old, x_new, bn_old);
}

// Diagnostic percentile retuning on (a,b), targeting the previous left-child count.
inline void retune_node_percentile(tree::tree_p nd,
                                     xinfo& xi,              // Reuse the supplied grid during this traversal.
                                     size_t p, size_t n,
                                     const double* x_old,    // Current coordinates from st.Xtr.
                                     size_t a, size_t b,
                                     double c, double s,
                                     const std::vector<tree::tree_cp>& bn_old)
  {
    if (!nd || !nd->getl()) return; // bottom
    size_t v = nd->getv();
    // For other split coordinates, recurse without changing this node.
    if (v != a && v != b) {
      retune_node_percentile(nd->getl(), xi, p, n, x_old, a, b, c, s, bn_old);
      retune_node_percentile(nd->getr(), xi, p, n, x_old, a, b, c, s, bn_old);
      return;
    }
    // Collect observations visiting this node and their previous left/right labels.
    std::vector<size_t> idx; idx.reserve(n);
    std::vector<int>    lab; lab.reserve(n);
    for (size_t i=0; i<n; ++i) {
      tree::tree_cp btm = bn_old[i];
      if (is_descendant(btm, nd)) {
        int is_left = is_descendant(btm, nd->getl()) ? 1 : 0;
        idx.push_back(i); lab.push_back(is_left);
      }
    }
    if (idx.empty()) {
      // Continue recursively through the children.
      retune_node_percentile(nd->getl(), xi, p, n, x_old, a, b, c, s, bn_old);
      retune_node_percentile(nd->getr(), xi, p, n, x_old, a, b, c, s, bn_old);
      return;
    }
    // Count observations previously routed to the left child.
    size_t Lcnt = 0; for (int v_ : lab) if (v_==1) ++Lcnt;
    // Compute the updated split-coordinate values for the two affected axes.
    std::vector<double> vals; vals.reserve(idx.size());
    if (v == a) {
      for (size_t i : idx) {
        double xa = x_old[a + p*i], xb = x_old[b + p*i];
        vals.push_back( c*xa - s*xb );
      }
    } else { // v==b
      for (size_t i : idx) {
        double xa = x_old[a + p*i], xb = x_old[b + p*i];
        vals.push_back( s*xa + c*xb );
      }
    }
    // Use order statistic Lcnt-1 as a threshold target; ties and grid snapping can change membership.
    if (Lcnt > 0 && Lcnt <= vals.size()) {
      size_t kth = Lcnt-1;
      std::nth_element(vals.begin(), vals.begin()+kth, vals.end());
      double thr = vals[kth];
      // Snap to an admissible grid index under the ancestor constraints.
      int L=0, U=(int)xi[v].size()-1;
      nd->rg(v, &L, &U);
      if (U >= L && U >= 0) {
        size_t cidx = nearest_index_in_range(xi[v], L, U, thr);
        nd->setc((size_t)cidx);
      }
    }
    // Continue recursively through the children.
    retune_node_percentile(nd->getl(), xi, p, n, x_old, a, b, c, s, bn_old);
    retune_node_percentile(nd->getr(), xi, p, n, x_old, a, b, c, s, bn_old);
  }
  
inline void retune_cutpoints_percentile_match(tree& tr,
                                              xinfo& xi,    // Reuse the supplied grid.
                                              size_t p, size_t n,
                                              const double* x_old, // st.Xtr
                                              size_t a, size_t b,
                                              double c, double s)
{
  std::vector<tree::tree_cp> bn_old;
  precompute_bottom_nodes(tr, xi, p, n, const_cast<double*>(x_old), bn_old);
  retune_node_percentile(&tr, xi, p, n, x_old, a, b, c, s, bn_old);
}


// === Proposal diagnostics: how much does a rotation+retune change the partition? ===
// We use leaf node ids (tree::nid()) which are path-based and stable across copies.
struct LeafStat { size_t n = 0; double sy = 0.0; };

inline void compute_partition_diagnostics(
    tree& t_old,
    tree& t_new,
    xinfo& xi,
    const double* X_old,
    const double* X_new,
    const double* y,
    size_t p,
    size_t n,
    double& out_memb_change,
    double& out_leaf_n_L1,
    double& out_leaf_sy_L1)
{
  out_memb_change = 0.0;
  out_leaf_n_L1 = 0.0;
  out_leaf_sy_L1 = 0.0;
  if (n == 0 || p == 0 || !X_old || !X_new || !y) {
    out_memb_change = NA_REAL;
    out_leaf_n_L1   = NA_REAL;
    out_leaf_sy_L1  = NA_REAL;
    return;
  }

  std::unordered_map<size_t, LeafStat> old_stats;
  std::unordered_map<size_t, LeafStat> new_stats;
  old_stats.reserve(n);
  new_stats.reserve(n);

  size_t changed = 0;
  double sum_abs = 0.0;

  for (size_t i = 0; i < n; ++i) {
    const double* xo = X_old + p*i;
    const double* xn = X_new + p*i;
    tree::tree_cp lo = t_old.bn(const_cast<double*>(xo), xi);
    tree::tree_cp ln = t_new.bn(const_cast<double*>(xn), xi);
    if (!lo || !ln) continue;
    const size_t ido = lo->nid();
    const size_t idn = ln->nid();
    if (ido != idn) ++changed;

    const double yi = y[i];
    sum_abs += std::fabs(yi);

   auto &so = old_stats[ido];
    so.n += 1;
    so.sy += yi;

    auto &sn = new_stats[idn];
    sn.n += 1;
    sn.sy += yi;
  }

  out_memb_change = (double)changed / (double)n;

  double l1n = 0.0;
  double l1sy = 0.0;
  std::unordered_set<size_t> seen;
  seen.reserve(old_stats.size() + new_stats.size());

  for (const auto& kv : old_stats) {
    const size_t id = kv.first;
    const LeafStat& so = kv.second;
    const auto it = new_stats.find(id);
    const size_t nn = (it == new_stats.end() ? 0 : it->second.n);
    const double syn = (it == new_stats.end() ? 0.0 : it->second.sy);
    l1n  += std::fabs((double)nn - (double)so.n);
    l1sy += std::fabs(syn - so.sy);
    seen.insert(id);
  }
  for (const auto& kv : new_stats) {
    const size_t id = kv.first;
    if (seen.find(id) != seen.end()) continue;
    const LeafStat& sn = kv.second;
    l1n  += (double)sn.n;
   l1sy += std::fabs(sn.sy);
  }

  out_leaf_n_L1 = l1n / (2.0 * (double)n);
  const double denom = 2.0 * (sum_abs + 1e-12);
  out_leaf_sy_L1 = l1sy / denom;
}

// Same diagnostics, but using precomputed leaf assignments (bn) under the
// old/new states. This avoids calling tree::bn() when bn vectors are already
// available from the MH step.
inline void compute_partition_diagnostics_from_bn(
    const std::vector<tree::tree_cp>& bn_old,
    const std::vector<tree::tree_cp>& bn_new,
    const double* y,
    size_t n,
    double& out_memb_change,
    double& out_leaf_n_L1,
    double& out_leaf_sy_L1)
{
  out_memb_change = NA_REAL;
  out_leaf_n_L1   = NA_REAL;
  out_leaf_sy_L1  = NA_REAL;

  if (n == 0 || y == nullptr || bn_old.size() < n || bn_new.size() < n) return;

  struct LeafStat { size_t n = 0; double sy = 0.0; };
  std::unordered_map<size_t, LeafStat> old_stats;
  std::unordered_map<size_t, LeafStat> new_stats;
  old_stats.reserve(64);
  new_stats.reserve(64);

  size_t changed = 0;
  double sum_abs = 0.0;

  for (size_t i = 0; i < n; ++i) {
    const tree::tree_cp lo = bn_old[i];
    const tree::tree_cp ln = bn_new[i];
    if (!lo || !ln) continue;

    const size_t ido = lo->nid();
    const size_t idn = ln->nid();
    if (ido != idn) ++changed;

    const double yi = y[i];
    sum_abs += std::fabs(yi);

    auto& so = old_stats[ido];
    so.n += 1;
    so.sy += yi;

    auto& sn = new_stats[idn];
    sn.n += 1;
    sn.sy += yi;
  }

  out_memb_change = (double)changed / (double)n;

  double l1n = 0.0;
  double l1sy = 0.0;
  std::unordered_set<size_t> seen;
  seen.reserve(old_stats.size() * 2 + 1);

  for (const auto& kv : old_stats) {
    const size_t id = kv.first;
    const LeafStat& so = kv.second;
    const auto it = new_stats.find(id);
    const size_t nn = (it == new_stats.end() ? 0 : it->second.n);
    const double syn = (it == new_stats.end() ? 0.0 : it->second.sy);
    l1n  += std::fabs((double)nn - (double)so.n);
    l1sy += std::fabs(syn - so.sy);
    seen.insert(id);
  }
  for (const auto& kv : new_stats) {
    const size_t id = kv.first;
    if (seen.find(id) != seen.end()) continue;
    const LeafStat& sn = kv.second;
    l1n  += (double)sn.n;
    l1sy += std::fabs(sn.sy);
  }

  out_leaf_n_L1 = l1n / (2.0 * (double)n);
  const double denom = 2.0 * (sum_abs + 1e-12);
  out_leaf_sy_L1 = l1sy / denom;
}


// === Marginal likelihoods (integrating out leaf means) ===

// Homoskedastic case: sigma^2 shared; prior mu ~ N(0, tau^2)
// log p(y | partition) up to a constant that cancels in MH
inline double tree_log_marglik_homo(tree& t, xinfo& xi, dinfo* di_ptr, double sigma, double tau)
{
  const size_t p = di_ptr->p;
  const size_t n = di_ptr->n;
  double* x = di_ptr->x;
  double* y = di_ptr->y;

  // Accumulate per-bottom node: n, sum y
  // NOTE: unordered_map is materially faster than std::map here.
  struct Stat { size_t n = 0; double sy = 0.0; };
  std::unordered_map<const tree*, Stat> stats;
  const size_t nb = t.nbots();
  stats.reserve(nb * 2 + 1);

  for (size_t i=0; i<n; ++i) {
    double* xirow = x + p*i;
    tree::tree_cp b = t.bn(xirow, xi);
    Stat& st = stats[b];
    st.n  += 1;
    st.sy += y[i];
  }

  const double s2 = sigma*sigma;
  if (!std::isfinite(s2) || s2 <= 0.0) return std::numeric_limits<double>::quiet_NaN();

  double ll = 0.0;
  const double tau2 = tau*tau;
  for (const auto &kv : stats) {
    const size_t ni = kv.second.n;
    const double sy = kv.second.sy;
    // −0.5 log(1 + n τ^2/σ^2) + 0.5 * (τ^2 * sy^2)/(σ^2 * (σ^2 + n τ^2))
    // Derivation: integrate out mu with N(0,τ^2); keep only terms depending on partition
    const double a = 1.0 + ((double)ni) * tau2 / s2;
    ll += -0.5*std::log(a) + 0.5 * (tau2 * sy * sy) / ( s2 * (s2 + (double)ni * tau2) );
  }
  return ll;
}

// Heteroskedastic case: sigma_i known per observation; prior mu ~ N(0, tau^2)
// We only need b = sum 1/sigma_i^2 and M = sum y_i/sigma_i^2 per leaf.
// log p(y|partition) up to constant: -0.5 log(1 + τ^2 b) + 0.5 τ^2 M^2/(1 + τ^2 b)
inline double tree_log_marglik_hetero(tree& t, xinfo& xi, dinfo* di_ptr, const double *sigma, double tau)
{
  const size_t p = di_ptr->p;
  const size_t n = di_ptr->n;
  double* x = di_ptr->x;
  double* y = di_ptr->y;

  struct StatHM { double b = 0.0; double M = 0.0; };
  std::unordered_map<const tree*, StatHM> stats;
  const size_t nb = t.nbots();
  stats.reserve(nb * 2 + 1);

  for (size_t i=0; i<n; ++i) {
    double* xirow = x + p*i;
    tree::tree_cp bnode = t.bn(xirow, xi);
    const double s2i = sigma[i]*sigma[i];
    if (!std::isfinite(s2i) || s2i <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    StatHM& st = stats[bnode];
    st.b += 1.0 / s2i;
    st.M += y[i] / s2i;
  }

  double ll = 0.0;
  const double tau2 = tau*tau;
  for (const auto &kv : stats) {
    const double bsum = kv.second.b;
    const double Msum = kv.second.M;
    const double a = 1.0 + tau2 * bsum;
    ll += -0.5*std::log(a) + 0.5 * (tau2 * Msum * Msum) / a;
  }
  return ll;
}

// -----------------------------------------------------------------------------
// Faster variants using precomputed leaf assignments (bn) instead of calling
// tree::bn() inside the likelihood.
//
// These are *pure refactors* of tree_log_marglik_{homo,hetero}:
//   - same math
//   - same numerical behavior up to unordered_map iteration order
//   - avoids an extra pass of bn() when bn vectors are already available
// -----------------------------------------------------------------------------

inline double tree_log_marglik_homo_from_bn(const std::vector<tree::tree_cp>& bn,
                                           const double* y,
                                           size_t n,
                                           double sigma,
                                           double tau,
                                           size_t nb_hint = 0)
{
  if (n == 0 || bn.size() < n || y == nullptr) return 0.0;
  const double s2 = sigma * sigma;
  if (!std::isfinite(s2) || s2 <= 0.0) return std::numeric_limits<double>::quiet_NaN();

  struct Stat { size_t n = 0; double sy = 0.0; };
  std::unordered_map<size_t, Stat> stats; // key: leaf nid
  if (nb_hint > 0) stats.reserve(nb_hint * 2 + 1);

  for (size_t i = 0; i < n; ++i) {
    const tree::tree_cp leaf = bn[i];
    if (!leaf) continue;
    const size_t id = leaf->nid();
    Stat& st = stats[id];
    st.n  += 1;
    st.sy += y[i];
  }

  double ll = 0.0;
  const double tau2 = tau * tau;
  for (const auto& kv : stats) {
    const size_t ni = kv.second.n;
    const double sy = kv.second.sy;
    const double a = 1.0 + ((double)ni) * tau2 / s2;
    ll += -0.5 * std::log(a) + 0.5 * (tau2 * sy * sy) / (s2 * (s2 + (double)ni * tau2));
  }
  return ll;
}

inline double tree_log_marglik_hetero_from_bn(const std::vector<tree::tree_cp>& bn,
                                             const double* y,
                                             const double* sigma,
                                             size_t n,
                                             double tau,
                                             size_t nb_hint = 0)
{
  if (n == 0 || bn.size() < n || y == nullptr || sigma == nullptr) return 0.0;

  struct StatHM { double b = 0.0; double M = 0.0; };
  std::unordered_map<size_t, StatHM> stats; // key: leaf nid
  if (nb_hint > 0) stats.reserve(nb_hint * 2 + 1);

  for (size_t i = 0; i < n; ++i) {
    const tree::tree_cp leaf = bn[i];
    if (!leaf) continue;
    const double s2i = sigma[i] * sigma[i];
    if (!std::isfinite(s2i) || s2i <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    const size_t id = leaf->nid();
    StatHM& st = stats[id];
    st.b += 1.0 / s2i;
    st.M += y[i] / s2i;
  }

  double ll = 0.0;
  const double tau2 = tau * tau;
  for (const auto& kv : stats) {
    const double bsum = kv.second.b;
    const double Msum = kv.second.M;
    const double a = 1.0 + tau2 * bsum;
    ll += -0.5 * std::log(a) + 0.5 * (tau2 * Msum * Msum) / a;
  }
  return ll;
}

inline double gaussian_loglik_homo_from_bn_theta(const std::vector<tree::tree_cp>& bn,
                                                 const double* y,
                                                 size_t n,
                                                 double sigma)
{
  if (n == 0 || bn.size() < n || y == nullptr) return 0.0;
  const double s2 = sigma * sigma;
  if (!std::isfinite(s2) || s2 <= 0.0) return std::numeric_limits<double>::quiet_NaN();

  const double log2pi = std::log(2.0 * 3.141592653589793238462643383279502884);
  double ll = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const tree::tree_cp leaf = bn[i];
    if (!leaf) return std::numeric_limits<double>::quiet_NaN();
    const double err = y[i] - leaf->gettheta();
    ll += -0.5 * (log2pi + std::log(s2) + (err * err) / s2);
  }
  return ll;
}

inline double gaussian_loglik_hetero_from_bn_theta(const std::vector<tree::tree_cp>& bn,
                                                   const double* y,
                                                   const double* sigma,
                                                   size_t n)
{
  if (n == 0 || bn.size() < n || y == nullptr || sigma == nullptr) return 0.0;

  const double log2pi = std::log(2.0 * 3.141592653589793238462643383279502884);
  double ll = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const tree::tree_cp leaf = bn[i];
    if (!leaf) return std::numeric_limits<double>::quiet_NaN();
    const double s2i = sigma[i] * sigma[i];
    if (!std::isfinite(s2i) || s2i <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double err = y[i] - leaf->gettheta();
    ll += -0.5 * (log2pi + std::log(s2i) + (err * err) / s2i);
  }
  return ll;
}







// Apply one Givens rotation to coordinates a and b of every observation.
inline void apply_givens_inplace(double* X, size_t p, size_t n,
                                 size_t a, size_t b, double c, double s)
{
  for (size_t i=0; i<n; ++i) {
    double xa = X[a + p*i];
    double xb = X[b + p*i];
    double na =  c*xa - s*xb;
    double nb =  s*xa + c*xb;
    X[a + p*i] = na;
    X[b + p*i] = nb;
  }
}

// Left-multiply R by G(a,b,theta), which mixes only rows a and b.
// No dense p-by-p Givens matrix is formed.
// Updating the two affected rows takes O(p) operations.
inline void apply_givens_left_inplace(Eigen::MatrixXd& R,
                                      size_t a, size_t b,
                                      double c, double s)
{
  const int A = (int)a;
  const int B = (int)b;
  const int P = (int)R.cols();
  for (int j = 0; j < P; ++j) {
    const double ra = R(A, j);
    const double rb = R(B, j);
    R(A, j) =  c * ra - s * rb;
    R(B, j) =  s * ra + c * rb;
  }
}

// Mark every coordinate touched by the rotation sequence.
inline std::vector<char> touched_from_seq(size_t p, const std::vector<RotState::Pending>& seq)
{
  std::vector<char> tch(p, 0);
  for (const auto& u : seq) { tch[u.a] = 1; tch[u.b] = 1; }
  return tch;
}

// Diagnostic percentile matching on all touched coordinates; exact membership is not guaranteed.
inline void retune_cutpoints_percentile_match_multi(tree& tr,
                                                    xinfo& xi,
                                                    size_t p, size_t n,
                                                    const double* X_old,
                                                    const double* X_new,
                                                    const std::vector<char>& touched)
{
  // 1. Cache routing to terminal nodes under the old state.
  std::vector<tree::tree_cp> bn_old;
  precompute_bottom_nodes(tr, xi, p, n, const_cast<double*>(X_old), bn_old);

  // 2. Visit internal nodes using depth-first traversal.
  std::vector<tree::tree_p> stack;
  stack.push_back(&tr);
  while (!stack.empty()) {
    tree::tree_p nd = stack.back(); stack.pop_back();
    if (!nd || !nd->getl()) continue; // bottom skip

    size_t v = nd->getv();
    // For untouched coordinates, continue to the children without retuning.
    if (!touched.empty() && !touched[v]) {
      stack.push_back(nd->getl());
      stack.push_back(nd->getr());
      continue;
    }

    // Collect observations visiting this node and their old left/right labels.
    std::vector<size_t> idx; idx.reserve(n);
    std::vector<int>    lab; lab.reserve(n);
    for (size_t i=0; i<n; ++i) {
      tree::tree_cp btm = bn_old[i];
      if (is_descendant(btm, nd)) {
        int is_left = is_descendant(btm, nd->getl()) ? 1 : 0;
        idx.push_back(i); lab.push_back(is_left);
      }
    }
    if (!idx.empty()) {
      size_t Lcnt = 0; for (int z : lab) if (z==1) ++Lcnt;

      // Collect this node's split-coordinate values under the new design.
      std::vector<double> vals; vals.reserve(idx.size());
      for (size_t i : idx) vals.push_back(X_new[v + p*i]);

      // Select an order statistic as the threshold target.
      if (Lcnt > 0 && Lcnt <= vals.size()) {
        size_t kth = Lcnt - 1;
        std::nth_element(vals.begin(), vals.begin()+kth, vals.end());
        double thr = vals[kth];

        int L=0, U=(int)xi[v].size()-1;
        nd->rg(v, &L, &U);
        if (U >= L && U >= 0) {
          size_t cidx = nearest_index_in_range(xi[v], L, U, thr);
          nd->setc((size_t)cidx);
        }
      }
    }
    // Continue to the children.
    stack.push_back(nd->getl());
    stack.push_back(nd->getr());
  }
}


// === Cutpoint snapshots (avoid deep-copying `tree`) ============================
//
// In rotation-MH we *only* change cutpoints (c) on existing internal nodes.
// Copying `tree` (which recursively allocates nodes) is expensive and is done
// once-per-tree-per-iteration in the original prototype.  Instead, we snapshot
//
// Note: we intentionally snapshot **only internal nodes** (nd->getl()!=nullptr).
struct CutSnapshot {
  std::vector<tree::tree_p> nodes; // internal-node pointers
  std::vector<size_t>       cuts;  // parallel to nodes
};

inline void snapshot_internal_cuts(tree& tr, CutSnapshot& snap)
{
  tree::npv all;
  tr.getnodes(all);
  snap.nodes.clear();
  snap.cuts.clear();
  snap.nodes.reserve(all.size());
  snap.cuts.reserve(all.size());
  for (auto nd : all) {
    if (!nd || !nd->getl()) continue;
    snap.nodes.push_back(nd);
    snap.cuts.push_back(nd->getc());
  }
}

inline void capture_cuts_from_nodes(const std::vector<tree::tree_p>& nodes, std::vector<size_t>& cuts)
{
  cuts.resize(nodes.size());
  for (size_t i=0; i<nodes.size(); ++i) cuts[i] = nodes[i]->getc();
}

inline void restore_internal_cuts(const CutSnapshot& snap)
{
  for (size_t i=0; i<snap.nodes.size(); ++i) snap.nodes[i]->setc(snap.cuts[i]);
}

// === Path-based aggregation for percentile centers ============================
//
// The naive implementation scanned all observations for *each* internal node
// (O(n * #internal_nodes) descendant checks).  Here we aggregate in one (or two)
// passes over observations by walking each observation's leaf-to-root path.
// Complexity becomes O(n * depth), which is typically much smaller.
//
// For each internal node, we need:
//   - vals: X_vals[v] for all obs that pass through the node (under a fixed routing)
//   - Lcnt: how many of those obs go to the LEFT child under that routing
struct NodeBag {
  size_t Lcnt = 0;
  std::vector<double> vals;
};

inline void collect_node_bags_from_leaves(
  const std::vector<tree::tree_cp>& leaf_of_obs,
  size_t p,
  size_t n,
  const double* X_vals,               // values to collect (X_new or X_target)
  const std::vector<char>& touched,   // variable filter; if empty -> treat as all touched
  std::unordered_map<const tree*, NodeBag>& out)
{
  out.clear();
  if (n == 0) return;

  // 1st pass: counts per node (for reserve)
  std::unordered_map<const tree*, size_t> counts;
  counts.reserve(64);

  for (size_t i=0; i<n; ++i) {
    const tree* cur = leaf_of_obs[i];
    if (!cur) continue;
    const tree* par = cur->getp();
    while (par) {
      if (par->getl()) { // internal
        const size_t v = par->getv();
        const bool ok = touched.empty() || (v < touched.size() && touched[v]);
        if (ok) counts[par] += 1;
      }
      cur = par;
      par = cur->getp();
    }
  }

  out.reserve(counts.size() * 2 + 1);
  for (const auto& kv : counts) {
    NodeBag b;
    b.Lcnt = 0;
    b.vals.reserve(kv.second);
    out.emplace(kv.first, std::move(b));
  }

  // 2nd pass: fill vals + Lcnt
  for (size_t i=0; i<n; ++i) {
    const tree* cur = leaf_of_obs[i];
    if (!cur) continue;
    const tree* par = cur->getp();
    while (par) {
      if (par->getl()) {
        const size_t v = par->getv();
        const bool ok = touched.empty() || (v < touched.size() && touched[v]);
        if (ok) {
          auto it = out.find(par);
          if (it != out.end()) {
            it->second.vals.push_back(X_vals[v + p*i]);
            if (cur == par->getl()) it->second.Lcnt += 1;
          }
        }
      }
      cur = par;
      par = cur->getp();
    }
  }
}

// === Diagnostics using cut snapshots (no tree copies) =========================
inline void compute_partition_diagnostics_cut_snapshots(
  tree& tr,
  const CutSnapshot& snap_old,
  const CutSnapshot& snap_new,
  xinfo& xi,
  const double* X_old,
  const double* X_new,
  const double* y, // residuals
  size_t p, size_t n,
  double& out_memb_change,
  double& out_leaf_n_L1,
  double& out_leaf_sy_L1)
{
  out_memb_change = NA_REAL;
  out_leaf_n_L1   = NA_REAL;
  out_leaf_sy_L1  = NA_REAL;
  if (n == 0) return;

  // Save current state (assume "new" in our callers) and restore at exit.
  // (snap_new is what we want to leave the tree in.)
  restore_internal_cuts(snap_old);

  struct LeafStat { size_t n = 0; double sy = 0.0; };
  std::unordered_map<size_t, LeafStat> old_stats;
  old_stats.reserve(64);

  std::vector<size_t> old_leaf_id(n);
  double sum_abs = 0.0;

  // Old routing/stats under X_old
  for (size_t i=0; i<n; ++i) {
    const tree::tree_cp b = tr.bn(const_cast<double*>(X_old) + p*i, xi);
    const size_t id = b->nid();
    old_leaf_id[i] = id;
    LeafStat& st = old_stats[id];
    st.n  += 1;
    st.sy += y[i];
    sum_abs += std::fabs(y[i]);
  }

  // New routing/stats under X_new
  restore_internal_cuts(snap_new);

  std::unordered_map<size_t, LeafStat> new_stats;
  new_stats.reserve(64);

  size_t changed = 0;
  for (size_t i=0; i<n; ++i) {
    const tree::tree_cp b = tr.bn(const_cast<double*>(X_new) + p*i, xi);
    const size_t id = b->nid();
    if (id != old_leaf_id[i]) changed += 1;
    LeafStat& st = new_stats[id];
    st.n  += 1;
    st.sy += y[i];
  }

  out_memb_change = (double)changed / (double)n;

  // L1 distances between old/new leaf count vectors and leaf residual-sum vectors
  double l1n = 0.0;
  double l1sy = 0.0;

  std::unordered_set<size_t> seen;
  seen.reserve(old_stats.size() * 2 + 1);

  for (const auto& kv : old_stats) {
    const size_t id = kv.first;
    const LeafStat& so = kv.second;
    const auto it = new_stats.find(id);
    const size_t nn = (it == new_stats.end() ? 0 : it->second.n);
    const double syn = (it == new_stats.end() ? 0.0 : it->second.sy);
    l1n  += std::fabs((double)nn - (double)so.n);
    l1sy += std::fabs(syn - so.sy);
    seen.insert(id);
  }
  for (const auto& kv : new_stats) {
    const size_t id = kv.first;
    if (seen.find(id) != seen.end()) continue;
    const LeafStat& sn = kv.second;
    l1n  += (double)sn.n;
    l1sy += std::fabs(sn.sy);
  }

  out_leaf_n_L1 = l1n / (2.0 * (double)n);
  const double denom = 2.0 * (sum_abs + 1e-12);
  out_leaf_sy_L1 = l1sy / denom;

  // Ensure we leave the tree in the "new" state
  restore_internal_cuts(snap_new);
}


// One cutpoint law is used for generation and evaluation. The source bags
// are frozen before changing any ancestor. Every internal node is checked,
// including untouched labels and deterministic holds. Failure has its original
// mass and is never retried until success.
struct CutRestoreGuard {
  CutSnapshot saved;
  bool restore=true;
  explicit CutRestoreGuard(tree& tr) { snapshot_internal_cuts(tr,saved); }
  ~CutRestoreGuard() { if(restore)restore_internal_cuts(saved); }
};
inline double finite_midpoint(double lo,double hi) {
  // Adjacent order statistics are finite. Avoid overflow for opposite signs.
  return std::signbit(lo)==std::signbit(hi)?lo+0.5*(hi-lo):0.5*lo+0.5*hi;
}
inline int midpoint_center(std::vector<double>& vals,size_t left,
                           const std::vector<double>& grid,const formal::Interval& a) {
  if(!left || left>=vals.size() || a.empty())
    throw std::invalid_argument("Midpoint requires observations in both source children.");
  std::nth_element(vals.begin(),vals.begin()+left-1,vals.end());
  const double lo=vals[left-1],hi=*std::min_element(vals.begin()+left,vals.end());
  const double mid=finite_midpoint(lo,hi);
  // The source order statistics and midpoint are unchanged. Distances are
  // monotone on either side of lower_bound(mid), including rounding plateaus.
  const auto first=grid.begin()+a.lo,last=grid.begin()+a.hi+1;
  const int right=static_cast<int>(std::lower_bound(first,last,mid)-grid.begin());
  const auto distance=[&](int c) -> long double {
    return std::fabs(static_cast<long double>(grid[c])-static_cast<long double>(mid));
  };
  long double best=std::numeric_limits<long double>::infinity();
  if(right<=a.hi)best=distance(right);
  if(right>a.lo)best=std::min(best,distance(right-1));
  if(!std::isfinite(best)) {
    // On platforms with double-width long double, distance may overflow.
    // Retain the original first-index rule even for this exceptional case.
    int answer=a.lo;long double current=std::numeric_limits<long double>::infinity();
    for(int c=a.lo;c<=a.hi;++c)if(distance(c)<current){current=distance(c);answer=c;}
    return answer;
  }
  if(right>a.lo && distance(right-1)==best) {
    int lo_index=a.lo,hi_index=right-1;
    while(lo_index<hi_index) {
      const int k=lo_index+(hi_index-lo_index)/2;
      if(distance(k)>best)lo_index=k+1;else hi_index=k;
    }
    return lo_index; // first minimum, not the last duplicate left of midpoint
  }
  return right;
}
inline double sequential_cutpoint_rule(tree& tr,const formal::Context& destination,
       size_t p,size_t n,const std::vector<tree::tree_cp>& source_leaves,
       const double* destination_X,const std::vector<char>& touched,
       rn* generator,const CutSnapshot* target=nullptr) {
  CutRestoreGuard guard(tr);
  if(source_leaves.size()!=n || (n && !destination_X) || ((!generator)==(!target)))return formal::neginf();
  std::unordered_map<const tree*,NodeBag> bags;
  collect_node_bags_from_leaves(source_leaves,p,n,destination_X,touched,bags);
  std::unordered_map<const tree*,size_t> desired;
  if(target) {
    if(target->nodes.size()!=target->cuts.size() || target->nodes.size()!=guard.saved.nodes.size())return formal::neginf();
    for(size_t k=0;k<target->nodes.size();++k)desired[target->nodes[k]]=target->cuts[k];
  }
  double logq=0.;
  // getnodes is a deterministic root-left-right preorder in both directions.
  for(size_t k=0;k<guard.saved.nodes.size();++k) {
    tree* node=guard.saved.nodes[k];const size_t v=node->getv();
    const formal::Interval a=destination.allowed(node,v);
    if(a.empty())return formal::neginf();
    const size_t original=guard.saved.cuts[k];
    const auto it=bags.find(node);
    const bool affected=touched.empty() || (v<touched.size()&&touched[v]);
    const bool random=affected && it!=bags.end() && it->second.Lcnt>0 && it->second.Lcnt<it->second.vals.size();
    size_t chosen=original;
    if(target) {
      const auto jt=desired.find(node);if(jt==desired.end())return formal::neginf();
      chosen=jt->second;
    }
    if(!random) {
      if(!a.contains(original) || (target&&chosen!=original))return formal::neginf();
    } else {
      NodeBag& bag=it->second;
      const int center=midpoint_center(bag.vals,bag.Lcnt,destination.grid[v],a);
      double term=0.;
      if(generator)chosen=static_cast<size_t>(sample_cutpoint_index(a.lo,a.hi,center,*generator,&term));
      else {
        if(!a.contains(chosen))return formal::neginf();
        term=logq_cutpoint_index(static_cast<int>(chosen),a.lo,a.hi,center);
      }
      if(!std::isfinite(term))return formal::neginf();
      logq+=term;
    }
    node->setc(chosen);
  }
  guard.restore=(target!=nullptr); // evaluation restores its source, generation commits only on success
  return logq;
}
inline double propose_cutpoints_percentile_multi(tree& tr,const formal::Context& ctx,size_t p,size_t n,
       const std::vector<tree::tree_cp>& bn_old,const double* X_new,
       const std::vector<char>& touched,rn& gen) {
  return sequential_cutpoint_rule(tr,ctx,p,n,bn_old,X_new,touched,&gen);
}
inline double propose_cutpoints_percentile_multi(tree& tr,xinfo& xi,size_t p,size_t n,
       const std::vector<tree::tree_cp>& bn_old,const double* X_new,
       const std::vector<char>& touched,rn& gen) {
  const formal::Context ctx(xi,X_new,p,n);
  return propose_cutpoints_percentile_multi(tr,ctx,p,n,bn_old,X_new,touched,gen);
}
inline double backward_logq_percentile_multi_inplace(tree& tr,const CutSnapshot& snap_old,
       const CutSnapshot& snap_prop,const formal::Context& ctx,size_t p,size_t n,
       const std::vector<tree::tree_cp>& bn_ref,const double* X_target,const std::vector<char>& touched) {
  const double ans=sequential_cutpoint_rule(tr,ctx,p,n,bn_ref,X_target,touched,nullptr,&snap_old);
  restore_internal_cuts(snap_prop);
  return ans;
}
inline double backward_logq_percentile_multi_inplace(tree& tr,const CutSnapshot& snap_old,
       const CutSnapshot& snap_prop,xinfo& xi,size_t p,size_t n,
       const std::vector<tree::tree_cp>& bn_ref,const double* X_target,const std::vector<char>& touched) {
  const formal::Context ctx(xi,X_target,p,n);
  return backward_logq_percentile_multi_inplace(tr,snap_old,snap_prop,ctx,p,n,bn_ref,X_target,touched);
}
struct RotationTargetTerms {
  double likelihood_old=NA_REAL,likelihood_new=NA_REAL;
  double prior_old=NA_REAL,prior_new=NA_REAL;
  double logq_forward=NA_REAL,logq_reverse=NA_REAL;
  double logacc=NA_REAL;
};

// ============================================================================
// Centered bounded Givens proposal with joint cutpoint retuning.
// The acceptance uses the full discrete prior and collapsed likelihood.
// Forward/reverse cutpoint probabilities use the same single-scale index law.
// Centered angles and single-scale cutpoint proposals share one joint MH step.
// Reverse angle arithmetic remains explicit to preserve floating-point order.
// State-dependent cutpoint proposal probabilities do not cancel in general.
// ============================================================================

inline double clamp_double_(double x, double lo, double hi)
{
  return (x < lo ? lo : (x > hi ? hi : x));
}

inline double logspace_sub_(double log_a, double log_b)
{
  // log(exp(log_a) - exp(log_b)), assuming log_a >= log_b.
  if (log_b == -std::numeric_limits<double>::infinity()) return log_a;
  if (log_a < log_b) std::swap(log_a, log_b);
  const double d = log_b - log_a;
  if (d >= 0.0) return -std::numeric_limits<double>::infinity();
  return log_a + std::log1p(-std::exp(d));
}

inline double log_truncnorm_density_(double x, double mu, double sd,
                                     double lower, double upper)
{
  if (!(sd > 0.0) || !std::isfinite(sd) || upper <= lower) {
    return -std::numeric_limits<double>::infinity();
  }
  if (x < lower || x > upper) return -std::numeric_limits<double>::infinity();

  const double log_hi = R::pnorm(upper, mu, sd, 1, 1);
  const double log_lo = R::pnorm(lower, mu, sd, 1, 1);
  const double log_z  = logspace_sub_(log_hi, log_lo);
  if (!std::isfinite(log_z)) return -std::numeric_limits<double>::infinity();
  return R::dnorm(x, mu, sd, 1) - log_z;
}

inline double sample_truncnorm_(double mu, double sd, double lower, double upper, rn& gen)
{
  if (!(sd > 0.0) || !std::isfinite(sd) || upper <= lower) return 0.0;
  double Flo = R::pnorm(lower, mu, sd, 1, 0);
  double Fhi = R::pnorm(upper, mu, sd, 1, 0);
  if (!std::isfinite(Flo) || !std::isfinite(Fhi) || !(Fhi > Flo)) {
    // Very extreme mean.  Fallback keeps the proposal well-defined in practice;
    // this fallback is retained for exact reference behavior.
    return clamp_double_(mu + sd * gen.normal(), lower, upper);
  }
  double u = Flo + gen.uniform() * (Fhi - Flo);
  const double eps = 1e-14;
  if (u <= eps) u = eps;
  if (u >= 1.0 - eps) u = 1.0 - eps;
  return R::qnorm(u, mu, sd, 1, 0);
}

inline size_t xinfo_count_for_var_(size_t v,
                                   const std::vector<int>& nc_per_var,
                                   const xinfo& xi)
{
  if (v < nc_per_var.size() && nc_per_var[v] > 0)
    return static_cast<size_t>(nc_per_var[v]);
  if (v < xi.size()) return xi[v].size();
  return 0;
}

// Read only the proposed rotated training design, stored as X[v + p*i].
// Uniform grids retain constant-coordinate duplicates. Quantile grids use
// floor-index order statistics without interpolation or deduplication.
inline void rebuild_xinfo_one_column_(const double* X,
                                      size_t p,
                                      size_t n,
                                      size_t v,
                                      const std::vector<int>& nc_per_var,
                                      XInfoPolicy policy,
                                      xinfo& xi,
                                      std::vector<unsigned char>* validated_active=nullptr)
{
  if (X == nullptr || p == 0 || n == 0 || v >= p) return;
  if (xi.size() != p) xi.resize(p);

  const size_t nc = xinfo_count_for_var_(v, nc_per_var, xi);
  xi[v].assign(nc, 0.0);
  if (validated_active && (validated_active->size()!=p || policy!=XInfoPolicy::UniformFixedNC))
    throw std::logic_error("Fused support requires a uniform grid and full support vector.");
  if (nc == 0) {
    if(validated_active) {
      double lo=X[v],hi=lo;
      for(size_t i=0;i<n;++i) {
        const double z=X[v+p*i];
        if(!std::isfinite(z))throw std::invalid_argument("Formal tree support received nonfinite coordinates.");
        lo=std::min(lo,z);hi=std::max(hi,z);
      }
      (*validated_active)[v]=(lo<hi);
    }
    return;
  }

  if (policy == XInfoPolicy::GlobalCoveringUniform) {
    const double radius = global_covering_radius_from_design(X,p,n);
    const double inc = 2.0*radius/static_cast<double>(nc+1);
    for (size_t k=0;k<nc;++k)
      xi[v][k] = -radius + static_cast<double>(k+1)*inc;
  } else if (policy == XInfoPolicy::QuantileFixedNC) {
    std::vector<double> vals(n);
    for (size_t i = 0; i < n; ++i) vals[i] = X[v + p*i];
    std::sort(vals.begin(), vals.end());
    const size_t last = vals.size() - 1;
    for (size_t k = 0; k < nc; ++k) {
      const double q = static_cast<double>(k + 1) /
                       static_cast<double>(nc + 1);
      size_t idx = static_cast<size_t>(std::floor(q * static_cast<double>(last)));
      if (idx > last) idx = last;
      xi[v][k] = vals[idx];
    }
  } else {
    double minv = X[v];
    double maxv = X[v];
    if(validated_active && !std::isfinite(minv))
      throw std::invalid_argument("Formal tree support received nonfinite coordinates.");
    for (size_t i = 1; i < n; ++i) {
      const double z = X[v + p*i];
      if(validated_active && !std::isfinite(z))
        throw std::invalid_argument("Formal tree support received nonfinite coordinates.");
      if (z < minv) minv = z;
      if (z > maxv) maxv = z;
    }
    if(validated_active)(*validated_active)[v]=(minv<maxv);
    const double inc = (maxv - minv) / static_cast<double>(nc + 1);
    for (size_t k = 0; k < nc; ++k)
      xi[v][k] = minv + static_cast<double>(k + 1) * inc;
  }
}

// This mutates a proposal grid, not the accepted grid. The caller commits it
// only on acceptance; a rejection leaves the current grid and tree unchanged.
inline void rebuild_xinfo_touched_(const double* X,
                                   size_t p,
                                   size_t n,
                                   const std::vector<char>& touched,
                                   const std::vector<int>& nc_per_var,
                                   XInfoPolicy policy,
                                   xinfo& xi,
                                   std::vector<unsigned char>* validated_active=nullptr)
{
  if (xi.size() != p) xi.resize(p);
  for (size_t v = 0; v < p; ++v) {
    if (!touched.empty() && (v >= touched.size() || !touched[v])) continue;
    rebuild_xinfo_one_column_(X, p, n, v, nc_per_var, policy, xi, validated_active);
  }
}

inline bool rotation_centered_update_tree(tree& tr,
                                      xinfo& xi_cur,
                                      dinfo*& di_ptr,
                                      pinfo& pi,
                                      double sigma,
                                      double rotation_step_size,
                                      double theta_max,
                                      size_t num_planes,
                                      bool retune_cutpoints,
                                      rn& gen,
                                      RotState& st,
                                      size_t p, size_t n,
                                      bool dynamic_xinfo_on_rotation=false,
                                      const std::vector<int>& nc_per_var = std::vector<int>(),
                                      XInfoPolicy xinfo_policy = XInfoPolicy::UniformFixedNC,
                                      size_t np=0, dinfo* di_test=nullptr,
                                      double* out_logacc=nullptr,
                                      double* out_ll_current=nullptr,
                                      double* out_ll_proposed=nullptr,
                                      double* out_global_ll_current=nullptr,
                                      double* out_global_ll_proposed=nullptr,
                                      double* out_memb_change=nullptr,
                                      double* out_leaf_n_L1=nullptr,
                                      double* out_leaf_sy_L1=nullptr,

                                      const std::vector<size_t>* cand_vars=nullptr,
                                      double support_expand_prob=0.0,
                                      bool compute_diagnostics=false,
                                      bool accumulate_angles=false,
                                      RotTiming* out_timing=nullptr,
                                      const DepthPriorSpec& depth_prior=DepthPriorSpec(),
                                      const std::vector<double>* log_weights=nullptr,
                                      RotationTargetTerms* target_terms=nullptr,
                                      formal::TreeCache* state_cache=nullptr)
{
  if(target_terms)*target_terms=RotationTargetTerms{};
  const bool do_time = (out_timing != nullptr);
  std::chrono::steady_clock::time_point t_total0, t0;
  if (do_time) {
    *out_timing = RotTiming{};
    t_total0 = std::chrono::steady_clock::now();
    t0 = t_total0;
  }

  auto finish_reject = [&](const CutSnapshot* snap_old, double* x_saved) -> bool {
    if (snap_old) restore_internal_cuts(*snap_old);
    if (x_saved) di_ptr->x = x_saved;
    if (do_time) {
      const auto t1 = std::chrono::steady_clock::now();
      out_timing->t_total_ms = _ms_between_(t_total0, t1);
    }
    return false;
  };

  if (p < 2 || n == 0) {
    if (out_logacc) *out_logacc = 0.0;
    if (out_ll_current) *out_ll_current = NA_REAL;
    if (out_ll_proposed) *out_ll_proposed = NA_REAL;
    return false;
  }

  if (dynamic_xinfo_on_rotation &&
      xinfo_policy == XInfoPolicy::GlobalCoveringUniform) {
    if (out_logacc) *out_logacc = 0.0;
    if (out_ll_current) *out_ll_current = NA_REAL;
    if (out_ll_proposed) *out_ll_proposed = NA_REAL;
    return false;
  }

  if (!(rotation_step_size > 0.0) || !std::isfinite(rotation_step_size)) rotation_step_size = 0.05;
  if (!(theta_max > 0.0) || !std::isfinite(theta_max)) theta_max = std::min(0.5, 4.0 * rotation_step_size);
  theta_max = std::min(theta_max, 3.141592653589793238462643383279502884 - 1e-8);
  if (num_planes == 0) num_planes = 1;

  if (st.R.size()==0) st.R = Eigen::MatrixXd::Identity((int)p,(int)p);
  if ((size_t)st.Xtr.size() != p*n) {
    st.Xtr.assign(p*n, 0.0);
    std::memcpy(st.Xtr.data(), di_ptr->x, sizeof(double)*p*n);
  }


  const bool canonical=st.canonical_coordinates;
  if(canonical && (st.raw_p!=p || st.raw_n!=n || !st.raw_train))
    throw std::invalid_argument("Canonical training design registration mismatch.");
  if(xinfo_policy==XInfoPolicy::QuantileFixedNC && !canonical)
    throw std::invalid_argument("Quantile rotation requires canonical input registration.");
  double* Xold_ptr = st.Xtr.data();
  const double* yptr = di_ptr->y;
  const formal::Context old_context=state_cache
    ?state_cache->context(xi_cur,Xold_ptr,p,n,log_weights)
    :formal::Context(xi_cur,Xold_ptr,p,n,log_weights);
  const formal::PriorTerms old_prior=state_cache
    ?state_cache->get(tr,old_context,pi,depth_prior)
    :formal::prior(tr,old_context,pi,depth_prior);
  const double lp_old=old_prior.total();
  if(!std::isfinite(lp_old))throw std::runtime_error("Rotation started from an invalid formal tree state.");
  if(target_terms)target_terms->prior_old=lp_old;

  // Draw centered truncated-normal angles sequentially on an ordered path.
  // Dynamic grids follow the selected endpoint policy; fixed grids stay unchanged.
  // Include forward and reverse cutpoint probabilities in the joint MH ratio.
  static thread_local std::vector<double> Xprop_scratch;
  const size_t pn = p * n;
  if (Xprop_scratch.size() != pn) Xprop_scratch.resize(pn);
  std::memcpy(Xprop_scratch.data(), Xold_ptr, sizeof(double) * pn);
  double* Xprop_ptr = Xprop_scratch.data();

  std::vector<RotState::Pending> seq;
  seq.reserve(num_planes);
  std::vector<double> dtheta_seq;
  dtheta_seq.reserve(num_planes);

  Eigen::MatrixXd Qprop;
  if(canonical)Qprop=st.R;
  double logq_theta_fwd = 0.0;
  for (size_t k = 0; k < num_planes; ++k) {
    size_t a, b;
    sample_rotation_pair(p, gen, cand_vars, a, b, support_expand_prob);

    const double mu_fwd = 0.0;
    const double dtheta = sample_truncnorm_(mu_fwd, rotation_step_size, -theta_max, theta_max, gen);
    logq_theta_fwd += log_truncnorm_density_(dtheta, mu_fwd, rotation_step_size,
                                             -theta_max, theta_max);

    RotState::Pending u{a, b, std::cos(dtheta), std::sin(dtheta)};
    seq.push_back(u);
    dtheta_seq.push_back(dtheta);
    if(canonical)apply_givens_left_inplace(Qprop,u.a,u.b,u.c,u.s);
    else apply_givens_inplace(Xprop_ptr, p, n, u.a, u.b, u.c, u.s);

  }

  std::vector<char> touched = touched_from_seq(p, seq);
  // Score the endpoint using a coordinate function of Q, not accumulated X.
  // Untouched rows of Q, X and grid remain identical.
  if(canonical)canonical_project(Qprop,st.raw_train,p,n,Xprop_ptr,&touched);

  xinfo xi_prop;
  xinfo* xi_fwd_ptr = &xi_cur;
  std::vector<unsigned char> fused_active;
  std::vector<unsigned char>* fused_ptr=nullptr;
  if(state_cache && dynamic_xinfo_on_rotation && xinfo_policy==XInfoPolicy::UniformFixedNC) {
    fused_active=state_cache->geometry.active;fused_ptr=&fused_active;
  }
  if (dynamic_xinfo_on_rotation) {
    xi_prop = xi_cur;
    rebuild_xinfo_touched_(Xprop_ptr, p, n, touched, nc_per_var,
                           xinfo_policy, xi_prop, fused_ptr);
    xi_fwd_ptr = &xi_prop;
  }
  xinfo& xi_fwd = *xi_fwd_ptr;
  formal::GeometryCache proposal_geometry;
  if(state_cache)proposal_geometry=state_cache->geometry.proposal(xi_fwd,Xprop_ptr,touched,fused_ptr);
  const formal::Context new_context=state_cache
    ?formal::Context(xi_fwd,proposal_geometry.active,log_weights,formal::ValidatedGeometryTag{})
    :formal::Context(xi_fwd,Xprop_ptr,p,n,log_weights);

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_seq_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_xprop_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  CutSnapshot snap_old;
  snapshot_internal_cuts(tr, snap_old);
  const size_t nb_hint = tr.nbots();

  static thread_local std::vector<tree::tree_cp> bn_old_scratch;
  bn_old_scratch.resize(n);
  for (size_t i = 0; i < n; ++i) bn_old_scratch[i] = tr.bn(Xold_ptr + p*i, xi_cur);
  const double ll_old = tree_log_marglik_homo_from_bn(bn_old_scratch, yptr, n, sigma, pi.tau, nb_hint);

  double logq_fwd = 0.0;
  CutSnapshot snap_prop;
  snap_prop.nodes = snap_old.nodes;

  if (retune_cutpoints) {
    logq_fwd = propose_cutpoints_percentile_multi(
      tr, new_context, p, n, bn_old_scratch, Xprop_ptr, touched, gen
    );
    capture_cuts_from_nodes(snap_prop.nodes, snap_prop.cuts);
  } else {
    // Legacy no-retune proposal: C' = C deterministically.
    // The cutpoint proposal ratio is exactly 1, so log q_C terms are zero.
    snap_prop.cuts = snap_old.cuts;
  }

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_cut_fwd_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  double* x_saved = di_ptr->x;
  const formal::PriorTerms new_prior=formal::prior(tr,new_context,pi,depth_prior);
  const double lp_new=new_prior.total();
  if(target_terms){target_terms->prior_new=lp_new;target_terms->logq_forward=logq_fwd;target_terms->likelihood_old=ll_old;}
  if(!std::isfinite(logq_fwd)||!std::isfinite(lp_new)) {
    if(out_logacc)*out_logacc=formal::neginf();
    if(target_terms)target_terms->logacc=formal::neginf();
    return finish_reject(&snap_old,x_saved);
  }
  static thread_local std::vector<tree::tree_cp> bn_new_scratch;
  bn_new_scratch.resize(n);
  for (size_t i = 0; i < n; ++i) bn_new_scratch[i] = tr.bn(Xprop_ptr + p*i, xi_fwd);
  const double ll_new = tree_log_marglik_homo_from_bn(bn_new_scratch, yptr, n, sigma, pi.tau, nb_hint);

  if (out_ll_current)  *out_ll_current  = ll_old;
  if (out_ll_proposed) *out_ll_proposed = ll_new;
  if (out_global_ll_current)  *out_global_ll_current  = gaussian_loglik_homo_from_bn_theta(bn_old_scratch, yptr, n, sigma);
  if (out_global_ll_proposed) *out_global_ll_proposed = gaussian_loglik_homo_from_bn_theta(bn_new_scratch, yptr, n, sigma);

  if (compute_diagnostics && (out_memb_change || out_leaf_n_L1 || out_leaf_sy_L1)) {
    double mc, ln, lsy;
    compute_partition_diagnostics_from_bn(bn_old_scratch, bn_new_scratch, yptr, n, mc, ln, lsy);
    if (out_memb_change) *out_memb_change = mc;
    if (out_leaf_n_L1)   *out_leaf_n_L1   = ln;
    if (out_leaf_sy_L1)  *out_leaf_sy_L1  = lsy;
  } else {
    if (out_memb_change) *out_memb_change = NA_REAL;
    if (out_leaf_n_L1)   *out_leaf_n_L1   = NA_REAL;
    if (out_leaf_sy_L1)  *out_leaf_sy_L1  = NA_REAL;
  }

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_diag_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  // Reverse cutpoint and reverse angle proposal densities at proposed state.
  double logq_bwd = 0.0;
  if (retune_cutpoints) {
    logq_bwd = backward_logq_percentile_multi_inplace(
      tr, snap_old, snap_prop, old_context, p, n, bn_new_scratch, Xold_ptr, touched
    );
  }
  // If retune_cutpoints == false, C' = C deterministically in both directions,
  // hence log q_C(C | y) - log q_C(C' | x) = 0.

  // Reverse angle density.  Starting at the proposed state (R', C'), the
  // reverse proposal must draw the same planes in reverse order and angles
  // -dtheta_K,...,-dtheta_1. Reverse arithmetic is kept in the same order
  // as in the reference zero-drift kernel for finite-precision equivalence.
  double logq_theta_bwd = 0.0;
  for (size_t rr = seq.size(); rr-- > 0; ) {
    const double dtheta = dtheta_seq[rr];
    const double mu_bwd = 0.0;
    logq_theta_bwd += log_truncnorm_density_(-dtheta, mu_bwd, rotation_step_size,
                                             -theta_max, theta_max);
  }

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_cut_bwd_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  const double logacc = (ll_new - ll_old) + (lp_new - lp_old)
                      + (logq_bwd - logq_fwd)
                      + (logq_theta_bwd - logq_theta_fwd);
  if(target_terms){target_terms->likelihood_new=ll_new;target_terms->logq_reverse=logq_bwd;target_terms->logacc=logacc;}

  if (!std::isfinite(ll_old) || !std::isfinite(ll_new) ||
      !std::isfinite(logq_fwd) || !std::isfinite(logq_bwd) ||
      !std::isfinite(logq_theta_fwd) || !std::isfinite(logq_theta_bwd) ||
      !std::isfinite(logacc)) {
    // A zero reverse mass or invalid target is a rejection, not log ratio zero.
    if (out_logacc) *out_logacc = formal::neginf();
    return finish_reject(&snap_old, x_saved);
  }
  if (out_logacc) *out_logacc = logacc;

  const bool accept = (std::log(gen.uniform()) < logacc);
  if (accept) {
    if(canonical)st.R=Qprop;
    else for (const auto& u : seq) apply_givens_left_inplace(st.R, u.a, u.b, u.c, u.s);
    std::memcpy(st.Xtr.data(), Xprop_ptr, sizeof(double) * pn);
    di_ptr->x = st.Xtr.data();
    if (dynamic_xinfo_on_rotation) xi_cur.swap(xi_prop);
    if(state_cache) {
      state_cache->commit_geometry(proposal_geometry,st.Xtr.data());
      // Use accepted grid: new_context's proposal reference may now be swapped.
      const formal::Context accepted_context(xi_cur,state_cache->geometry.active,
                                             log_weights,formal::ValidatedGeometryTag{});
      state_cache->remember(tr,accepted_context,pi,depth_prior,new_prior);
    }

    if (np && di_test) {
      st.has_test = true;
      if ((size_t)st.Xte.size() != p*np) {
        st.Xte.assign(p*np, 0.0);
        std::memcpy(st.Xte.data(), di_test->x, sizeof(double)*p*np);
      }
      if(canonical)st.canonical_test_dirty=true;
      else for (const auto& u : seq) st.pending.push_back(u);
    }

    if (accumulate_angles) {
      if (st.angle_sum.size()==0 || st.angle_sum.rows()!=(int)p || st.angle_sum.cols()!=(int)p) {
        st.angle_sum = Eigen::MatrixXd::Zero((int)p,(int)p);
        st.angle_count = Eigen::MatrixXi::Zero((int)p,(int)p);
      }
      for (size_t kk = 0; kk < seq.size(); ++kk) {
        const RotState::Pending& u = seq[kk];
        const double theta = dtheta_seq[kk];
        const int ia = (int)u.a;
        const int ib = (int)u.b;
        st.angle_sum(ia, ib) += theta;
        st.angle_sum(ib, ia) -= theta;
        st.angle_count(ia, ib) += 1;
        st.angle_count(ib, ia) += 1;
      }
    }

    if (do_time) {
      const auto t1 = std::chrono::steady_clock::now();
      out_timing->t_accept_ms = _ms_between_(t0, t1);
      out_timing->t_total_ms  = _ms_between_(t_total0, t1);
    }
    return true;
  }

  return finish_reject(&snap_old, x_saved);
}

inline bool heter_rotation_centered_update_tree(tree& tr,
                                            xinfo& xi_cur,
                                            dinfo*& di_ptr,
                                            pinfo& pi,
                                            double* sigma,
                                            double rotation_step_size,
                                            double theta_max,
                                            size_t num_planes,
                                            bool retune_cutpoints,
                                            rn& gen,
                                            RotState& st,
                                            size_t p, size_t n,
                                            bool dynamic_xinfo_on_rotation=false,
                                            const std::vector<int>& nc_per_var = std::vector<int>(),
                                            XInfoPolicy xinfo_policy = XInfoPolicy::UniformFixedNC,
                                            size_t np=0, dinfo* di_test=nullptr,
                                            double* out_logacc=nullptr,
                                            double* out_ll_current=nullptr,
                                            double* out_ll_proposed=nullptr,
                                            double* out_global_ll_current=nullptr,
                                            double* out_global_ll_proposed=nullptr,
                                            double* out_memb_change=nullptr,
                                            double* out_leaf_n_L1=nullptr,
                                            double* out_leaf_sy_L1=nullptr,

                                            const std::vector<size_t>* cand_vars=nullptr,
                                            double support_expand_prob=0.0,
                                            bool compute_diagnostics=false,
                                            bool accumulate_angles=false,
                                            RotTiming* out_timing=nullptr,
                                      const DepthPriorSpec& depth_prior=DepthPriorSpec(),
                                      const std::vector<double>* log_weights=nullptr,
                                      RotationTargetTerms* target_terms=nullptr,
                                      formal::TreeCache* state_cache=nullptr)
{
  if(target_terms)*target_terms=RotationTargetTerms{};
  const bool do_time = (out_timing != nullptr);
  std::chrono::steady_clock::time_point t_total0, t0;
  if (do_time) {
    *out_timing = RotTiming{};
    t_total0 = std::chrono::steady_clock::now();
    t0 = t_total0;
  }

  auto finish_reject = [&](const CutSnapshot* snap_old, double* x_saved) -> bool {
    if (snap_old) restore_internal_cuts(*snap_old);
    if (x_saved) di_ptr->x = x_saved;
    if (do_time) {
      const auto t1 = std::chrono::steady_clock::now();
      out_timing->t_total_ms = _ms_between_(t_total0, t1);
    }
    return false;
  };

  if (p < 2 || n == 0) {
    if (out_logacc) *out_logacc = 0.0;
    if (out_ll_current) *out_ll_current = NA_REAL;
    if (out_ll_proposed) *out_ll_proposed = NA_REAL;
    return false;
  }
  if (dynamic_xinfo_on_rotation &&
      xinfo_policy == XInfoPolicy::GlobalCoveringUniform) {
    if (out_logacc) *out_logacc = 0.0;
    if (out_ll_current) *out_ll_current = NA_REAL;
    if (out_ll_proposed) *out_ll_proposed = NA_REAL;
    return false;
  }

  if (!(rotation_step_size > 0.0) || !std::isfinite(rotation_step_size)) rotation_step_size = 0.05;
  if (!(theta_max > 0.0) || !std::isfinite(theta_max)) theta_max = std::min(0.5, 4.0 * rotation_step_size);
  theta_max = std::min(theta_max, 3.141592653589793238462643383279502884 - 1e-8);
  if (num_planes == 0) num_planes = 1;

  if (st.R.size()==0) st.R = Eigen::MatrixXd::Identity((int)p,(int)p);
  if ((size_t)st.Xtr.size() != p*n) {
    st.Xtr.assign(p*n, 0.0);
    std::memcpy(st.Xtr.data(), di_ptr->x, sizeof(double)*p*n);
  }
  if (np > 0 && di_test) {
    st.has_test = true;
    if ((size_t)st.Xte.size() != p*np) {
      st.Xte.assign(p*np, 0.0);
      std::memcpy(st.Xte.data(), di_test->x, sizeof(double)*p*np);
    }
  }


  const bool canonical=st.canonical_coordinates;
  if(canonical && (st.raw_p!=p || st.raw_n!=n || !st.raw_train))
    throw std::invalid_argument("Canonical training design registration mismatch.");
  if(xinfo_policy==XInfoPolicy::QuantileFixedNC && !canonical)
    throw std::invalid_argument("Quantile rotation requires canonical input registration.");
  double* Xold_ptr = st.Xtr.data();
  const double* yptr = di_ptr->y;
  const formal::Context old_context=state_cache
    ?state_cache->context(xi_cur,Xold_ptr,p,n,log_weights)
    :formal::Context(xi_cur,Xold_ptr,p,n,log_weights);
  const formal::PriorTerms old_prior=state_cache
    ?state_cache->get(tr,old_context,pi,depth_prior)
    :formal::prior(tr,old_context,pi,depth_prior);
  const double lp_old=old_prior.total();
  if(!std::isfinite(lp_old))throw std::runtime_error("Rotation started from an invalid formal tree state.");
  if(target_terms)target_terms->prior_old=lp_old;
  const double* sigptr = sigma;

  // Draw centered truncated-normal angles sequentially on an ordered path.
  // Dynamic grids follow the selected endpoint policy; fixed grids stay unchanged.
  // Include forward and reverse cutpoint probabilities in the joint MH ratio.
  static thread_local std::vector<double> Xprop_scratch;
  const size_t pn = p * n;
  if (Xprop_scratch.size() != pn) Xprop_scratch.resize(pn);
  std::memcpy(Xprop_scratch.data(), Xold_ptr, sizeof(double) * pn);
  double* Xprop_ptr = Xprop_scratch.data();

  std::vector<RotState::Pending> seq;
  seq.reserve(num_planes);
  std::vector<double> dtheta_seq;
  dtheta_seq.reserve(num_planes);

  Eigen::MatrixXd Qprop;
  if(canonical)Qprop=st.R;
  double logq_theta_fwd = 0.0;
  for (size_t k = 0; k < num_planes; ++k) {
    size_t a, b;
    sample_rotation_pair(p, gen, cand_vars, a, b, support_expand_prob);

    const double mu_fwd = 0.0;
    const double dtheta = sample_truncnorm_(mu_fwd, rotation_step_size, -theta_max, theta_max, gen);
    logq_theta_fwd += log_truncnorm_density_(dtheta, mu_fwd, rotation_step_size,
                                             -theta_max, theta_max);

    RotState::Pending u{a, b, std::cos(dtheta), std::sin(dtheta)};
    seq.push_back(u);
    dtheta_seq.push_back(dtheta);
    if(canonical)apply_givens_left_inplace(Qprop,u.a,u.b,u.c,u.s);
    else apply_givens_inplace(Xprop_ptr, p, n, u.a, u.b, u.c, u.s);

  }

  std::vector<char> touched = touched_from_seq(p, seq);
  // Score the endpoint using a coordinate function of Q, not accumulated X.
  // Untouched rows of Q, X and grid remain identical.
  if(canonical)canonical_project(Qprop,st.raw_train,p,n,Xprop_ptr,&touched);

  xinfo xi_prop;
  xinfo* xi_fwd_ptr = &xi_cur;
  std::vector<unsigned char> fused_active;
  std::vector<unsigned char>* fused_ptr=nullptr;
  if(state_cache && dynamic_xinfo_on_rotation && xinfo_policy==XInfoPolicy::UniformFixedNC) {
    fused_active=state_cache->geometry.active;fused_ptr=&fused_active;
  }
  if (dynamic_xinfo_on_rotation) {
    xi_prop = xi_cur;
    rebuild_xinfo_touched_(Xprop_ptr, p, n, touched, nc_per_var,
                           xinfo_policy, xi_prop, fused_ptr);
    xi_fwd_ptr = &xi_prop;
  }
  xinfo& xi_fwd = *xi_fwd_ptr;
  formal::GeometryCache proposal_geometry;
  if(state_cache)proposal_geometry=state_cache->geometry.proposal(xi_fwd,Xprop_ptr,touched,fused_ptr);
  const formal::Context new_context=state_cache
    ?formal::Context(xi_fwd,proposal_geometry.active,log_weights,formal::ValidatedGeometryTag{})
    :formal::Context(xi_fwd,Xprop_ptr,p,n,log_weights);

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_seq_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_xprop_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  CutSnapshot snap_old;
  snapshot_internal_cuts(tr, snap_old);
  const size_t nb_hint = tr.nbots();

  static thread_local std::vector<tree::tree_cp> bn_old_scratch;
  bn_old_scratch.resize(n);
  for (size_t i = 0; i < n; ++i) bn_old_scratch[i] = tr.bn(Xold_ptr + p*i, xi_cur);
  const double ll_old = tree_log_marglik_hetero_from_bn(bn_old_scratch, yptr, sigptr, n, pi.tau, nb_hint);

  double logq_fwd = 0.0;
  CutSnapshot snap_prop;
  snap_prop.nodes = snap_old.nodes;

  if (retune_cutpoints) {
    logq_fwd = propose_cutpoints_percentile_multi(
      tr, new_context, p, n, bn_old_scratch, Xprop_ptr, touched, gen
    );
    capture_cuts_from_nodes(snap_prop.nodes, snap_prop.cuts);
  } else {
    // Legacy no-retune proposal: C' = C deterministically.
    // The cutpoint proposal ratio is exactly 1, so log q_C terms are zero.
    snap_prop.cuts = snap_old.cuts;
  }

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_cut_fwd_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  double* x_saved = di_ptr->x;
  const formal::PriorTerms new_prior=formal::prior(tr,new_context,pi,depth_prior);
  const double lp_new=new_prior.total();
  if(target_terms){target_terms->prior_new=lp_new;target_terms->logq_forward=logq_fwd;target_terms->likelihood_old=ll_old;}
  if(!std::isfinite(logq_fwd)||!std::isfinite(lp_new)) {
    if(out_logacc)*out_logacc=formal::neginf();
    if(target_terms)target_terms->logacc=formal::neginf();
    return finish_reject(&snap_old,x_saved);
  }
  static thread_local std::vector<tree::tree_cp> bn_new_scratch;
  bn_new_scratch.resize(n);
  for (size_t i = 0; i < n; ++i) bn_new_scratch[i] = tr.bn(Xprop_ptr + p*i, xi_fwd);
  const double ll_new = tree_log_marglik_hetero_from_bn(bn_new_scratch, yptr, sigptr, n, pi.tau, nb_hint);

  if (out_ll_current)  *out_ll_current  = ll_old;
  if (out_ll_proposed) *out_ll_proposed = ll_new;
  if (out_global_ll_current)  *out_global_ll_current  = gaussian_loglik_hetero_from_bn_theta(bn_old_scratch, yptr, sigptr, n);
  if (out_global_ll_proposed) *out_global_ll_proposed = gaussian_loglik_hetero_from_bn_theta(bn_new_scratch, yptr, sigptr, n);

  if (compute_diagnostics && (out_memb_change || out_leaf_n_L1 || out_leaf_sy_L1)) {
    double mc, ln, lsy;
    compute_partition_diagnostics_from_bn(bn_old_scratch, bn_new_scratch, yptr, n, mc, ln, lsy);
    if (out_memb_change) *out_memb_change = mc;
    if (out_leaf_n_L1)   *out_leaf_n_L1   = ln;
    if (out_leaf_sy_L1)  *out_leaf_sy_L1  = lsy;
  } else {
    if (out_memb_change) *out_memb_change = NA_REAL;
    if (out_leaf_n_L1)   *out_leaf_n_L1   = NA_REAL;
    if (out_leaf_sy_L1)  *out_leaf_sy_L1  = NA_REAL;
  }

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_diag_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  double logq_bwd = 0.0;
  if (retune_cutpoints) {
    logq_bwd = backward_logq_percentile_multi_inplace(
      tr, snap_old, snap_prop, old_context, p, n, bn_new_scratch, Xold_ptr, touched
    );
  }
  // If retune_cutpoints == false, C' = C deterministically in both directions,
  // hence log q_C(C | y) - log q_C(C' | x) = 0.

  // Preserve the reference reverse sequence and floating-point summation.
  // Centered angle densities are symmetric, but their sums are deliberately
  // retained here to avoid changing the floating-point acceptance calculation.
  double logq_theta_bwd = 0.0;
  for (size_t rr = seq.size(); rr-- > 0; ) {
    const double dtheta = dtheta_seq[rr];
    const double mu_bwd = 0.0;
    logq_theta_bwd += log_truncnorm_density_(-dtheta, mu_bwd, rotation_step_size,
                                             -theta_max, theta_max);
  }

  if (do_time) {
    const auto t1 = std::chrono::steady_clock::now();
    out_timing->t_cut_bwd_ms = _ms_between_(t0, t1);
    t0 = t1;
  }

  const double logacc = (ll_new - ll_old) + (lp_new - lp_old)
                      + (logq_bwd - logq_fwd)
                      + (logq_theta_bwd - logq_theta_fwd);
  if(target_terms){target_terms->likelihood_new=ll_new;target_terms->logq_reverse=logq_bwd;target_terms->logacc=logacc;}

  if (!std::isfinite(ll_old) || !std::isfinite(ll_new) ||
      !std::isfinite(logq_fwd) || !std::isfinite(logq_bwd) ||
      !std::isfinite(logq_theta_fwd) || !std::isfinite(logq_theta_bwd) ||
      !std::isfinite(logacc)) {
    // A zero reverse mass or invalid target is a rejection, not log ratio zero.
    if (out_logacc) *out_logacc = formal::neginf();
    return finish_reject(&snap_old, x_saved);
  }
  if (out_logacc) *out_logacc = logacc;

  const bool accept = (std::log(gen.uniform()) < logacc);
  if (accept) {
    if(canonical)st.R=Qprop;
    else for (const auto& u : seq) apply_givens_left_inplace(st.R, u.a, u.b, u.c, u.s);
    std::memcpy(st.Xtr.data(), Xprop_ptr, sizeof(double) * pn);
    di_ptr->x = st.Xtr.data();
    if (dynamic_xinfo_on_rotation) xi_cur.swap(xi_prop);
    if(state_cache) {
      state_cache->commit_geometry(proposal_geometry,st.Xtr.data());
      // Use accepted grid: new_context's proposal reference may now be swapped.
      const formal::Context accepted_context(xi_cur,state_cache->geometry.active,
                                             log_weights,formal::ValidatedGeometryTag{});
      state_cache->remember(tr,accepted_context,pi,depth_prior,new_prior);
    }

    if (np && di_test) {
      st.has_test = true;
      if ((size_t)st.Xte.size() != p*np) {
        st.Xte.assign(p*np, 0.0);
        std::memcpy(st.Xte.data(), di_test->x, sizeof(double)*p*np);
      }
      if(canonical)st.canonical_test_dirty=true;
      else for (const auto& u : seq) st.pending.push_back(u);
    }

    if (accumulate_angles) {
      if (st.angle_sum.size()==0 || st.angle_sum.rows()!=(int)p || st.angle_sum.cols()!=(int)p) {
        st.angle_sum = Eigen::MatrixXd::Zero((int)p,(int)p);
        st.angle_count = Eigen::MatrixXi::Zero((int)p,(int)p);
      }
      for (size_t kk = 0; kk < seq.size(); ++kk) {
        const RotState::Pending& u = seq[kk];
        const double theta = dtheta_seq[kk];
        const int ia = (int)u.a;
        const int ib = (int)u.b;
        st.angle_sum(ia, ib) += theta;
        st.angle_sum(ib, ia) -= theta;
        st.angle_count(ia, ib) += 1;
        st.angle_count(ib, ia) += 1;
      }
    }

    if (do_time) {
      const auto t1 = std::chrono::steady_clock::now();
      out_timing->t_accept_ms = _ms_between_(t0, t1);
      out_timing->t_total_ms  = _ms_between_(t_total0, t1);
    }
    return true;
  }

  return finish_reject(&snap_old, x_saved);
}

} // namespace robart

#endif // ROBART_ROTATION_MH_H
