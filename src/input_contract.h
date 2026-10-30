#ifndef ROBART_INPUT_CONTRACT_H
#define ROBART_INPUT_CONTRACT_H

#include <cmath>
#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

// Validation only: these helpers do not allocate sampler state or consume RNG.
namespace robart_contract {
inline void require(bool ok, const std::string& message) {
  if (!ok) throw std::invalid_argument(message);
}
inline void count(std::size_t value, const char* name, bool positive = true) {
  require((!positive || value > 0) && value <= static_cast<std::size_t>(INT_MAX),
          std::string(name) + (positive ? " must be an integer in [1, INT_MAX]." :
                                          " must be an integer in [0, INT_MAX]."));
}
inline std::size_t product(std::size_t x, std::size_t y, const char* name) {
  const std::size_t limit = static_cast<std::size_t>(
    std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
  require(x == 0 || y <= limit / x, std::string(name) + " exceeds the addressable array size.");
  return x * y;
}
inline void schedule(std::size_t n, std::size_t p, std::size_t np, std::size_t m,
                     std::size_t nd, std::size_t burn, std::size_t thin,
                     std::size_t printevery, std::size_t warmup,
                     std::size_t reps, std::size_t K) {
  count(n,"n"); count(p,"p"); count(np,"np",false); count(m,"ntree");
  count(nd,"native post-thinning iteration budget"); count(burn,"nskip",false);
  count(thin,"keepevery"); count(printevery,"printevery");
  count(warmup,"warm_up",false); count(reps,"tree_update_reps"); count(K,"K");
  require(nd % thin == 0, "The native post-burn budget must be divisible by keepevery.");
  require(burn <= static_cast<std::size_t>(INT_MAX) - nd,
          "Total iterations exceed the supported integer iteration range.");
  require(warmup < static_cast<std::size_t>(INT_MAX), "warm_up + 1 overflows the iteration range.");
  product(n,p,"Training design"); product(np,p,"Test design");
  product(nd/thin,n,"Training draws"); product(nd/thin,np,"Test draws");
  product(nd/thin,p,"Variable-probability draws"); product(nd+burn,m,"Rotation trace");
  product(product(p,p,"Rotation matrix"),m,"Rotation snapshots");
}
inline void positive(double x, const char* name) {
  require(std::isfinite(x) && x > 0.0, std::string(name) + " must be finite and positive.");
}
inline void dart(double a, double b, double rho, double theta, double omega) {
  positive(a,"DART a"); positive(b,"DART b"); positive(rho,"DART rho");
  require(std::isfinite(theta) && theta >= 0.0, "DART theta must be finite and nonnegative.");
  positive(omega,"DART omega");
}
} // namespace robart_contract
#endif
