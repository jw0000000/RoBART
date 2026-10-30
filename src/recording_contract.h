#ifndef ROBART_RECORDING_CONTRACT_H
#define ROBART_RECORDING_CONTRACT_H

#include "input_contract.h"

// Recording contracts only. No sampler allocation, RNG call, or state mutation.
namespace robart_contract {
inline void recording_schedule(std::size_t trace_every,
                               std::size_t timing_every,
                               bool has_trace_family) {
  count(trace_every, "rot_trace_every");
  count(timing_every, "rot_timing_every", false);
  if (timing_every == 0) return;
  require(has_trace_family,
          "rot_timing_every requires a rotation trace family to be enabled.");
  require(timing_every % trace_every == 0,
          "rot_timing_every must be a multiple of rot_trace_every.");
}

inline int snapshot_iteration(double value, std::size_t total) {
  count(total, "total iterations");
  require(std::isfinite(value) && value == std::floor(value) &&
            value >= 1.0 && value <= static_cast<double>(total),
          "save_state_iters must contain positive integer iterations not exceeding total iterations.");
  return static_cast<int>(value);
}
} // namespace robart_contract
#endif
