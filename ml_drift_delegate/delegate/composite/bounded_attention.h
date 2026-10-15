// Copyright 2026 Patrice Déry (Agello, PowerBeef/LiteRT)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Agello (T4): attention tails bounded to the filled KV length.
//
// Models exported with odml.runtime_bmm (Gemma 4) end each attention with
//   runtime_bmm(Q, K) -> SELECT_V2(mask, scores, c) [-> MAXIMUM -> MINIMUM]
//     -> SOFTMAX -> runtime_bmm(probabilities, V)
// The two products stop at the filled length (param_tensor), but the select and
// the softmax run over the whole KV capacity: at a 32K window they took about
// 60% of prefill GPU time. FuseBoundedAttentionSoftmax rewrites the tail the way
// ML Drift's own Gemma 4 builder and the SDPA fallback build it:
//   - the select becomes a logits-first elementwise op (kBoundedMaskType), so
//     ML Drift fuses it into the QK product, which only writes the filled part;
//   - the softmax becomes a reduction (kBoundedSoftmaxReduceType: per row, the
//     sum's inverse and the max), bounded by the same param_tensor entry;
//   - the PV product reads the masked logits and computes the probabilities
//     itself from that reduction (its src_exp input), within its own bound.
// No pass over the whole capacity remains. Keys past the filled length are
// masked, so whatever a kernel rounds the bound up to reads masked logits,
// which contribute zero.
#ifndef ML_DRIFT_DELEGATE_DELEGATE_COMPOSITE_BOUNDED_ATTENTION_H_
#define ML_DRIFT_DELEGATE_DELEGATE_COMPOSITE_BOUNDED_ATTENTION_H_

#include <optional>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift

namespace litert::ml_drift {

constexpr const char kBoundedMaskType[] = "agello_bounded_mask";
constexpr const char kBoundedSoftmaxReduceType[] = "agello_bounded_softmax_reduce";

// out = mask ? clamp(logits, clamp_low, clamp_high) : mask_value, with the
// logits as the first input so the op links into their producer.
struct BoundedMaskAttributes {
  float mask_value = -10000.0f;
  std::optional<float> clamp_low;
  std::optional<float> clamp_high;
};

// A softmax reduction over channels [0, params[end_ch_index]), or from a sliding
// window's start (ml_drift::RuntimeWindowStartSlice) when window_pos_index is set.
struct BoundedSoftmaxReduceAttributes {
  int end_ch_index = 2;
  std::optional<int> window_pos_index;
  int window_size = 0;
};

// Whether the rewrite runs: yes unless AGELLO_BOUNDED_ATTENTION=0 (for A/B
// measurements). Part of the program cache key.
bool BoundedAttentionEnabled();

// Whether local layers get their sliding window: yes unless AGELLO_BOUNDED_WINDOW=0
// (for A/B measurements). Part of the program cache key.
bool BoundedWindowEnabled();

// The sliding window of the graph's local attention layers: W when every
// LOGICAL_AND(GREATER_EQUAL(p, keys), LESS(p, keys + W)) mask with int32
// constant keys agrees on it, nullopt otherwise. Tails whose runtime_bmm is not
// global (is_global = false) then read only keys in (p - W, p], from the window's
// start rounded down to 32 keys.
std::optional<int> FindSlidingWindow(const ::ml_drift::GraphFloat32& graph);

// Rewrites every matching attention tail; returns how many it rewrote.
absl::StatusOr<int> FuseBoundedAttentionSoftmax(::ml_drift::GraphFloat32* graph);

}  // namespace litert::ml_drift

#endif  // ML_DRIFT_DELEGATE_DELEGATE_COMPOSITE_BOUNDED_ATTENTION_H_
