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

#include "ml_drift_delegate/delegate/composite/bounded_attention.h"

#include <any>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::GraphFloat32;
using ::ml_drift::Node;
using ::ml_drift::OperationType;
using ::ml_drift::Value;

// A value read by exactly one node and not a graph output.
bool SingleUse(const GraphFloat32& graph, const Value* value) {
  return !graph.IsGraphOutput(value->id) && graph.FindConsumers(value->id).size() == 1;
}

std::optional<float> ScalarParam(const Node& node) {
  const auto* attr = std::any_cast<::ml_drift::ElementwiseAttributes>(&node.operation.attributes);
  if (attr == nullptr) return std::nullopt;
  const auto* scalar = std::get_if<::ml_drift::ScalarValue>(&attr->param);
  if (scalar == nullptr) return std::nullopt;
  const float* value = std::get_if<float>(scalar);
  if (value == nullptr) return std::nullopt;
  return *value;
}

// The int32 values of a constant value, or nullptr.
const std::vector<int32_t>* IntConstant(const GraphFloat32& graph, const Value* value) {
  const Node* producer = graph.FindProducer(value->id);
  if (producer == nullptr || producer->operation.type != ToString(OperationType::kConstant)) {
    return nullptr;
  }
  const auto* attr = std::any_cast<::ml_drift::ConstTensorAttributes>(&producer->operation.attributes);
  if (attr == nullptr) return nullptr;
  const auto* tensor = std::get_if<::ml_drift::TensorInt32>(&attr->tensor);
  return tensor == nullptr ? nullptr : &tensor->data;
}

}  // namespace

std::optional<int> FindSlidingWindow(const GraphFloat32& graph) {
  // A sliding-window mask keeps key k for position p when p >= k and p < k + W:
  // LOGICAL_AND(GREATER_EQUAL(p, keys), LESS(p, keys + W)), the keys and keys + W
  // being int32 constants. Every such mask must agree on W.
  std::optional<int> window;
  for (const Node* node : graph.nodes()) {
    if (node == nullptr || node->operation.type != ToString(OperationType::kLogicalAnd)) continue;
    const auto inputs = graph.FindInputs(node->id);
    if (inputs.size() != 2) continue;
    const Node* ge = graph.FindProducer(inputs[0]->id);
    const Node* lt = graph.FindProducer(inputs[1]->id);
    if (ge == nullptr || lt == nullptr) continue;
    if (ge->operation.type == ToString(OperationType::kLess)) std::swap(ge, lt);
    if (ge->operation.type != ToString(OperationType::kGreaterEqual) ||
        lt->operation.type != ToString(OperationType::kLess)) {
      continue;
    }
    const auto ge_inputs = graph.FindInputs(ge->id);
    const auto lt_inputs = graph.FindInputs(lt->id);
    if (ge_inputs.size() != 2 || lt_inputs.size() != 2 || ge_inputs[0]->id != lt_inputs[0]->id) {
      continue;
    }
    const std::vector<int32_t>* keys = IntConstant(graph, ge_inputs[1]);
    const std::vector<int32_t>* shifted = IntConstant(graph, lt_inputs[1]);
    if (keys == nullptr || shifted == nullptr || keys->empty() || keys->size() != shifted->size()) {
      continue;
    }
    const int w = (*shifted)[0] - (*keys)[0];
    bool uniform = w > 0;
    for (size_t i = 0; uniform && i < keys->size(); ++i) {
      uniform = (*shifted)[i] - (*keys)[i] == w;
    }
    if (!uniform || (window && *window != w)) return std::nullopt;
    window = w;
  }
  return window;
}

namespace {

// The nodes of one attention tail.
struct Tail {
  Node* qk = nullptr;
  Value* logits = nullptr;
  Node* select = nullptr;
  Value* mask = nullptr;
  Value* fill = nullptr;
  Value* selected = nullptr;  // The select's output.
  Node* maximum = nullptr;    // Optional capping (enable_infinite_float_capping).
  Node* minimum = nullptr;
  std::optional<float> clamp_low;
  std::optional<float> clamp_high;
  Node* softmax = nullptr;
  Value* probabilities = nullptr;
  Node* pv = nullptr;
  Value* params = nullptr;
};

// Matches the tail that ends in `softmax`; nullopt if anything differs from the
// expected shape of a Gemma 4 runtime_bmm attention.
std::optional<Tail> MatchTail(const GraphFloat32& graph, Node* softmax) {
  Tail tail;
  tail.softmax = softmax;
  if (const auto* attr = std::any_cast<::ml_drift::SoftmaxAttributes>(&softmax->operation.attributes);
      attr != nullptr && attr->axis != ::ml_drift::Axis::kChannels) {
    return std::nullopt;
  }
  const auto softmax_inputs = graph.FindInputs(softmax->id);
  const auto softmax_outputs = graph.FindOutputs(softmax->id);
  if (softmax_inputs.size() != 1 || softmax_outputs.size() != 1) return std::nullopt;
  tail.probabilities = softmax_outputs[0];

  // The PV product: the FC form (external weights), bounded by its source,
  // reading [probabilities, weights, params] with no bias yet.
  if (!SingleUse(graph, tail.probabilities)) return std::nullopt;
  tail.pv = graph.FindConsumers(tail.probabilities->id)[0];
  if (tail.pv->operation.type != kRuntimeBatchedMatMulType) return std::nullopt;
  const auto* pv_attr =
      std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.pv->operation.attributes);
  if (pv_attr == nullptr || !pv_attr->external_weights.has_value() ||
      !pv_attr->runtime_check.src_end_ch_index.has_value() ||
      pv_attr->runtime_check.dst_end_ch_index.has_value() || pv_attr->src_exp_input) {
    return std::nullopt;
  }
  const auto pv_inputs = graph.FindInputs(tail.pv->id);
  if (pv_inputs.size() != 3 || pv_inputs[0]->id != tail.probabilities->id) return std::nullopt;
  tail.params = pv_inputs[2];

  // Back from the softmax: [MINIMUM <- MAXIMUM <-] SELECT_V2.
  Value* value = softmax_inputs[0];
  if (!SingleUse(graph, value)) return std::nullopt;
  Node* producer = graph.FindProducer(value->id);
  if (producer != nullptr && producer->operation.type == ToString(OperationType::kMinimum)) {
    tail.minimum = producer;
    tail.clamp_high = ScalarParam(*producer);
    const auto inputs = graph.FindInputs(producer->id);
    if (!tail.clamp_high || inputs.size() != 1) return std::nullopt;
    value = inputs[0];
    if (!SingleUse(graph, value)) return std::nullopt;
    producer = graph.FindProducer(value->id);
    if (producer == nullptr || producer->operation.type != ToString(OperationType::kMaximum)) {
      return std::nullopt;
    }
    tail.maximum = producer;
    tail.clamp_low = ScalarParam(*producer);
    const auto max_inputs = graph.FindInputs(producer->id);
    if (!tail.clamp_low || max_inputs.size() != 1) return std::nullopt;
    value = max_inputs[0];
    if (!SingleUse(graph, value)) return std::nullopt;
    producer = graph.FindProducer(value->id);
  }
  if (producer == nullptr || producer->operation.type != ToString(OperationType::kSelectV2)) {
    return std::nullopt;
  }
  tail.select = producer;
  tail.selected = value;
  const auto select_inputs = graph.FindInputs(tail.select->id);
  if (select_inputs.size() != 3) return std::nullopt;
  tail.mask = select_inputs[0];
  tail.logits = select_inputs[1];
  tail.fill = select_inputs[2];
  if (tail.mask->tensor.type != ::ml_drift::DataType::kBool) return std::nullopt;
  const Node* fill_producer = graph.FindProducer(tail.fill->id);
  if (fill_producer == nullptr ||
      fill_producer->operation.type != ToString(OperationType::kConstant) ||
      tail.fill->tensor.shape.DimensionsProduct() != 1) {
    return std::nullopt;
  }

  // The QK product: bounded by its destination on the same params entry.
  if (!SingleUse(graph, tail.logits)) return std::nullopt;
  tail.qk = graph.FindProducer(tail.logits->id);
  if (tail.qk == nullptr || tail.qk->operation.type != kRuntimeBatchedMatMulType) {
    return std::nullopt;
  }
  const auto* qk_attr =
      std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.qk->operation.attributes);
  if (qk_attr == nullptr || !qk_attr->runtime_check.dst_end_ch_index.has_value() ||
      *qk_attr->runtime_check.dst_end_ch_index != *pv_attr->runtime_check.src_end_ch_index) {
    return std::nullopt;
  }
  const auto qk_inputs = graph.FindInputs(tail.qk->id);
  if (qk_inputs.empty() || qk_inputs.back()->id != tail.params->id) return std::nullopt;
  if (tail.logits->tensor.shape != tail.selected->tensor.shape) return std::nullopt;
  return tail;
}

absl::Status Rewrite(GraphFloat32* graph, const Tail& tail, std::optional<int> window) {
  // 1. The select becomes the bounded mask: (logits, mask), with the capping
  //    folded in. Masked keys get what the capped select gave them.
  for (Value* input : {tail.mask, tail.logits, tail.fill}) {
    ABSL_RETURN_IF_ERROR(graph->RemoveConsumer(tail.select->id, input->id));
  }
  graph->AddConsumer(tail.select->id, tail.logits->id);
  graph->AddConsumer(tail.select->id, tail.mask->id);
  BoundedMaskAttributes mask_attr;
  if (tail.clamp_low && tail.clamp_high) {
    mask_attr.mask_value = *tail.clamp_low;
    mask_attr.clamp_low = tail.clamp_low;
    mask_attr.clamp_high = tail.clamp_high;
  }
  tail.select->operation.type = kBoundedMaskType;
  tail.select->operation.attributes = mask_attr;
  if (graph->FindConsumers(tail.fill->id).empty()) {
    const Node* fill_producer = graph->FindProducer(tail.fill->id);
    if (fill_producer != nullptr) ABSL_RETURN_IF_ERROR(graph->DeleteNode(fill_producer->id));
    ABSL_RETURN_IF_ERROR(graph->DeleteValue(tail.fill->id));
  }

  // 2. Drop the capping nodes: the softmax reads the masked logits directly.
  if (tail.minimum != nullptr) {
    const Value* capped = graph->FindInputs(tail.softmax->id)[0];
    const Value* clamped_low = graph->FindOutputs(tail.maximum->id)[0];
    ABSL_RETURN_IF_ERROR(graph->ReplaceInput(tail.softmax->id, capped->id, tail.selected->id));
    const auto capped_id = capped->id;
    const auto clamped_low_id = clamped_low->id;
    ABSL_RETURN_IF_ERROR(graph->DeleteNode(tail.minimum->id));
    ABSL_RETURN_IF_ERROR(graph->DeleteNode(tail.maximum->id));
    ABSL_RETURN_IF_ERROR(graph->DeleteValue(capped_id));
    ABSL_RETURN_IF_ERROR(graph->DeleteValue(clamped_low_id));
  }

  // 3. The softmax becomes the bounded reduction: (masked logits, params) ->
  //    per row (1 / sum, max) in a 4-channel tensor.
  auto* qk_attr = std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.qk->operation.attributes);
  auto* pv_attr = std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.pv->operation.attributes);
  graph->AddConsumer(tail.softmax->id, tail.params->id);
  tail.softmax->operation.type = kBoundedSoftmaxReduceType;
  BoundedSoftmaxReduceAttributes reduce_attr{.end_ch_index =
                                                 *pv_attr->runtime_check.src_end_ch_index};

  // A sliding-window layer (not global) also starts at its window: the QK product
  // skips the keys below it (its fused mask masks them), the reduction and the PV
  // product start there. params[0] is the first new token's position.
  if (window && qk_attr->is_global.has_value() && !*qk_attr->is_global) {
    qk_attr->runtime_check.window_pos_index = 0;
    qk_attr->runtime_check.window_size = *window;
    qk_attr->runtime_check.dst_window = true;
    pv_attr->runtime_check.window_pos_index = 0;
    pv_attr->runtime_check.window_size = *window;
    pv_attr->runtime_check.src_window = true;
    reduce_attr.window_pos_index = 0;
    reduce_attr.window_size = *window;
  }
  tail.softmax->operation.attributes = reduce_attr;
  ::ml_drift::BHWC& reduced_shape = tail.probabilities->tensor.shape;
  reduced_shape.c = 4;

  // 4. The PV product reads the masked logits, with the reduction as src_exp:
  //    [masked logits, weights, reduction, params].
  ABSL_RETURN_IF_ERROR(
      graph->ReplaceInput(tail.pv->id, tail.probabilities->id, tail.selected->id));
  ABSL_RETURN_IF_ERROR(graph->RemoveConsumer(tail.pv->id, tail.params->id));
  graph->AddConsumer(tail.pv->id, tail.probabilities->id);
  graph->AddConsumer(tail.pv->id, tail.params->id);
  pv_attr->src_exp_input = true;
  return absl::OkStatus();
}

}  // namespace

bool BoundedAttentionEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_BOUNDED_ATTENTION");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

bool BoundedWindowEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_BOUNDED_WINDOW");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

absl::StatusOr<int> FuseBoundedAttentionSoftmax(GraphFloat32* graph) {
  if (graph == nullptr) return 0;
  const std::optional<int> window =
      BoundedWindowEnabled() ? FindSlidingWindow(*graph) : std::nullopt;
  std::vector<::ml_drift::NodeId> softmax_ids;
  for (Node* node : graph->nodes()) {
    if (node != nullptr && node->operation.type == ToString(OperationType::kSoftmax)) {
      softmax_ids.push_back(node->id);
    }
  }
  int rewritten = 0;
  for (::ml_drift::NodeId id : softmax_ids) {
    Node* softmax = graph->GetNode(id);
    if (softmax == nullptr) continue;
    std::optional<Tail> tail = MatchTail(*graph, softmax);
    if (!tail) continue;
    ABSL_RETURN_IF_ERROR(Rewrite(graph, *tail, window));
    ++rewritten;
  }
  return rewritten;
}

}  // namespace litert::ml_drift
