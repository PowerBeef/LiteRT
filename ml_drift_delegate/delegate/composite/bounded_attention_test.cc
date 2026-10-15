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

#include <gtest/gtest.h>
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::GraphFloat32;
using ::ml_drift::Node;
using ::ml_drift::OperationType;
using ::ml_drift::Value;

Value* NewValue(GraphFloat32& graph, DataType type, BHWC shape) {
  Value* value = graph.NewValue();
  value->tensor.type = type;
  value->tensor.shape = shape;
  return value;
}

Node* NewNode(GraphFloat32& graph, const std::string& type, std::initializer_list<Value*> inputs,
              Value* output) {
  Node* node = graph.NewNode();
  node->operation.type = type;
  for (Value* input : inputs) graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  return node;
}

// A Gemma 4 decode attention tail: QK -> SELECT_V2 -> MAXIMUM -> MINIMUM -> SOFTMAX -> PV.
struct Attention {
  GraphFloat32 graph;
  Value* logits;
  Value* mask;
  Value* selected;
  Value* probabilities;
  Value* params;
  Value* v;
  Node* select;
  Node* softmax;
  Node* pv;
};

void Build(Attention& a) {
  GraphFloat32& g = a.graph;
  Value* q = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4, 256));
  Value* k = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 64 * 256));
  a.v = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 256 * 64));
  a.params = NewValue(g, DataType::kInt32, BHWC(1, 1, 1, 7));
  a.mask = NewValue(g, DataType::kBool, BHWC(1, 1, 4, 64));
  a.logits = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4, 64));

  RuntimeBatchedMatMulAttributes qk_attr;
  qk_attr.runtime_check.dst_end_ch_index = 2;
  qk_attr.external_weights = ExternalWeightsAttributes{};
  NewNode(g, kRuntimeBatchedMatMulType, {q, k, a.params}, a.logits)->operation.attributes =
      qk_attr;

  Value* fill = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 1));
  Node* constant = g.NewNode();
  constant->operation.type = ToString(OperationType::kConstant);
  g.SetProducer(constant->id, fill->id);

  a.selected = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4, 64));
  a.select = NewNode(g, ToString(OperationType::kSelectV2), {a.mask, a.logits, fill}, a.selected);
  Value* raised = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4, 64));
  NewNode(g, ToString(OperationType::kMaximum), {a.selected}, raised)->operation.attributes =
      ::ml_drift::ElementwiseAttributes{::ml_drift::ScalarValue(-65504.0f)};
  Value* capped = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4, 64));
  NewNode(g, ToString(OperationType::kMinimum), {raised}, capped)->operation.attributes =
      ::ml_drift::ElementwiseAttributes{::ml_drift::ScalarValue(65504.0f)};
  a.probabilities = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4, 64));
  a.softmax = NewNode(g, ToString(OperationType::kSoftmax), {capped}, a.probabilities);
  a.softmax->operation.attributes = ::ml_drift::SoftmaxAttributes{::ml_drift::Axis::kChannels};

  Value* out = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4, 256));
  RuntimeBatchedMatMulAttributes pv_attr;
  pv_attr.runtime_check.src_end_ch_index = 2;
  pv_attr.external_weights = ExternalWeightsAttributes{};
  a.pv = NewNode(g, kRuntimeBatchedMatMulType, {a.probabilities, a.v, a.params}, out);
  a.pv->operation.attributes = pv_attr;
}

TEST(BoundedAttentionTest, RewritesTheAttentionTail) {
  Attention a;
  Build(a);
  const auto rewritten = FuseBoundedAttentionSoftmax(&a.graph);
  ASSERT_TRUE(rewritten.ok()) << rewritten.status();
  EXPECT_EQ(*rewritten, 1);

  // QK, the bounded mask, the reduction and PV: the capping and the constant are gone.
  EXPECT_EQ(a.graph.nodes().size(), 4u);

  EXPECT_EQ(a.select->operation.type, kBoundedMaskType);
  const auto mask_inputs = a.graph.FindInputs(a.select->id);
  ASSERT_EQ(mask_inputs.size(), 2u);
  EXPECT_EQ(mask_inputs[0]->id, a.logits->id);
  EXPECT_EQ(mask_inputs[1]->id, a.mask->id);
  const auto& mask_attr = std::any_cast<const BoundedMaskAttributes&>(a.select->operation.attributes);
  EXPECT_EQ(mask_attr.mask_value, -65504.0f);
  EXPECT_EQ(mask_attr.clamp_low, -65504.0f);
  EXPECT_EQ(mask_attr.clamp_high, 65504.0f);

  EXPECT_EQ(a.softmax->operation.type, kBoundedSoftmaxReduceType);
  const auto reduce_inputs = a.graph.FindInputs(a.softmax->id);
  ASSERT_EQ(reduce_inputs.size(), 2u);
  EXPECT_EQ(reduce_inputs[0]->id, a.selected->id);
  EXPECT_EQ(reduce_inputs[1]->id, a.params->id);
  EXPECT_EQ(a.probabilities->tensor.shape, BHWC(1, 2, 4, 4));
  EXPECT_EQ(std::any_cast<const BoundedSoftmaxReduceAttributes&>(a.softmax->operation.attributes)
                .end_ch_index,
            2);

  const auto pv_inputs = a.graph.FindInputs(a.pv->id);
  ASSERT_EQ(pv_inputs.size(), 4u);
  EXPECT_EQ(pv_inputs[0]->id, a.selected->id);
  EXPECT_EQ(pv_inputs[1]->id, a.v->id);
  EXPECT_EQ(pv_inputs[2]->id, a.probabilities->id);
  EXPECT_EQ(pv_inputs[3]->id, a.params->id);
  EXPECT_TRUE(
      std::any_cast<const RuntimeBatchedMatMulAttributes&>(a.pv->operation.attributes).src_exp_input);
}

TEST(BoundedAttentionTest, LeavesOtherSoftmaxesAlone) {
  GraphFloat32 graph;
  Value* in = NewValue(graph, DataType::kFloat32, BHWC(1, 1, 1, 16));
  Value* out = NewValue(graph, DataType::kFloat32, BHWC(1, 1, 1, 16));
  Node* softmax = NewNode(graph, ToString(OperationType::kSoftmax), {in}, out);
  const auto rewritten = FuseBoundedAttentionSoftmax(&graph);
  ASSERT_TRUE(rewritten.ok());
  EXPECT_EQ(*rewritten, 0);
  EXPECT_EQ(softmax->operation.type, ToString(OperationType::kSoftmax));
}

TEST(BoundedAttentionTest, SkipsATailWhoseProbabilitiesHaveAnotherReader) {
  Attention a;
  Build(a);
  Value* extra = NewValue(a.graph, DataType::kFloat32, BHWC(1, 2, 4, 64));
  NewNode(a.graph, ToString(OperationType::kCopy), {a.probabilities}, extra);
  const auto rewritten = FuseBoundedAttentionSoftmax(&a.graph);
  ASSERT_TRUE(rewritten.ok());
  EXPECT_EQ(*rewritten, 0);
  EXPECT_EQ(a.softmax->operation.type, ToString(OperationType::kSoftmax));
}

}  // namespace
}  // namespace litert::ml_drift
