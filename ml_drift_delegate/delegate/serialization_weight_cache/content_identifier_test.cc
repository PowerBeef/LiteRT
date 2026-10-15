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

// Agello (T11): the weight cache's content identifier ignores the sizes of
// tensors that aren't constants (the KV caches), and still follows the
// constant inputs.
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "tflite/c/common.h"

namespace ml_drift {
namespace {

// One node reading a KV cache (an arena tensor) and a weight (a constant).
struct FakeSubgraph {
  std::vector<TfLiteTensor> tensors = std::vector<TfLiteTensor>(3);
  TfLiteNode node{};
  TfLiteRegistration registration{};
  TfLiteContext context{};
  TfLiteDelegateParams params{};

  FakeSubgraph(size_t kv_bytes, size_t weight_bytes) {
    tensors[0].allocation_type = kTfLiteArenaRw;
    tensors[0].bytes = kv_bytes;
    tensors[0].type = kTfLiteInt8;
    tensors[1].allocation_type = kTfLiteMmapRo;
    tensors[1].bytes = weight_bytes;
    tensors[1].type = kTfLiteInt8;
    tensors[2].allocation_type = kTfLiteArenaRw;
    tensors[2].bytes = kv_bytes;
    tensors[2].type = kTfLiteFloat16;
    node.inputs = Ints({0, 1});
    node.outputs = Ints({2});
    registration.builtin_code = 9;  // FULLY_CONNECTED
    context.tensors = tensors.data();
    context.tensors_size = static_cast<int>(tensors.size());
    context.impl_ = this;
    context.GetNodeAndRegistration = [](TfLiteContext* context, int,
                                        TfLiteNode** node,
                                        TfLiteRegistration** registration) {
      auto* self = static_cast<FakeSubgraph*>(context->impl_);
      *node = &self->node;
      *registration = &self->registration;
      return kTfLiteOk;
    };
    params.nodes_to_replace = Ints({0});
    params.input_tensors = Ints({0});
    params.output_tensors = Ints({2});
  }
  ~FakeSubgraph() {
    TfLiteIntArrayFree(node.inputs);
    TfLiteIntArrayFree(node.outputs);
    TfLiteIntArrayFree(params.nodes_to_replace);
    TfLiteIntArrayFree(params.input_tensors);
    TfLiteIntArrayFree(params.output_tensors);
  }

  static TfLiteIntArray* Ints(std::vector<int> values) {
    TfLiteIntArray* array = TfLiteIntArrayCreate(static_cast<int>(values.size()));
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
      array->data[i] = values[i];
    }
    return array;
  }

  uint64_t Content() {
    return SerializationWeightCache::GenerateContentModelIdentifier(
        "model", &context, &params, "prefix", kDefault,
        /*prefer_texture_weights=*/false,
        /*allow_src_quantized_fc_conv_ops=*/true,
        /*prepare_weights_in_batches=*/true,
        /*serialize_external_tensors=*/true, /*ordered_by_size=*/true);
  }
  uint64_t Unique() {
    return SerializationWeightCache::GenerateUniqueModelIdentifier(
        "model", &context, &params, "prefix", kDefault,
        /*prefer_texture_weights=*/false,
        /*allow_src_quantized_fc_conv_ops=*/true,
        /*prepare_weights_in_batches=*/true,
        /*serialize_external_tensors=*/true, /*ordered_by_size=*/true);
  }
};

TEST(ContentModelIdentifierTest, IgnoresTheKvCacheSize) {
  FakeSubgraph at_8k(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  FakeSubgraph at_32k(/*kv_bytes=*/32768, /*weight_bytes=*/4096);
  EXPECT_EQ(at_8k.Content(), at_32k.Content());
  // TFLite's fingerprint, the upstream key, tells them apart.
  EXPECT_NE(at_8k.Unique(), at_32k.Unique());
}

TEST(ContentModelIdentifierTest, FollowsTheConstantInputs) {
  FakeSubgraph a(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  FakeSubgraph b(/*kv_bytes=*/8192, /*weight_bytes=*/2048);
  EXPECT_NE(a.Content(), b.Content());
  FakeSubgraph c(/*kv_bytes=*/8192, /*weight_bytes=*/4096);
  c.registration.builtin_code = 3;  // CONV_2D
  EXPECT_NE(a.Content(), c.Content());
}

}  // namespace
}  // namespace ml_drift
