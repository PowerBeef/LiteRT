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

#include "ml_drift_delegate/delegate/composite/bounded_attention_kernel.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/bounded_attention.h"

namespace litert::ml_drift {

absl::Status CreateBoundedMaskFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                       const std::vector<::ml_drift::Value*>& outputs,
                                       const ::ml_drift::Node& node,
                                       ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr = std::any_cast<BoundedMaskAttributes>(&node.operation.attributes);
  if (attr == nullptr || inputs.size() != 2 || outputs.size() != 1) {
    return absl::InvalidArgumentError("Bounded mask expects [logits, mask] and one output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto logits, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto mask, model_builder->GetTensor(inputs[1]->id));
  const ::ml_drift::DataType type = logits.tensor_desc.GetDataType();

  // Elementwise with the logits first, so MergeNodes fuses it into the QK
  // product's epilogue: it then runs only where that product writes. The mask
  // broadcasts like any elementwise second input (GQA-folded rows read their
  // query token's row), as in the SDPA fallback's SelectBoolMask.
  ::ml_drift::ElementwiseDescriptor op_desc;
  op_desc.args.AddFloat("mask_value", attr->mask_value, type);
  if (attr->clamp_low && attr->clamp_high) {
    op_desc.args.AddFloat("clamp_low", *attr->clamp_low, type);
    op_desc.args.AddFloat("clamp_high", *attr->clamp_high, type);
  }
  for (const char* c : {"x", "y", "z", "w"}) {
    const std::string value = std::string("in_value.") + c;
    const std::string kept_value =
        attr->clamp_low && attr->clamp_high
            ? "clamp(" + value + ", args.clamp_low, args.clamp_high)"
            : value;
    op_desc.code += std::string("  out_value.") + c + " = in2_value." + c + " ? " + kept_value +
                    " : args.mask_value;\n";
  }
  auto dst = model_builder->AddTensor(logits.tensor_desc);
  ::ml_drift::OperationDef definition;
  definition.src_tensors.push_back(logits.tensor_desc);
  definition.src_tensors.push_back(mask.tensor_desc);
  definition.dst_tensors.push_back(dst.tensor_desc);
  auto op = std::make_unique<::ml_drift::GPUOperation>(::ml_drift::CreateGpuOperation(
      definition, std::move(op_desc), mask.tensor_desc.GetBHWCShape(),
      dst.tensor_desc.GetBHWCShape()));
  model_builder->AddGpuOperation({logits, mask}, {dst}, std::move(op), kBoundedMaskType);
  return model_builder->UpdateOutputTensor(dst, outputs[0]->id);
}

absl::Status CreateBoundedSoftmaxReduceFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                                const std::vector<::ml_drift::Value*>& outputs,
                                                const ::ml_drift::Node& node,
                                                ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr = std::any_cast<BoundedSoftmaxReduceAttributes>(&node.operation.attributes);
  if (attr == nullptr || inputs.size() != 2 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "Bounded softmax reduction expects [logits, params] and one output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto logits, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto params, model_builder->GetTensor(inputs[1]->id));
  ::ml_drift::SoftmaxRuntimeCheckDesc runtime_check;
  runtime_check.end_ch_index = attr->end_ch_index;
  runtime_check.window_pos_index = attr->window_pos_index;
  runtime_check.window_size = attr->window_size;
  auto reduced = model_builder->SoftmaxReduce(logits, runtime_check, &params);
  return model_builder->UpdateOutputTensor(reduced, outputs[0]->id);
}

}  // namespace litert::ml_drift
