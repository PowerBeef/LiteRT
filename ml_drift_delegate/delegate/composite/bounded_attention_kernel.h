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

// GPU operations for the nodes FuseBoundedAttentionSoftmax creates (see
// bounded_attention.h).
#ifndef ML_DRIFT_DELEGATE_DELEGATE_COMPOSITE_BOUNDED_ATTENTION_KERNEL_H_
#define ML_DRIFT_DELEGATE_DELEGATE_COMPOSITE_BOUNDED_ATTENTION_KERNEL_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift

namespace litert::ml_drift {

// kBoundedMaskType: inputs [logits, mask].
absl::Status CreateBoundedMaskFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                       const std::vector<::ml_drift::Value*>& outputs,
                                       const ::ml_drift::Node& node,
                                       ::ml_drift::GpuModelBuilder* model_builder);

// kBoundedSoftmaxReduceType: inputs [masked logits, params (buffer storage)].
absl::Status CreateBoundedSoftmaxReduceFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                                const std::vector<::ml_drift::Value*>& outputs,
                                                const ::ml_drift::Node& node,
                                                ::ml_drift::GpuModelBuilder* model_builder);

}  // namespace litert::ml_drift

#endif  // ML_DRIFT_DELEGATE_DELEGATE_COMPOSITE_BOUNDED_ATTENTION_KERNEL_H_
