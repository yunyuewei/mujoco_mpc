// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/planners/hierarchical_sampling/policy.h"

#include <absl/random/random.h>

#include <absl/log/check.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

using mjpc::spline::TimeSpline;

// allocate memory
void HierarchicalSamplingPolicy::Allocate(const mjModel* model, const Task& task,
                              int horizon) {
  // model
  this->model = model;

  // spline points
  num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                         "sampling_spline_points");
  dim_high_level_action = model->nv;  // number of high-level actions, set to joint number for MS model
  high_level_actions.resize(kMaxTrajectoryHorizon * dim_high_level_action);   // (horizon-1 x num_action)

  // plan = TimeSpline(/*dim=*/model->nu);
  plan = TimeSpline(/*dim=*/dim_high_level_action);
  plan.Reserve(num_spline_points);
}

// reset memory to zeros
void HierarchicalSamplingPolicy::Reset(int horizon, const double* initial_repeated_action) {
  plan.Clear();
  if (initial_repeated_action != nullptr) {
    plan.AddNode(0, absl::MakeConstSpan(initial_repeated_action, dim_high_level_action));
  }
}

// set action from policy
void HierarchicalSamplingPolicy::Action(double* action, const double* state,
                            double time) const {
  CHECK(action != nullptr);
  
  //TODO: sample high-level action and then convert to low-level action
  std::vector<double> high_level_action(dim_high_level_action);
  plan.Sample(time, absl::MakeSpan(&high_level_action[0], dim_high_level_action));
  HighToLowAction(&high_level_action[0], action, state, time);
  // Clamp controls
  Clamp(action, model->actuator_ctrlrange, model->nu);
}

void HierarchicalSamplingPolicy::HighToLowAction(double* high_level_action, double* action, const double* state, double time) const {
  CHECK(action != nullptr);
  //add random noise for test
  // sampling token
  absl::BitGen gen_;
  for (int k = 0; k < dim_high_level_action; k++) {
      double noise = absl::Gaussian<double>(gen_, 0.0, 0.05);
      action[k] = noise;
    }
}

// copy policy
void HierarchicalSamplingPolicy::CopyFrom(const HierarchicalSamplingPolicy& policy, int horizon) {
  this->plan = policy.plan;
  num_spline_points = policy.num_spline_points;
}

// copy parameters
void HierarchicalSamplingPolicy::SetPlan(const TimeSpline& plan) {
  this->plan = plan;
}

}  // namespace mjpc
