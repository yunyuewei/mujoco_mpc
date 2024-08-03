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

#include "mjpc/planners/hierarchical_pd/policy.h"

#include <absl/random/random.h>

#include <absl/log/check.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"
#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <time.h>
#include <glpk.h>


namespace mjpc {

using mjpc::spline::TimeSpline;

// allocate memory
void HierarchicalPDPolicy::Allocate(const mjModel* model, const Task& task,
                              int horizon) {
  // model
  this->model = model;

  // spline points
  num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                         "sampling_spline_points");
  dim_high_level_action = model->nv;  // number of high-level actions, set to joint number for MS model
  // dim_high_level_action = 10;  // number of high-level actions, set to joint number for MS model
  high_level_actions.resize(kMaxTrajectoryHorizon * dim_high_level_action);   // (horizon-1 x num_action)

  // plan = TimeSpline(/*dim=*/model->nu);
  plan = TimeSpline(/*dim=*/dim_high_level_action);
  plan.Reserve(num_spline_points);
  
	// initialize data copy
  data_copy = mj_makeData(model);
	data_copy2 = mj_makeData(model);

  // mj_forward(model, data_copy);
  // mj_forward(model, data_copy2);

  
//   // initialize LP matrices


}

// reset memory to zeros
void HierarchicalPDPolicy::Reset(int horizon, const double* initial_repeated_action) {
  plan.Clear();
  if (initial_repeated_action != nullptr) {
    plan.AddNode(0, absl::MakeConstSpan(initial_repeated_action, dim_high_level_action));
  }
}

// set action from policy
void HierarchicalPDPolicy::Action(double* action, const double* state,
                            double time) const {
  CHECK(action != nullptr);
  
  //TODO: sample high-level action and then convert to low-level action
  // std::vector<double> high_level_action(dim_high_level_action);
  plan.Sample(time, absl::MakeSpan(&action[0], dim_high_level_action));
  // HighToLowAction(&high_level_action[0], action, );
  // // Clamp controls
//   Clamp(action, model->actuator_ctrlrange, model->nu);
  Clamp(action, model->jnt_range, model->nq);
}

void HierarchicalPDPolicy::HierarchicalAction(double* action, mjData* data) const {
  CHECK(action != nullptr);
  
  //TODO: sample high-level action and then convert to low-level action
  std::vector<double> high_level_action(dim_high_level_action);
  plan.Sample(data->time, absl::MakeSpan(&high_level_action[0], dim_high_level_action));

  // Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(data->qvel, model->nu);
  // Eigen::VectorXd qpos = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nu);


  // std::cout<<"pos and vel "<<qvel.maxCoeff()<<" "<<qvel.minCoeff()<<" "<<qpos.maxCoeff()<<" "<<qpos.minCoeff()<<std::endl;
  
  HighToLowAction(&high_level_action[0], action, data);
  // Clamp controls
  Clamp(action, model->actuator_ctrlrange, model->nu);
}

void HierarchicalPDPolicy::HighToLowAction(double* high_level_action, double* action, mjData* data) const {
  CHECK(action != nullptr);
  // absl::BitGen gen_;

  // Clamp(high_level_action, model->jnt_range, model->nq);
  // Eigen::VectorXd high_act_vec = Eigen::Map<Eigen::VectorXd>(high_level_action, model->nv);
  // std::cout<<"high level action "<<high_act_vec.maxCoeff()<<" "<<high_act_vec.minCoeff()<<std::endl;

  // position
  // high_level_action[6] = 2;
  // high_level_action[56] = 3.14;
  // high_level_action[76] = 3.14;

  // high_level_action[56] = 1.57;
  // high_level_action[76] = 1.57;

  

  // high_level_action[56] = 2;
  // high_level_action[76] = 2;

  // restrict the next position given the current position
  // for (int i=0; i<model->nv; i++) {
  //   // double pos_diff = high_level_action[i] - data->qpos[i];
  //   // if (pos_diff > 3*model->opt.timestep) {
  //   //   high_level_action[i] = data->qpos[i] + 3*model->opt.timestep;
  //   // }
  //   // if (pos_diff < -3*model->opt.timestep) {
  //   //   high_level_action[i] = data->qpos[i] - 3*model->opt.timestep;
  //   // }
  //   // if (i < 6) {
  //     high_level_action[i] = data_copy->qpos[i];
  //   // }
  //   // high_level_action[i] = data_copy->qpos[i];
  // }

  // assign current qpos to excluded joint
  for (int i = 0; i < excludeList.size(); ++i) {
        high_level_action[excludeList[i]] = data_copy->qpos[excludeList[i]];
  }
  // if (data_copy->time > 100){
  //   std::cout<<" action "<<high_level_action[6]<<std::endl;
  // }
  

  

  Clamp(high_level_action, model->jnt_range, model->nq);
  Eigen::VectorXd action_vec = get_ctrl(high_level_action);

  // velocity
  // for (int i=0; i<model->nv; i++) {
  //   high_level_action[i] = -data_copy->qpos[i]/model->opt.timestep;
  // }
  // // high_level_action[6] = 2;
  // high_level_action[56] = (3.14-data_copy->qpos[56])/model->opt.timestep;
  // high_level_action[76] = (3.14-data_copy->qpos[76])/model->opt.timestep;
  // // high_level_action[56] = 3;
  // // high_level_action[56] = 3;
  // for (int i=0; i<model->nv; i++) {
  //   if (high_level_action[i] > 100) {
  //     high_level_action[i] = 100;
  //   }
  //   if (high_level_action[i] < -100) {
  //     high_level_action[i] = -1;
  //   }
  // }

  // Eigen::VectorXd action_vec = get_ctrl2(high_level_action);
  


  
  
  
  // std::cout<<"get ctrl used "<<(double)(clock() - start)/CLOCKS_PER_SEC<<std::endl;

  for (int k = 0; k < model->nu; k++) {
      action[k] =action_vec(k);
    }

}

// get control from mj data
Eigen::VectorXd HierarchicalPDPolicy::get_mus_ctrl() const {

  Eigen::VectorXd act0 = Eigen::Map<Eigen::VectorXd>(data_copy->act, model->nu);
  Eigen::VectorXd one_vec = Eigen::VectorXd::Ones(model->nu);
  // Eigen::VectorXd ctrl0 = Eigen::Map<Eigen::VectorXd>(data_copy2->ctrl, model->nu);
  // std::cout<<"act "<< act0 <<std::endl;
  // std::cout<<"ctrl0 "<< ctrl0 <<std::endl;
  
  double ts = model->opt.timestep;
  // for (int i=0; i<model->nu; i++) {
  //   act0[i] = (double)i/model->nu;
  // }
  Eigen::VectorXd tA = 0.01 * (0.5*one_vec.array() + 1.5 * act0.array());
  Eigen::VectorXd tD = 0.04 / (0.5*one_vec.array() + 1.5 * act0.array());
  Eigen::VectorXd tausmooth = 5 * one_vec;
  Eigen::VectorXd tau1 = ((tA - tD) * 1.875).array() / tausmooth.array();
  Eigen::VectorXd tau2 = (tA + tD) * 0.5;


  Eigen::VectorXd gain(model->nu);
  Eigen::VectorXd bias(model->nu);
  Eigen::VectorXd force0(model->nu);
  
  Eigen::VectorXd len_ranges(model->nu);
  for (int idx_actuator = 0; idx_actuator < model->nu; ++idx_actuator) {
    double length = data_copy2->actuator_length[idx_actuator];
    mjtNum* lengthrange = (mjtNum*) mju_malloc(2 * sizeof(mjtNum));
    lengthrange[0] = model->actuator_lengthrange[2*idx_actuator];
    lengthrange[1] = model->actuator_lengthrange[2*idx_actuator+1];
    double velocity = data_copy2->actuator_velocity[idx_actuator];
    double acc0 = model->actuator_acc0[idx_actuator];
    mjtNum* prmb = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
    // Eigen::VectorXd prmb(9);
    for (int j = 0; j<9; j++) {
      prmb[j] = model->actuator_biasprm[10*idx_actuator+j];
    // std::cout<<"biasprm "<<j<<" "<<prmb[j]<<std::endl;
    }
    // std::cout<<std::endl;
    // Eigen::VectorXd prmg(9);
    mjtNum* prmg = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
    for (int j = 0; j<9; j++) {
      prmg[j] = model->actuator_gainprm[10*idx_actuator+j];
    // std::cout<<"gainprm "<<j<<" "<<prmg[j]<<std::endl;
    }
    
    len_ranges(idx_actuator) = lengthrange[1] - lengthrange[0];
    // std::cout<<std::endl;
    bias[idx_actuator] = mju_muscleBias(length, lengthrange, acc0, prmb);
    double g = mju_muscleGain(length, velocity, lengthrange, acc0, prmg);
    if (g >= -1) {
      g = -1;
    }
    force0(idx_actuator) = prmg[2];
    gain[idx_actuator] = g;

    //delete pointers
    delete lengthrange;
    delete prmb;
    delete prmg;
  }

  // PD control
  double kp = 50;
  double kd = 10;
  // double kp = 0.05;
  // double kd = 0.01;
  // std::cout<<"force0 "<<force0<<std::endl;
  // muscle force = max(0, kp*(target_length-current_length)-kd*qvel)
  Eigen::VectorXd target_muscle_length = Eigen::Map<Eigen::VectorXd>(data_copy2->actuator_length, model->nu);
  Eigen::VectorXd current_muscle_length = Eigen::Map<Eigen::VectorXd>(data_copy->actuator_length, model->nu);
  Eigen::VectorXd muscle_velocity = Eigen::Map<Eigen::VectorXd>(data_copy->actuator_velocity, model->nu);
  // std::cout<<"current muscle vel"<<muscle_velocity.maxCoeff()<<" "<<muscle_velocity.minCoeff()<<std::endl;
  // std::cout<<"current muscle len "<<current_muscle_length.maxCoeff()<<" "<<current_muscle_length.minCoeff()<<std::endl;
  Eigen::VectorXd length_diff = target_muscle_length - current_muscle_length;
  // std::cout<<"muscle len diff "<<length_diff.maxCoeff()<<" "<<length_diff.minCoeff()<<std::endl;

  Eigen::VectorXd muscle_force = kp*(target_muscle_length-current_muscle_length).array()/len_ranges.array() - kd * muscle_velocity.array()/len_ranges.array();
  Eigen::VectorXd length0 = Eigen::Map<Eigen::VectorXd>(model->actuator_length0, model->nu);
  
  for (int i=0; i<model->nu; i++) {
    if (muscle_force(i) > 0) {
      muscle_force(i) = 0;
    }
    else {
      // muscle_force(i) = muscle_force(i)*force0(i);
      muscle_force(i) = muscle_force(i);
    }
  }
  // std::cout<<"muscle force "<<muscle_force.maxCoeff()<<" "<<muscle_force.minCoeff()<<std::endl;
  // std::cout<<"muscle force "<<muscle_force.minCoeff()<<" "<<muscle_force(79)<<std::endl;
  

  Eigen::VectorXd target_act = (muscle_force.array()-bias.array()) / gain.array();

  Eigen::VectorXd b1 = act0.array() + (ts*(one_vec - act0)).array()  / (tau2 + tau1 * (one_vec - act0)).array();
  Eigen::VectorXd b2 = act0.array() - act0.array() * ts / (tau2- tau1 * act0).array();

  // std::cout<<"b1 "<<b1<<std::endl;
  // std::cout<<"b2 "<<b2<<std::endl;
  // for (int i=0; i<model->nu; i++) {
  //   std::cout<<"muscle "<<i<<": "<<act0(i)<<" "<<b1(i)<<" "<<b2(i)<<std::endl;
  // }
  

  for (int i=0; i<model->nu; i++) {
    double ub, lb;
    if (b1[i] > b2[i]) {
      ub = b1[i];
      lb = b2[i];
    }
    else {
      ub = b2[i];
      lb = b1[i];
    }
    if (target_act[i] < lb) {
      target_act[i] = lb;
    }
    if (target_act[i] > ub) {
      target_act[i] = ub;
    }
    // if (target_act[i] < 0) {
    //   target_act[i] = 0;
    // }
    // if (target_act[i] > 1) {
    //   target_act[i] = 1;
    // }
  }
  // std::cout<<"gain "<<gain<<std::endl;
  // std::cout<<"bias "<<bias<<std::endl;
  // std::cout<<"target_act "<<target_act<<std::endl;
  // std::cout<<"time "<<data_copy->time<<std::endl;
  // int low_track = 79;
  // int high_track = 79;

  // int low_track = 565;
  // int high_track = 578;
  // for (int i=0; i<model->nu; i++) {
  //   if (i<low_track || i>high_track 
  //   // || data_copy->time == 0
  //   ) {
  //     continue;
  //   }
  //   // std::cout<<"muscle "<<i<<": "<<gain(i)<<" "<<bias(i)<<" "<<target_act(i)<<" "<<b1(i)<<" "<<b2(i)<<" "<<length_diff(i)<<" "<<data_copy->qpos[56]<<" "<<data_copy2->qpos[56]<<std::endl;
  //   std::cout<<"muscle "<<i<<": "<<" "<<target_muscle_length(i)<<" "<<current_muscle_length(i)<<" "<<target_muscle_length(i)-current_muscle_length(i)<<" "<<length_diff(i)<<" "<<muscle_velocity(i)<<" "<<data_copy->qpos[56]<<" "<<data_copy2->qpos[56]<<std::endl;
  // }

  // for (int i=0; i<model->nu; i++) {
  //   if (i<low_track || i>high_track 
  //   // || data_copy->time == 0
  //   ) {
  //     continue;
  //   }
  //   // std::cout<<"muscle "<<i<<": "<<gain(i)<<" "<<bias(i)<<" "<<target_act(i)<<" "<<b1(i)<<" "<<b2(i)<<" "<<length_diff(i)<<" "<<data_copy->qpos[56]<<" "<<data_copy2->qpos[56]<<std::endl;
  //   std::cout<<"act "<<i<<": "<<gain(i)<<" "<<bias(i)<<" "<<act0(i)<<" "<<target_act(i)<<" "<<b1(i)<<" "<<b2(i)<<" "<<muscle_force(i)<<std::endl;
  // }
  
  // std::cout<<"t2 "<<tau2<<std::endl;
  Eigen::VectorXd nominator = act0.array() * act0.array() * tau1.array() -
                              act0.array() * tau2.array() +
                              ts * act0.array() -
                              target_act.array() * act0.array() * tau1.array() +
                              target_act.array() * tau2.array();
  // std::cout<<"nominator "<<nominator<<std::endl;
  

  Eigen::VectorXd denominator = act0.array() * tau1.array() +
                                ts * one_vec.array() -
                                target_act.array() * tau1.array();
  
  // std::cout<<"denominator "<<denominator<<std::endl;
  // for (int i=0; i<model->nu; i++) {
  //   std::cout<<"muscle "<<i<<": "<<nominator(i)<<" "<<denominator(i)<<" "<<target_act(i)<<" "<<tau2(i)<<std::endl;
  // }
  
  // mju_error_i(
  //     "run end here", 0
  //     );
  // for (int i=0; i<model->nu; i++) {
  //   if (abs(denominator[i]) < 1e-6) {
  //     if (denominator[i] < 0) {
  //       denominator[i] = -1e-6;
  //     }
  //     else {
  //     denominator[i] = 1e-6 ;
  //     }
  //   }
  // }
  
  Eigen::VectorXd ctrl_vec = nominator.array() / (denominator.array());


  // for (int i=0; i<model->nu; i++) {
  //   if (i<low_track || i>high_track 
  //   // || data_copy->time == 0
  //   ) {
  //     continue;
  //   }
  //   std::cout<<"control "<<i<<": "<<nominator(i)<<" "<<denominator(i)<<" "<<target_act(i)<<" "<<ctrl_vec(i)<<std::endl;
  // }
  



  // Eigen::VectorXd ctrl_vec = Eigen::VectorXd::Zero(model->nu);
  // Eigen::VectorXd ctrl_vec = act_vec.array() / ((gain.array() * ts));

  // for (int i = 0; i < model->nu; i++) {
  // ctrl_vec[i] = std::clamp(ctrl_vec[i], 0.0, 1.0);
  // }
  // return ctrl;

  // std::cout<<AM.size()<<P.size()<<q.size()<<lb.size()<<x.size()<<ctrl2.size()<<std::endl;

  // double* ctrl = ctrl_vec.data();
  // std::cout<<"control "<<ctrl_vec.sum()<<std::endl;
  for (int i = 0; i < model->nu; i++) {
    // ctrl_vec[i] = std::clamp(ctrl_vec[i], 0.0, 1.0);
    if (ctrl_vec(i) > 1) {
      ctrl_vec(i) = 1;
    }
    if (ctrl_vec(i) < 0) {
      ctrl_vec(i) = 0;
    }
  }
  // std::cout<<"control "<<data_copy->time<<" "<<ctrl_vec.sum()<<std::endl;
  
  // for (int i = 0; i < model->nu; i++) {
  //   std::cout<<"ctrl "<<i<<" "<<ctrl[i]<<std::endl;
  // }
  // std::cout<<"finish get ctrl"<<std::endl;
  return ctrl_vec;
}

// get control by position
Eigen::VectorXd HierarchicalPDPolicy::get_ctrl(double* target_qpos) const{
  mju_copy(data_copy2->qvel, target_qpos, model->nq);
  mju_subFrom(data_copy2->qvel, data_copy2->qpos, model->nq);
  mju_scl(data_copy2->qvel, data_copy2->qvel, 1/model->opt.timestep, model->nq);
  mju_copy(data_copy2->qpos, target_qpos, model->nq);
  mj_step1(model, data_copy2);// gain, bias, and moment depend on qpos and qvel
  Eigen::VectorXd ctrl_vec = get_mus_ctrl();
  return ctrl_vec;
}


//get control by velocity
Eigen::VectorXd HierarchicalPDPolicy::get_ctrl2(double* target_qvel) const{



  mju_copy(data_copy2->qvel, target_qvel, model->nq);
  // mju_scl(data_copy2->qvel, data_copy2->qvel, 1/model->opt.timestep, model->nq);
  mju_scl(data_copy2->qpos, data_copy2->qvel, model->opt.timestep, model->nq);
  mju_addTo(data_copy2->qpos, data_copy->qpos, model->nq);
  Clamp(data_copy2->qpos, model->jnt_range, model->nq);
  mj_step1(model, data_copy2);// gain, bias, and moment depend on qpos and qvel
  Eigen::VectorXd ctrl_vec = get_mus_ctrl();
  return ctrl_vec;
}



// copy policy
void HierarchicalPDPolicy::CopyFrom(const HierarchicalPDPolicy& policy, int horizon) {
  this->plan = policy.plan;
  num_spline_points = policy.num_spline_points;
  // data_copy = mj_copyData(model, policy.data_copy);
  // data_copy2 = mj_copyData(model, policy.data_copy2); 
  mju_copy(data_copy->qpos, policy.data_copy->qpos, model->nq);
  mju_copy(data_copy->qvel, policy.data_copy->qvel, model->nv);
  mju_copy(data_copy->act,  policy.data_copy->act,  model->na);
  mju_copy(data_copy->actuator_length, policy.data_copy->actuator_length, model->nu);
  mju_copy(data_copy->actuator_velocity, policy.data_copy->actuator_velocity, model->nu);

  mju_copy(data_copy2->qpos, policy.data_copy2->qpos, model->nq);
  mju_copy(data_copy2->qvel, policy.data_copy2->qvel, model->nv);
  mju_copy(data_copy2->act,  policy.data_copy2->act,  model->na);
  mju_copy(data_copy2->actuator_length, policy.data_copy2->actuator_length, model->nu);
  mju_copy(data_copy2->actuator_velocity, policy.data_copy2->actuator_velocity, model->nu);
  
}

// copy parameters
void HierarchicalPDPolicy::SetPlan(const TimeSpline& plan) {
  this->plan = plan;
}

}  // namespace mjpc
