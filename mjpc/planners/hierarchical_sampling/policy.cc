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
#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>

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
  // HighToLowAction(&high_level_action[0], action, );
  // // Clamp controls
  Clamp(action, model->actuator_ctrlrange, model->nu);
}

void HierarchicalSamplingPolicy::HierarchicalAction(double* action, mjData* data) const {
  CHECK(action != nullptr);
  
  //TODO: sample high-level action and then convert to low-level action
  std::vector<double> high_level_action(dim_high_level_action);
  plan.Sample(data->time, absl::MakeSpan(&high_level_action[0], dim_high_level_action));
  HighToLowAction(&high_level_action[0], action, data);
  // Clamp controls
  Clamp(action, model->actuator_ctrlrange, model->nu);
}

void HierarchicalSamplingPolicy::HighToLowAction(double* high_level_action, double* action, mjData* data) const {
  CHECK(action != nullptr);
  // std::cout<<"model para"<<model->nq<<model->nv<<std::endl;
  // for (int k = 0; k < model->nq; k++) {
  //   std::cout<<"before joint "<<k<<": ";
  //   // std::cout<<temp_qpos[k]<<" "<<high_level_action[k]<<" ";

  //   std::cout<<model->jnt_range[k*2]<<" "<<model->jnt_range[k*2+1]<<std::endl;
  //   }
  // std::cout<<std::endl;
  absl::BitGen gen_;
  // for (int k = 0; k < dim_high_level_action; k++) {
  //     double noise = absl::Gaussian<double>(gen_, 0.0, 0.005);
  //     high_level_action[k] = noise;
  //   }
  Clamp(high_level_action, model->jnt_range, model->nq);

  // mjModel* model_copy = mj_copyModel(NULL, model);
  //test for qfrc: pass
  // mjData* data_copy = mj_copyData(NULL, model_copy, data);
  // double* temp_qpos = data_copy->qpos;
  // double jnt_err = 0;
  // for (int k = 0; k < model->nq; k++) {
  //   // std::cout<<"before joint "<<k<<": ";
  //   // std::cout<<temp_qpos[k]<<" "<<high_level_action[k]<<" "<<model->jnt_range[k*2]<<" "<<model->jnt_range[k*2+1]<<std::endl;

  //   // std::cout<<temp_qpos[k]-high_level_action[k]<<" ";
  //   jnt_err += mju_abs(temp_qpos[k]-high_level_action[k]);
  //   }
  // std::cout<<"before joint "<<jnt_err<<std::endl;
  

  // for (int k = 0; k < 5; k++) {
  //   double* qfrc_inverse = get_qfrc(model_copy, data_copy, high_level_action);
  //   data_copy->qfrc_applied = qfrc_inverse;
  //   // std::cout<<"before time "<<data_copy->time<<std::endl; 
  //   mj_step(model_copy, data_copy);
  //   // std::cout<<"after time "<<data_copy->time<<" "<<model->opt.timestep<<std::endl; 
  //   double* temp_qpos = data_copy->qpos;
  //   // std::cout<<"iteration "<<k<<std::endl;
  //   jnt_err = 0;
  //   for (int j = 0; j < model->nq; j++) {
  //     // std::cout<<"joint "<<j<<": ";
  //     // std::cout<<temp_qpos[j]<<" "<<high_level_action[j]<<std::endl;
  //     jnt_err += mju_abs(temp_qpos[k]-high_level_action[k]);
  //     }
  //   std::cout<<"iteration "<<k<< " joint "<<jnt_err<<std::endl;
  //   }
    
  
  // // for (int k = 0; k < model->nv; k++) {
  // //   std::cout<<qfrc_inverse[k]<<std::endl;
  // //   }

  //test QP
  // solve_qp();

  //test ctrl
  // get_qfrc(model_copy, data_copy, high_level_action);
  // std::cout<<"qfrc_inverse "<< qfrc_inverse[0] <<std::endl;
  // get_ctrl(model_copy, data, high_level_action, qfrc_inverse);


  // double* qfrc_inverse = get_qfrc(model_copy, data_copy, high_level_action);
  // // std::cout<<"qfrc_inverse "<< qfrc_inverse[0] <<std::endl;
  // double* ctrl = get_ctrl(model_copy, data, high_level_action, qfrc_inverse);

  // mju_error_i(
  //       "run end here", 0
  //       );
  
  

  //add random noise for test
  // sampling token
  // absl::BitGen gen_;
  // for (int k = 0; k < model->nu; k++) {
  //     action[k] = ctrl[k];
  //   }
  Clamp(action, model->actuator_ctrlrange, model->nu);
}


// // Solve a quadratic program
Eigen::VectorXd HierarchicalSamplingPolicy::solve_qp(
    Eigen::SparseMatrix<double> hessian, 
  Eigen::VectorXd gradient, 
  Eigen::VectorXd lowerBound,
  Eigen::VectorXd upperBound,
  Eigen::VectorXd initialGuess
) const{

  //test QP
  // allocate QP problem matrices and vectores
  // Eigen::SparseMatrix<double> hessian2(2, 2);      //P: n*n正定矩阵,必须为稀疏矩阵SparseMatrix
  // Eigen::VectorXd gradient2(2);                    //Q: n*1向量
  


  
  // Eigen::VectorXd lowerBound2(2);                  //L: m*1下限向量
  // Eigen::VectorXd upperBound2(2);                  //U: m*1上限向量
  // lowerBound2 << 0, 0;
  // upperBound2 << 1.5, 1.5;
  int m = lowerBound.size();
  int n = gradient.size();
  // std::cout<<"m "<<m<<" n "<<n<<std::endl;
  
  // // std::cout << "size" << hessian.rows() << "x" << hessian.cols() << " " << lowerBound.size() << std::endl;
  // hessian2.insert(0, 0) = 2.0; //注意稀疏矩阵的初始化方式,无法使用<<初始化
  // hessian2.insert(1, 1) = 2.0;
  // // std::cout << "hessian:" << std::endl
  // //           << hessian << std::endl;
  // gradient2 << -2, -2;

  Eigen::SparseMatrix<double> linearMatrix(m, n); //A: m*n矩阵,必须为稀疏矩阵SparseMatrix
  for (int i = 0; i < n; i++) {
    linearMatrix.insert(i, i) = 1.0;
  }
  // linearMatrix.insert(0, 0) = 1.0; //注意稀疏矩阵的初始化方式,无法使用<<初始化
  // linearMatrix.insert(1, 1) = 1.0;
  // std::cout << "linearMatrix:" << std::endl
  //           << linearMatrix << std::endl;
  

  // instantiate the solver
  OsqpEigen::Solver* solver = new OsqpEigen::Solver();

  // solver.settings()->setVerbosity(true);
  // solver.settings()->setWarmStart(true);

  // settings
  solver->settings()->setVerbosity(false);
  solver->settings()->setWarmStart(true);
  
  // set the initial data of the QP solver
  solver->data()->setNumberOfVariables(n);   //变量数n
  solver->data()->setNumberOfConstraints(m); //约束数m
  


  if(!solver->data()->setHessianMatrix(hessian)){
        std::cout << "H matrix error!" << std::endl;
    } 
  if(!solver->data()->setGradient(gradient)){
      std::cout << "f matrix error!" << std::endl;
  }
  if(!solver->data()->setLinearConstraintsMatrix(linearMatrix)){
      std::cout << "A matrix error!" << std::endl;
  }
  if(!solver->data()->setLowerBound(lowerBound)){
      std::cout << "lb matrix error!" << std::endl;
  }
  if(!solver->data()->setUpperBound(upperBound)){
      std::cout << "ub matrix error!" << std::endl;
  }


  // initial the solver
  if(!solver->initSolver()){
      std::cout << "solver initial error!" << std::endl;
  }

  // solver.data()->setHessianMatrix(hessian2);
  // solver.data()->setGradient(gradient2);
  // solver.data()->setLinearConstraintsMatrix(linearMatrix);
  // solver.data()->setLowerBound(lowerBound2);
  // solver.data()->setUpperBound(upperBound2);

  // solver.initSolver();
  solver->setPrimalVariable(initialGuess);
  
  // std::cout << "begin res " << initialGuess(0, 0) << std::endl;
  // Eigen::VectorXd QPSolution;
  // try {
  //   if (solver.solveProblem()!= OsqpEigen::ErrorExitFlag::NoError) {
  //     std::cout << "QP not solve" << std::endl;
  //     QPSolution = Eigen::VectorXd::Zero(n);
  //   }
  //   else {
  //     std::cout << "QP solved" << std::endl;
  //     QPSolution = solver.getSolution();
  //   }
  // }
  // catch (...) {
  //   std::cout << "QP not solve err" << std::endl;
  // }
  
  solver->solveProblem();
  // std::cout<<"solve res "<< solveres << std::endl;  
  Eigen::VectorXd QPSolution = solver->getSolution();
  // solver->getSolution();
  // solver.updateBounds(lowerBound, upperBound)
  // Eigen::VectorXd QPSolution = solver.m_solution;
  
  solver->clearSolverVariables();
  // solver.initSolver();
  // OsqpEigen::Solver::OSQPSolverDeleter(solver);
  // TODO: find a better way to delete solver
  solver = new OsqpEigen::Solver();
  // std::cout << "QPSolution " << QPSolution.size() << std::endl; 
  // delete solver;
  // solver.~Solver();
  // std::cout<<"before return"<<std::endl;
  return QPSolution;
}

// get qfrc
double* HierarchicalSamplingPolicy::get_qfrc(mjModel* model, mjData* data, double* target_qpos) const{
// Compute the generalized force needed to reach the target position in the next mujoco step
  // std::cout<<"begin get_qfrc"<<std::endl;
  mjData* data_copy = mj_copyData(NULL, model, data);

  // mjtNum* target_qacc = nullptr;
  mjtNum* target_qacc = (mjtNum*) mju_malloc(model->nq * sizeof(mjtNum));
  // std::cout<<"copy acc"<<std::endl;
  mju_copy(target_qacc, target_qpos, model->nq);

  
  // memcpy(&target_qacc[0], &target_qpos[0], model->nq * sizeof(double));
  //velocity
  // std::cout<<"get acc"<<std::endl;
  mju_subFrom(target_qacc, data_copy->qpos, model->nq);
  // for (int k = 0; k < model->nv; k++) {
  //   std::cout<<"joint "<<k<<": ";
  //   std::cout<<target_qacc[k]<<" ";
  //   }
  // std::cout<<std::endl;
  mju_scl(target_qacc, target_qacc, 1/model->opt.timestep, model->nq);
  // for (int k = 0; k < model->nv; k++) {
  //   std::cout<<"joint "<<k<<": ";
  //   std::cout<<target_qacc[k]<<" ";
  //   }
  // std::cout<<std::endl;
  
  //acc
  mju_subFrom(target_qacc, data_copy->qvel, model->nq);
  mju_scl(target_qacc, target_qacc, 1/model->opt.timestep, model->nq);
  
  // for (int k = 0; k < model->nv; k++) {
  //   std::cout<<"joint "<<k<<": ";
  //   std::cout<<target_qacc[k]<<" ";
  //   }
  // std::cout<<std::endl;
  data_copy->qacc = target_qacc;
  //inverse dynamics
  // std::cout<<"begin inverse dynamics"<<std::endl;
  model->opt.disableflags += mjDSBL_CONSTRAINT;
  mj_inverse(model, data_copy);
  model->opt.disableflags -= mjDSBL_CONSTRAINT;
  double* qfrc_inverse = data_copy->qfrc_inverse;

  // for (int k = 0; k < model->nv; k++) {
  //   std::cout<<"joint force "<<k<<": ";
  //   std::cout<<qfrc_inverse[k]<<" ";
  //   }
  // std::cout<<std::endl;
  
  return qfrc_inverse;
}

// get control
double* HierarchicalSamplingPolicy::get_ctrl(mjModel* model, mjData* data, double* target_qpos, double* qfrc) const{
  double qfrc_scaler = 100;
  double qvel_scaler = 5;

  // Compute the control needed to reach the target position in the next mujoco step
  Eigen::VectorXd act_vec = Eigen::Map<Eigen::VectorXd>(data->act, model->nu);
  Eigen::VectorXd ctrl0 = Eigen::Map<Eigen::VectorXd>(data->ctrl, model->nu);
  // std::cout<<"act "<< act <<std::endl;
  // std::cout<<"ctrl0 "<< ctrl0 <<std::endl;

  double ts = model->opt.timestep;
  Eigen::VectorXd tA = 0.01 * (0.5 + 1.5 * act_vec.array());
  Eigen::VectorXd tD = 0.04 / (0.5 + 1.5 * act_vec.array());
  Eigen::VectorXd tausmooth = 5 * Eigen::VectorXd::Ones(model->nu);
  Eigen::VectorXd t1 = ((tA - tD) * 1.875).array() / tausmooth.array();
  Eigen::VectorXd t2 = (tA + tD) * 0.5;

  // ---- gain, bias, and moment computation
  mjData* data_copy = mj_copyData(NULL, model, data);
  data_copy->qpos = target_qpos;
  mjtNum* qvel = (mjtNum*) mju_malloc(model->nq * sizeof(mjtNum));
  mju_copy(qvel, target_qpos, model->nq);
  mju_subFrom(qvel, data->qpos, model->nq);
  mju_scl(qvel, qvel, 1/model->opt.timestep, model->nq);
  mju_scl(qvel, qvel, 1/qvel_scaler, model->nq);
  data_copy->qvel= qvel; 
  mj_step1(model, data_copy); // gain, bias, and moment depend on qpos and qvel
  Eigen::VectorXd gain(model->nu);
  Eigen::VectorXd bias(model->nu);
  for (int idx_actuator = 0; idx_actuator < model->nu; ++idx_actuator) {
    double length = data_copy->actuator_length[idx_actuator];
    mjtNum* lengthange = (mjtNum*) mju_malloc(2 * sizeof(mjtNum));
    lengthange[0] = model->actuator_lengthrange[2*idx_actuator];
    lengthange[1] = model->actuator_lengthrange[2*idx_actuator+1];
    double velocity = data_copy->actuator_velocity[idx_actuator];
    double acc0 = model->actuator_acc0[idx_actuator];
    mjtNum* prmb = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
    for (int j = 0; j<9; j++) {
      prmb[j] = model->actuator_biasprm[10*idx_actuator+j];
    // std::cout<<"biasprm "<<j<<" "<<prmb[j]<<std::endl;
    }
    // std::cout<<std::endl;
    mjtNum* prmg = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
    for (int j = 0; j<9; j++) {
      prmg[j] = model->actuator_gainprm[10*idx_actuator+j];
    // std::cout<<"gainprm "<<j<<" "<<prmg[j]<<std::endl;
    }
    // std::cout<<std::endl;
    bias[idx_actuator] = mju_muscleBias(length, lengthange, acc0, prmb);
    gain[idx_actuator] = std::min(-1.0, mju_muscleGain(length, velocity, lengthange, acc0, prmg));
  }

  // std::cout<<"bias "<< bias <<std::endl;
  // std::cout<<"gain "<< gain <<std::endl;



  Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data_copy->actuator_moment, model->nv, model->nu);
  // ---- ctrl computation
  Eigen::MatrixXd P = 2 * AM.transpose() * AM;

  Eigen::VectorXd qfrc_vec = Eigen::Map<Eigen::VectorXd>(qfrc, model->nv);
  // std::cout<<"qfrc_vec "<<qfrc_vec<<std::endl;
  // std::cout<<"AM shape "<<AM.rows()<<" "<<AM.cols()<<std::endl;
  // std::cout<<"vac shape "<<(gain.array() * act_vec.array()).rows()<<" "<<(gain.array() * act_vec.array()).cols()<<std::endl;

  // std::cout<<(gain.array() * act_vec.array()).size()<<std::endl;
  Eigen::VectorXd gain_act = (gain.array() * act_vec.array());
  Eigen::VectorXd k =  AM * gain_act + AM * bias - (qfrc_vec/qfrc_scaler);
  Eigen::VectorXd q = 2 * k.transpose() * AM;
  Eigen::VectorXd lb = gain.array() * (1 - act_vec.array()) * ts / (t2.array() + t1.array() * (1 - act_vec.array()));
  Eigen::VectorXd ub = - gain.array() * act_vec.array() * ts / (t2.array() - t1.array() * act_vec.array());
  Eigen::VectorXd x0 = (gain.array() * (ctrl0.array() - act_vec.array()) * ts) / ((ctrl0.array() - act_vec.array()) * t1.array() + t2.array());
  
  Eigen::SparseMatrix<double> P_sparse = P.sparseView();
  
  // Eigen::VectorXd x = solve_qp(P_sparse, q, lb, ub, x0);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(model->nu); 
  
  Eigen::VectorXd ctrl_vec = act_vec.array() + x.array() * t2.array() / ((gain.array() * ts) - x.array() * t1.array());
  // for (int i = 0; i < ctrl.size(); ++i) {
  // ctrl[i] = std::clamp(ctrl[i], 0.0, 1.0);
  // }
  // return ctrl;

  // std::cout<<AM.size()<<P.size()<<q.size()<<lb.size()<<x.size()<<ctrl2.size()<<std::endl;

  double* ctrl = ctrl_vec.data();
  // for (int i = 0; i < model->nu; i++) {
  //   std::cout<<"ctrl "<<i<<" "<<ctrl[i]<<std::endl;
  // }
  return ctrl;
}


// double* HierarchicalSamplingPolicy::get_ctrl(mjModel* model, mjData* data, double* target_qpos, double* qfrc) const{
//   double* act = data->act;
//   // double* ctrl0 = data->ctrl;
//   // double ts = model->opt.timestep;
//   // double qfrc_scaler = 100;
//   double qvel_scaler = 5;

//   mjtNum* tA = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mjtNum* tD = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mjtNum* t1 = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mjtNum* t2 = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   double tau_smooth = 5;
//   mjtNum* t_base = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   //active
//   mju_zero(t_base, model->nu);
//   mju_copy(t_base, act, model->nu);
//   mju_scl(t_base, t_base, 1.5, model->nu);
//   for (int i = 0; i < model->nu; i++) {
//     t_base[i] += 0.5;
//   }

//   mju_copy(tA, t_base, model->nu);
//   mju_scl(tA, tA, 0.01, model->nu);
//   mju_zero(tD, model->nu);
//   for (int i = 0; i < model->nu; i++) {
//     tD[i] = 0.04/t_base[i];
//   }
  
//   mju_copy(t1, tA, model->nu);
//   mju_subFrom(t1, tD, model->nu);
//   mju_scl(t1, t1, 1.875, model->nu);
//   mju_scl(t1, t1, 1/tau_smooth, model->nu);

//   mju_copy(t2, tA, model->nu);
//   mju_addTo(t2, tD, model->nu);
//   mju_scl(t2, t2, 0.5, model->nu);

//   // gain, bias, moment computation
//   mjData* data_copy = mj_copyData(NULL, model, data);
//   data_copy->qpos = target_qpos;
//   mjtNum* qvel = (mjtNum*) mju_malloc(model->nq * sizeof(mjtNum));
//   mju_copy(qvel, target_qpos, model->nq);
//   mju_subFrom(qvel, data->qpos, model->nq);
//   mju_scl(qvel, qvel, 1/model->opt.timestep, model->nq);
//   mju_scl(qvel, qvel, 1/qvel_scaler, model->nq);
//   data_copy->qvel= qvel; 
//   mj_step1(model, data_copy);

//   mjtNum* gain = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mjtNum* bias = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));
//   mju_zero(gain, model->nu);
//   mju_zero(bias, model->nu);

//   for (int i = 0; i < model->nu; i++) {
//     double length = data_copy->actuator_length[i];
//     mjtNum* lengthange = (mjtNum*) mju_malloc(2 * sizeof(mjtNum));
//     lengthange[0] = model->actuator_lengthrange[2*i];
//     lengthange[1] = model->actuator_lengthrange[2*i+1];
//     double velocity = data_copy->actuator_velocity[i];
//     double acc0 = model->actuator_acc0[i];
//     mjtNum* prmb = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
//     for (int j = 0; j<9; j++) {
//       prmb[j] = model->actuator_biasprm[10*i+j];
//     // std::cout<<"biasprm "<<j<<" "<<prmb[j]<<std::endl;
//     }
//     // std::cout<<std::endl;
//     mjtNum* prmg = (mjtNum*) mju_malloc(9 * sizeof(mjtNum));
//     for (int j = 0; j<9; j++) {
//       prmg[j] = model->actuator_gainprm[10*i+j];
//     // std::cout<<"gainprm "<<j<<" "<<prmg[j]<<std::endl;
//     }
//     // std::cout<<std::endl;

//     bias[i] = mju_muscleBias(length, lengthange, acc0, prmb);
//     gain[i] = mju_muscleGain(length, velocity, lengthange, acc0, prmg);

//     // std::cout<<"muscle "<<i<<" "<<bias[i]<<" "<<gain[i]<<std::endl;
//     // break;
//   }

//   // Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data_copy->actuator_moment, model->nu, model->nu);

//   //matrixxd fill cols first, therefore the construction shape should be nv x nu 
//   Eigen::MatrixXd AM = Eigen::Map<Eigen::MatrixXd>(data_copy->actuator_moment, model->nv, model->nu);

//   // Eigen::SparseMatrix<double> AM_sparse = AM.sparseView();
//   // for (int i = 0; i<9; i++) {
//   //   std::cout<<"vec "<<i<<" "<<data_copy->actuator_moment[i]<<std::endl;
//   // }
//   // std::cout<<"AM "<<AM<<std::endl;
//   // std::cout<<"AM "<<AM_sparse<<std::endl;
//   // std::cout<<"AM: "<<AM(100, 0)<<" "<<AM(0, 100)<<std::endl;
//   // Eigen::MatrixXd P = 2*AM.transpose()*AM;
//   // std::cout<<"P: "<<P(100, 0)<<" "<<P(0, 100)<<std::endl;
//   mju_error_i(
//         "ctrl run end here", 0
//         );

//   // ---- ctrl computation
//   Eigen::MatrixXd P = 2 * AM.transpose() * AM;

//   // mjtNum* k_gain = (mjtNum*) mju_malloc(model->nu * sizeof(mjtNum));

  
//   Eigen::VectorXd k = AM * (gain.array() * act.array()) + AM * bias - (qfrc.array() / qfrc_scaler.array());
//   Eigen::VectorXd q = 2 * k.transpose() * AM;
//   Eigen::VectorXd lb = gain.array() * (1 - act.array()) * ts / (t2.array() + t1.array() * (1 - act.array()));
//   Eigen::VectorXd ub = - gain.array() * act.array() * ts / (t2.array() - t1.array() * act.array());
//   Eigen::VectorXd x0 = (gain.array() * (ctrl0.array() - act.array()) * ts) / ((ctrl0.array() - act.array()) * t1.array() + t2.array());
//   Eigen::VectorXd x = solve_qp(P, q, lb, ub, x0);
//   Eigen::VectorXd ctrl = act + x.array() * t2.array() / (gain.array() * ts - x.array() * t1.array());
//   for (int i = 0; i < ctrl.size(); ++i) {
//   ctrl[i] = std::clamp(ctrl[i], 0.0, 1.0);
//   }



//   double* ctrl = nullptr;


//   return ctrl;
// }

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
