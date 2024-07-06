#include <OsqpEigen/OsqpEigen.h>
// OSQP
#include <osqp.h>

// OsqpEigen
// #include <OsqpEigen/Compat.hpp>
#include <OsqpEigen/Constants.hpp>
#include <OsqpEigen/Data.hpp>
#include <OsqpEigen/Settings.hpp>
#include <Eigen/Dense>

#ifndef MJPC_PLANNERS_QP_SOLVER_H_
#define MJPC_PLANNERS_QP_SOLVER_H_

namespace mjpc {

// policy for sampling planner
class MySolver : public OsqpEigen::Solver {
 public:

  inline const OSQPSolution* getOSQPSolution() const noexcept
  const Eigen::Matrix<c_float, -1, 1>& getSolution();

//   const Eigen::Matrix<c_float, -1, 1>& OsqpEigen::Solver::getSolution()
//     {
//         // copy data from an array to Eigen vector
//         c_float* solution = getOSQPSolution()->x;
//         m_solution = Eigen::Map<Eigen::Matrix<c_float, -1, 1>>(solution, getData()->n, 1);

//         return m_solution;
//     }

  

};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_SAMPLING_POLICY_H_
