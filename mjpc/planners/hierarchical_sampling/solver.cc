
// OSQP
#include <osqp.h>

// OsqpEigen
// #include <OsqpEigen/Compat.hpp>
#include <OsqpEigen/Constants.hpp>
#include <OsqpEigen/Data.hpp>
#include <OsqpEigen/Settings.hpp>
#include "mjpc/planners/hierarchical_sampling/solver.h"
#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>


namespace mjpc {
const Eigen::Matrix<c_float, -1, 1>& OsqpEigen::Solver::getSolution()
    {
        // copy data from an array to Eigen vector
        c_float* solution = getOSQPSolution()->x;
        // m_solution = Eigen::Map<Eigen::Matrix<c_float, -1, 1>>(solution, getData()->n, 1);

        return Eigen::Map<Eigen::Matrix<c_float, -1, 1>>(solution, getData()->n, 1);
    }
}  // namespace mjpc