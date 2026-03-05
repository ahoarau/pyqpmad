#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <qpmad/solver.h>
#include <stdexcept>

namespace nb = nanobind;

struct InequalityDualResult {
  Eigen::VectorXd dual;
  Eigen::Matrix<qpmad::MatrixIndex, Eigen::Dynamic, 1> indices;
  Eigen::Matrix<bool, Eigen::Dynamic, 1> is_lower;
};

NB_MODULE(pyqpmad, m) {
  nb::class_<InequalityDualResult>(m, "InequalityDualResult",
                                   "Active inequality dual variables returned "
                                   "by Solver.get_inequality_dual().")
      .def_ro("dual", &InequalityDualResult::dual,
              "Lagrange multiplier magnitude for each active inequality "
              "constraint.")
      .def_ro("indices", &InequalityDualResult::indices,
              "qpmad internal constraint index for each active constraint. "
              "Simple bounds (lb/ub) are indexed first (0..n-1), followed by "
              "general constraints (rows of A).")
      .def_ro("is_lower", &InequalityDualResult::is_lower,
              "Boolean mask: nonzero when the lower bound is active, zero when "
              "the upper bound is active.");

  nb::enum_<qpmad::SolverParameters::HessianType>(m, "HessianType")
      .value("UNDEFINED", qpmad::SolverParameters::UNDEFINED)
      .value("HESSIAN_LOWER_TRIANGULAR",
             qpmad::SolverParameters::HESSIAN_LOWER_TRIANGULAR)
      .value("HESSIAN_CHOLESKY_FACTOR",
             qpmad::SolverParameters::HESSIAN_CHOLESKY_FACTOR)
      .value("HESSIAN_INVERTED_CHOLESKY_FACTOR",
             qpmad::SolverParameters::HESSIAN_INVERTED_CHOLESKY_FACTOR)
      .export_values();

  nb::class_<qpmad::SolverParameters>(m, "SolverParameters",
                                      "Configuration parameters for Solver.")
      .def(nb::init<>())
      .def_rw("hessian_type", &qpmad::SolverParameters::hessian_type_,
              "Describes the form of the Hessian matrix H passed to solve(). "
              "Use HESSIAN_LOWER_TRIANGULAR if H is a plain positive-definite "
              "matrix (default). Use HESSIAN_CHOLESKY_FACTOR or "
              "HESSIAN_INVERTED_CHOLESKY_FACTOR to skip re-factorization when "
              "the same H is reused across solves.")
      .def_rw("tolerance", &qpmad::SolverParameters::tolerance_,
              "Feasibility and optimality tolerance.")
      .def_rw("max_iter", &qpmad::SolverParameters::max_iter_,
              "Maximum number of active-set iterations. Negative means "
              "unlimited.")
      .def_rw("return_inverted_cholesky_factor",
              &qpmad::SolverParameters::return_inverted_cholesky_factor_,
              "If True, H is overwritten with the inverted Cholesky factor "
              "after solve(), which can be passed back as "
              "HESSIAN_INVERTED_CHOLESKY_FACTOR on subsequent calls to avoid "
              "re-factorization.");

  nb::enum_<qpmad::Solver::ReturnStatus>(m, "ReturnStatus")
      .value("OK", qpmad::Solver::OK)
      .value("MAXIMAL_NUMBER_OF_ITERATIONS",
             qpmad::Solver::MAXIMAL_NUMBER_OF_ITERATIONS)
      .value("UNDEFINED", qpmad::Solver::UNDEFINED)
      .export_values();

  nb::class_<qpmad::Solver>(m, "Solver",
                            "Goldfarb-Idnani dual active-set QP solver.")
      .def(nb::init<>())
      .def("reserve", &qpmad::Solver::reserve, nb::arg("primal_size"),
           nb::arg("num_simple_bounds"), nb::arg("num_general_constraints"),
           "Pre-allocate internal workspace for a problem of the given size. "
           "Call this once before the first solve() to avoid allocations in "
           "the solve loop. primal_size is the number of decision variables, "
           "num_simple_bounds is the length of lb/ub (0 or primal_size), and "
           "num_general_constraints is the number of rows in A.")
      .def(
          "solve",
          [](qpmad::Solver &s, Eigen::Ref<Eigen::VectorXd> primal,
             Eigen::Ref<Eigen::MatrixXd> H, Eigen::Ref<const Eigen::VectorXd> h,
             std::optional<Eigen::Ref<const Eigen::VectorXd>> lb,
             std::optional<Eigen::Ref<const Eigen::VectorXd>> ub,
             std::optional<Eigen::Ref<const Eigen::MatrixXd>> A,
             std::optional<Eigen::Ref<const Eigen::VectorXd>> Alb,
             std::optional<Eigen::Ref<const Eigen::VectorXd>> Aub,
             const qpmad::SolverParameters &params) {
            if (lb.has_value() != ub.has_value()) {
              throw std::invalid_argument(
                  "lb and ub must both be set or both be None");
            }
            if (A.has_value() != (Alb.has_value() && Aub.has_value())) {
              throw std::invalid_argument(
                  "A, Alb and Aub must all be set or all be None");
            }
            if (lb.has_value() && A.has_value()) {
              return s.solve(primal, H, h, lb.value(), ub.value(), A.value(),
                             Alb.value(), Aub.value(), params);
            }
            if (lb.has_value()) {
              return s.solve(primal, H, h, lb.value(), ub.value(), params);
            }
            if (A.has_value()) {
              return s.solve(primal, H, h, A.value(), Alb.value(), Aub.value(),
                             params);
            }
            // unconstrained: pass zero-size bounds (qpmad accepts lb.rows()==0)
            static const Eigen::VectorXd empty_v;
            return s.solve(primal, H, h, empty_v, empty_v, params);
          },
          nb::arg("primal"), nb::arg("H"), nb::arg("h"),
          nb::arg("lb") = nb::none(), nb::arg("ub") = nb::none(),
          nb::arg("A") = nb::none(), nb::arg("Alb") = nb::none(),
          nb::arg("Aub") = nb::none(),
          nb::arg("params") = qpmad::SolverParameters(),
          "Solve the QP: min 0.5 x'Hx + h'x "
          "s.t. lb <= x <= ub, Alb <= A x <= Aub.\n\n"
          "primal: decision variable vector (n,), modified in place with the "
          "solution.\n"
          "H: positive-definite Hessian matrix (n, n), modified in place "
          "during factorization.\n"
          "h: linear cost vector (n,).\n"
          "lb: lower simple bounds on x (n,), or None for no box constraints.\n"
          "ub: upper simple bounds on x (n,), or None for no box constraints.\n"
          "A: general constraint matrix (m, n), or None.\n"
          "Alb: lower bounds on A x (m,), or None.\n"
          "Aub: upper bounds on A x (m,), or None.\n"
          "params: SolverParameters instance.\n\n"
          "Returns a ReturnStatus value (OK on success).")
      .def(
          "get_inequality_dual",
          [](const qpmad::Solver &s) {
            InequalityDualResult result;
            s.getInequalityDual(result.dual, result.indices, result.is_lower);
            return result;
          },
          "Return the active inequality dual after a solve. "
          "dual[i] is the Lagrange multiplier magnitude, "
          "indices[i] is the qpmad internal constraint index (simple bounds "
          "first, then general constraints), and is_lower[i] is nonzero when "
          "the lower bound is active.")
      .def(
          "get_num_iterations",
          [](const qpmad::Solver &s) {
            return s.getNumberOfInequalityIterations();
          },
          "Return the number of inequality iterations from the last solve.");
}
