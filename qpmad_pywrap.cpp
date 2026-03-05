#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <qpmad/solver.h>

namespace nb = nanobind;

NB_MODULE(pyqpmad, m) {
  nb::enum_<qpmad::SolverParameters::HessianType>(m, "HessianType")
      .value("UNDEFINED", qpmad::SolverParameters::UNDEFINED)
      .value("HESSIAN_LOWER_TRIANGULAR",
             qpmad::SolverParameters::HESSIAN_LOWER_TRIANGULAR)
      .value("HESSIAN_CHOLESKY_FACTOR",
             qpmad::SolverParameters::HESSIAN_CHOLESKY_FACTOR)
      .value("HESSIAN_INVERTED_CHOLESKY_FACTOR",
             qpmad::SolverParameters::HESSIAN_INVERTED_CHOLESKY_FACTOR)
      .export_values();

  nb::class_<qpmad::SolverParameters>(m, "SolverParameters")
      .def(nb::init<>())
      .def_rw("hessian_type", &qpmad::SolverParameters::hessian_type_)
      .def_rw("tolerance", &qpmad::SolverParameters::tolerance_)
      .def_rw("max_iter", &qpmad::SolverParameters::max_iter_)
      .def_rw("return_inverted_cholesky_factor",
              &qpmad::SolverParameters::return_inverted_cholesky_factor_);

  nb::enum_<qpmad::Solver::ReturnStatus>(m, "ReturnStatus")
      .value("OK", qpmad::Solver::OK)
      .value("MAXIMAL_NUMBER_OF_ITERATIONS",
             qpmad::Solver::MAXIMAL_NUMBER_OF_ITERATIONS)
      .value("UNDEFINED", qpmad::Solver::UNDEFINED)
      .export_values();

  nb::class_<qpmad::Solver>(m, "Solver")
      .def(nb::init<>())
      .def(
          "solve",
          [](qpmad::Solver &s, Eigen::Ref<Eigen::VectorXd> primal,
             Eigen::Ref<Eigen::MatrixXd> H,
             const Eigen::Ref<const Eigen::VectorXd> &h,
             const std::optional<Eigen::Ref<const Eigen::VectorXd>> &lb,
             const std::optional<Eigen::Ref<const Eigen::VectorXd>> &ub,
             const std::optional<Eigen::Ref<const Eigen::MatrixXd>> &A,
             const std::optional<Eigen::Ref<const Eigen::VectorXd>> &Alb,
             const std::optional<Eigen::Ref<const Eigen::VectorXd>> &Aub,
             const std::optional<qpmad::SolverParameters> &params) {
            Eigen::VectorXd ev;
            Eigen::MatrixXd em;
            qpmad::SolverParameters default_params;
            const qpmad::SolverParameters &params_ref =
                params ? *params : default_params;

            return s.solve(primal, H, h, 
                           lb ? *lb : (Eigen::Ref<const Eigen::VectorXd>)ev,
                           ub ? *ub : (Eigen::Ref<const Eigen::VectorXd>)ev,
                           A ? *A : (Eigen::Ref<const Eigen::MatrixXd>)em,
                           Alb ? *Alb : (Eigen::Ref<const Eigen::VectorXd>)ev,
                           Aub ? *Aub : (Eigen::Ref<const Eigen::VectorXd>)ev,
                           params_ref);
          },
          nb::arg("primal").noconvert(), nb::arg("H").noconvert(), nb::arg("h"),
          nb::arg("lb") = nb::none(), nb::arg("ub") = nb::none(),
          nb::arg("A") = nb::none(), nb::arg("Alb") = nb::none(),
          nb::arg("Aub") = nb::none(), nb::arg("params") = nb::none());
}
