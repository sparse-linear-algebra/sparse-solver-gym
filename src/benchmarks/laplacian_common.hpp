#ifndef SPARSE_SOLVER_GYM_BENCHMARKS_LAPLACIAN_COMMON_HPP
#define SPARSE_SOLVER_GYM_BENCHMARKS_LAPLACIAN_COMMON_HPP

#include "sparse_solver_interface.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ssg::benchmarks {

struct validation_metrics {
  double relative_error_inf = 0.0;
  double relative_residual_inf = 0.0;
};

inline double exact_value(int64_t row, int64_t rhs_index) {
  return 1.0 + 0.01 * static_cast<double>((row + 17 * rhs_index) % 23);
}

inline validation_metrics validate_solution(
    const std::vector<double>& expected,
    const std::vector<double>& rhs,
    const ssi::matrix_t& solution,
    const std::function<double(int64_t, const std::vector<double>&)>& apply_row) {
  std::vector<double> computed(expected.size(), 0.0);
  std::function<void(const ssi::matrix_view_t&)> solution_reader =
      [&](const ssi::matrix_view_t& view) {
        for (int64_t row = view.rbeg; row < view.rend; ++row) {
          computed[static_cast<std::size_t>(row)] = view.value<double>(row, 0);
        }
      };
  solution.read_to_host(solution_reader);

  double error_inf = 0.0;
  double expected_inf = 0.0;
  double residual_inf = 0.0;
  double rhs_inf = 0.0;
  for (int64_t row = 0; row < static_cast<int64_t>(expected.size()); ++row) {
    const auto index = static_cast<std::size_t>(row);
    error_inf = std::max(error_inf, std::abs(computed[index] - expected[index]));
    expected_inf = std::max(expected_inf, std::abs(expected[index]));

    const double residual = rhs[index] - apply_row(row, computed);
    residual_inf = std::max(residual_inf, std::abs(residual));
    rhs_inf = std::max(rhs_inf, std::abs(rhs[index]));
  }

  return {
      error_inf / std::max(expected_inf, 1.0),
      residual_inf / std::max(rhs_inf, 1.0),
  };
}

inline ssi::sparse_problem_properties_t make_laplacian_properties(int64_t n) {
  ssi::sparse_problem_properties_t properties;
  properties.nrows = n;
  properties.ncols = n;
  properties.orientation = ssi::graph_orientation_t::row;
  properties.itype = ssi::itype_t::i64;
  properties.dtype = ssi::dtype_t::fp64;
  properties.structurally_symmetric = ssi::property_state_t::known_true;
  properties.numerically_symmetric = ssi::property_state_t::known_true;
  properties.positive_definite = ssi::property_state_t::known_true;
  properties.nonsingular = ssi::property_state_t::known_true;
  properties.strong_hall = ssi::property_state_t::known_true;
  properties.symmetric_storage = ssi::symmetric_storage_t::full;
  return properties;
}

inline const char* status_name(ssi::status_t status) {
  switch (status) {
    case ssi::status_t::ok:
      return "ok";
    case ssi::status_t::invalid_argument:
      return "invalid_argument";
    case ssi::status_t::out_of_range:
      return "out_of_range";
    case ssi::status_t::unsupported:
      return "unsupported";
    case ssi::status_t::singular:
      return "singular";
    case ssi::status_t::rank_deficient:
      return "rank_deficient";
    case ssi::status_t::indefinite:
      return "indefinite";
    case ssi::status_t::zero_pivot:
      return "zero_pivot";
    case ssi::status_t::breakdown:
      return "breakdown";
    case ssi::status_t::not_converged:
      return "not_converged";
    case ssi::status_t::exception:
      return "exception";
  }
  return "unknown";
}

inline std::string solve_result_message(
    std::string_view benchmark_name,
    const ssi::solve_result_t& result) {
  std::string message = std::string(benchmark_name) + " solve returned " +
                        status_name(result.status);
  if (!result.reason.empty()) {
    message += ": " + result.reason;
  }
  return message;
}

inline void require_successful_solve(
    std::string_view benchmark_name,
    const ssi::solve_result_t& result) {
  if (!result.success()) {
    const auto message = solve_result_message(benchmark_name, result);
    if (result.status == ssi::status_t::unsupported) {
      throw ssi::unsupported_error_t(message);
    }
    throw ssi::numeric_error_t(result.status, message);
  }
  if (!result.converged) {
    throw ssi::not_converged_error_t(
        std::string(benchmark_name) + " solve did not converge");
  }
}

}  // namespace ssg::benchmarks

#endif
