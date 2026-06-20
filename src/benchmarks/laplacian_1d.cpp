#include "benchmarks/laplacian_common.hpp"
#include "sparse_solver_gym/benchmark.hpp"
#include "sparse_solver_gym/perfetto_trace.hpp"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace {

constexpr int64_t laplacian_1d_n = 16;

void run_laplacian_1d_fp64(ssg::benchmark_context& context) {
  constexpr int64_t n = laplacian_1d_n;
  constexpr double tolerance = 1.0e-8;

  TRACE_EVENT("ssg.benchmark", "laplacian_1d_fp64.setup", "n", n);

  auto properties = ssg::benchmarks::make_laplacian_properties(n);

  auto problem = context.solver->make_sparse_problem(properties);
  auto graph = problem->make_graph();

  std::function<void(ssi::graph_count_builder_t&)> count_builder =
      [](ssi::graph_count_builder_t& builder) {
        for (int64_t row = builder.beg; row < builder.end; ++row) {
          int64_t degree = 1;
          if (row > 0) {
            ++degree;
          }
          if (row + 1 < builder.nrows) {
            ++degree;
          }
          builder.set_count(row, degree);
        }
      };

  std::function<void(ssi::graph_edge_builder_t&)> edge_builder =
      [](ssi::graph_edge_builder_t& builder) {
        for (int64_t row = builder.beg; row < builder.end; ++row) {
          int64_t edge = 0;
          if (row > 0) {
            builder.set_edge(row, edge++, row - 1);
          }
          builder.set_edge(row, edge++, row);
          if (row + 1 < builder.nrows) {
            builder.set_edge(row, edge++, row + 1);
          }
        }
      };

  graph->build_from_host(
      n, n, ssi::graph_orientation_t::row, count_builder, edge_builder);

  auto sparse_matrix = problem->make_sparse_matrix();
  std::function<void(ssi::sparse_value_builder_t&)> value_builder =
      [](ssi::sparse_value_builder_t& builder) {
        for (int64_t row = builder.beg; row < builder.end; ++row) {
          for (int64_t edge = 0; edge < builder.degree(row); ++edge) {
            const int64_t col = builder.edge_id(row, edge);
            const double value = col == row ? 2.0 : -1.0;
            builder.set_value<double>(row, edge, value);
          }
        }
      };
  sparse_matrix->build_from_host(
      ssi::dtype_t::fp64, ssi::graph_orientation_t::row, value_builder);
  problem->assert_properties(properties);

  auto rhs = context.solver->make_matrix(ssi::dtype_t::fp64);
  rhs->preallocate(n, 1);
  std::vector<double> expected(static_cast<std::size_t>(n), 0.0);
  std::vector<double> host_rhs(static_cast<std::size_t>(n), 0.0);
  for (int64_t row = 0; row < n; ++row) {
    expected[static_cast<std::size_t>(row)] = 1.0;
  }
  for (int64_t row = 0; row < n; ++row) {
    host_rhs[static_cast<std::size_t>(row)] =
        2.0 * expected[static_cast<std::size_t>(row)] -
        (row > 0 ? expected[static_cast<std::size_t>(row - 1)] : 0.0) -
        (row + 1 < n ? expected[static_cast<std::size_t>(row + 1)] : 0.0);
  }
  std::function<void(ssi::matrix_view_t&)> rhs_builder =
      [&](ssi::matrix_view_t& view) {
        for (int64_t row = view.rbeg; row < view.rend; ++row) {
          view.value_mut<double>(row, 0) = host_rhs[static_cast<std::size_t>(row)];
        }
      };
  rhs->build_from_host(rhs_builder);

  auto solution = context.solver->make_matrix(ssi::dtype_t::fp64);
  solution->preallocate(n, 1);

  auto symbolic = problem->make_symbolic_analysis();
  auto numeric = symbolic->make_numeric_factorization(sparse_matrix);
  ssg::benchmarks::require_successful_solve(
      "laplacian_1d_fp64", numeric->solve(*rhs, *solution));

  const auto metrics = ssg::benchmarks::validate_solution(
      expected,
      host_rhs,
      *solution,
      [n](int64_t row, const std::vector<double>& values) {
        return 2.0 * values[static_cast<std::size_t>(row)] -
               (row > 0 ? values[static_cast<std::size_t>(row - 1)] : 0.0) -
               (row + 1 < n ? values[static_cast<std::size_t>(row + 1)] : 0.0);
      });

  TRACE_EVENT(
      "ssg.benchmark",
      "laplacian_1d_fp64.verify",
      "relative_error_inf",
      metrics.relative_error_inf,
      "relative_residual_inf",
      metrics.relative_residual_inf);
  spdlog::info(
      "laplacian_1d_fp64 validation relative_error_inf={} relative_residual_inf={}",
      metrics.relative_error_inf,
      metrics.relative_residual_inf);
  if (metrics.relative_error_inf > tolerance || metrics.relative_residual_inf > tolerance) {
    throw std::runtime_error("laplacian_1d_fp64 validation failed");
  }
}

const ssg::benchmark_registration laplacian_1d_fp64_registration({
    "laplacian_1d_fp64",
    {"light", "correctness", "factorization", "fp64", "laplacian"},
    run_laplacian_1d_fp64,
    [] {
      return std::vector<ssi::sparse_problem_properties_t>{
          ssg::benchmarks::make_laplacian_properties(laplacian_1d_n)};
    },
});

}  // namespace
