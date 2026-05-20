#include "sparse_solver_gym/benchmark.hpp"
#include "sparse_solver_gym/perfetto_trace.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void run_laplacian_1d_fp64(ssg::benchmark_context& context) {
  constexpr int64_t n = 16;
  constexpr double tolerance = 1.0e-8;

  TRACE_EVENT("ssg.benchmark", "laplacian_1d_fp64.setup", "n", n);

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
  std::function<void(ssi::matrix_view_t&)> rhs_builder =
      [](ssi::matrix_view_t& view) {
        for (int64_t row = view.rbeg; row < view.rend; ++row) {
          const bool boundary = row == 0 || row + 1 == view.rend;
          view.value_mut<double>(row, 0) = boundary ? 1.0 : 0.0;
        }
      };
  rhs->build_from_host(rhs_builder);

  auto solution = context.solver->make_matrix(ssi::dtype_t::fp64);
  solution->preallocate(n, 1);

  auto symbolic = problem->make_symbolic_analysis();
  auto numeric = symbolic->make_numeric_factorization(sparse_matrix);
  numeric->solve(*rhs, *solution);

  double max_error = 0.0;
  std::function<void(const ssi::matrix_view_t&)> solution_reader =
      [&](const ssi::matrix_view_t& view) {
        for (int64_t row = view.rbeg; row < view.rend; ++row) {
          max_error = std::max(max_error, std::abs(view.value<double>(row, 0) - 1.0));
        }
      };
  solution->read_to_host(solution_reader);

  TRACE_EVENT("ssg.benchmark", "laplacian_1d_fp64.verify", "max_error", max_error);
  if (max_error > tolerance) {
    throw std::runtime_error("laplacian_1d_fp64 residual check failed");
  }
}

const ssg::benchmark_registration laplacian_1d_fp64_registration({
    "laplacian_1d_fp64",
    {"light", "correctness", "factorization", "fp64", "laplacian"},
    run_laplacian_1d_fp64,
});

}  // namespace
