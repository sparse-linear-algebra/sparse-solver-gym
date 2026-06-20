#include "benchmarks/laplacian_common.hpp"
#include "sparse_solver_gym/benchmark.hpp"
#include "sparse_solver_gym/perfetto_trace.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace {

constexpr int64_t streaming_rhs_count = 100;

struct laplacian_3d {
  int64_t mx;
  int64_t my;
  int64_t mz;

  int64_t n() const {
    return mx * my * mz;
  }

  int64_t index(int64_t x, int64_t y, int64_t z) const {
    return x + mx * (y + my * z);
  }

  void coordinates(int64_t row, int64_t& x, int64_t& y, int64_t& z) const {
    x = row % mx;
    const int64_t yz = row / mx;
    y = yz % my;
    z = yz / my;
  }

  int64_t degree(int64_t row) const {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
    coordinates(row, x, y, z);
    return 1 + (x > 0) + (x + 1 < mx) + (y > 0) + (y + 1 < my) + (z > 0) +
           (z + 1 < mz);
  }

  template <typename Fn>
  void for_each_neighbor(int64_t row, Fn&& fn) const {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
    coordinates(row, x, y, z);
    if (x > 0) {
      fn(index(x - 1, y, z));
    }
    if (x + 1 < mx) {
      fn(index(x + 1, y, z));
    }
    if (y > 0) {
      fn(index(x, y - 1, z));
    }
    if (y + 1 < my) {
      fn(index(x, y + 1, z));
    }
    if (z > 0) {
      fn(index(x, y, z - 1));
    }
    if (z + 1 < mz) {
      fn(index(x, y, z + 1));
    }
  }

  double apply_row(int64_t row, const std::vector<double>& values) const {
    double result = 6.0 * values[static_cast<std::size_t>(row)];
    for_each_neighbor(row, [&](int64_t col) {
      result -= values[static_cast<std::size_t>(col)];
    });
    return result;
  }
};

constexpr laplacian_3d streaming_laplacian{20, 20, 20};

void build_laplacian_3d_graph(const laplacian_3d& laplacian, ssi::graph_t& graph) {
  std::function<void(ssi::graph_count_builder_t&)> count_builder =
      [&](ssi::graph_count_builder_t& builder) {
        for (int64_t row = builder.beg; row < builder.end; ++row) {
          builder.set_count(row, laplacian.degree(row));
        }
      };

  std::function<void(ssi::graph_edge_builder_t&)> edge_builder =
      [&](ssi::graph_edge_builder_t& builder) {
        for (int64_t row = builder.beg; row < builder.end; ++row) {
          int64_t edge = 0;
          laplacian.for_each_neighbor(row, [&](int64_t col) {
            builder.set_edge(row, edge++, col);
          });
          builder.set_edge(row, edge++, row);
        }
      };

  graph.build_from_host(
      laplacian.n(),
      laplacian.n(),
      ssi::graph_orientation_t::row,
      count_builder,
      edge_builder);
}

void build_laplacian_3d_values(const laplacian_3d& laplacian, ssi::sparse_matrix_t& matrix) {
  std::function<void(ssi::sparse_value_builder_t&)> value_builder =
      [&](ssi::sparse_value_builder_t& builder) {
        for (int64_t row = builder.beg; row < builder.end; ++row) {
          for (int64_t edge = 0; edge < builder.degree(row); ++edge) {
            const int64_t col = builder.edge_id(row, edge);
            builder.set_value<double>(row, edge, col == row ? 6.0 : -1.0);
          }
        }
      };
  matrix.build_from_host(
      ssi::dtype_t::fp64, ssi::graph_orientation_t::row, value_builder);
}

void manufacture_rhs(
    const laplacian_3d& laplacian,
    int64_t rhs_index,
    std::vector<double>& expected,
    std::vector<double>& rhs) {
  for (int64_t row = 0; row < laplacian.n(); ++row) {
    expected[static_cast<std::size_t>(row)] =
        ssg::benchmarks::exact_value(row, rhs_index);
  }
  for (int64_t row = 0; row < laplacian.n(); ++row) {
    rhs[static_cast<std::size_t>(row)] = laplacian.apply_row(row, expected);
  }
}

void run_laplacian_3d_streaming_fp64(ssg::benchmark_context& context) {
  constexpr laplacian_3d laplacian = streaming_laplacian;
  constexpr int64_t rhs_count = streaming_rhs_count;
  constexpr double tolerance = 1.0e-8;
  const int64_t n = laplacian.n();

  TRACE_EVENT(
      "ssg.benchmark",
      "laplacian_3d_streaming_fp64.setup",
      "mx",
      laplacian.mx,
      "my",
      laplacian.my,
      "mz",
      laplacian.mz,
      "rhs_count",
      rhs_count);

  auto properties = ssg::benchmarks::make_laplacian_properties(n);
  auto problem = context.solver->make_sparse_problem(properties);
  auto graph = problem->make_graph();
  build_laplacian_3d_graph(laplacian, *graph);

  auto sparse_matrix = problem->make_sparse_matrix();
  build_laplacian_3d_values(laplacian, *sparse_matrix);
  problem->assert_properties(properties);

  auto symbolic = problem->make_symbolic_analysis();
  auto numeric = symbolic->make_numeric_factorization(sparse_matrix);

  std::vector<double> expected(static_cast<std::size_t>(n), 0.0);
  std::vector<double> host_rhs(static_cast<std::size_t>(n), 0.0);

  double max_relative_error_inf = 0.0;
  double max_relative_residual_inf = 0.0;
  for (int64_t rhs_index = 0; rhs_index < rhs_count; ++rhs_index) {
    manufacture_rhs(laplacian, rhs_index, expected, host_rhs);

    auto rhs = context.solver->make_matrix(ssi::dtype_t::fp64);
    rhs->preallocate(n, 1);
    std::function<void(ssi::matrix_view_t&)> rhs_builder =
        [&](ssi::matrix_view_t& view) {
          for (int64_t row = view.rbeg; row < view.rend; ++row) {
            view.value_mut<double>(row, 0) = host_rhs[static_cast<std::size_t>(row)];
          }
        };
    rhs->build_from_host(rhs_builder);

    auto solution = context.solver->make_matrix(ssi::dtype_t::fp64);
    solution->preallocate(n, 1);
    ssg::benchmarks::require_successful_solve(
        "laplacian_3d_streaming_fp64", numeric->solve(*rhs, *solution));

    const auto metrics = ssg::benchmarks::validate_solution(
        expected,
        host_rhs,
        *solution,
        [&](int64_t row, const std::vector<double>& values) {
          return laplacian.apply_row(row, values);
        });
    max_relative_error_inf =
        std::max(max_relative_error_inf, metrics.relative_error_inf);
    max_relative_residual_inf =
        std::max(max_relative_residual_inf, metrics.relative_residual_inf);
  }

  TRACE_EVENT(
      "ssg.benchmark",
      "laplacian_3d_streaming_fp64.verify",
      "max_relative_error_inf",
      max_relative_error_inf,
      "max_relative_residual_inf",
      max_relative_residual_inf);
  spdlog::info(
      "laplacian_3d_streaming_fp64 validation max_relative_error_inf={} "
      "max_relative_residual_inf={}",
      max_relative_error_inf,
      max_relative_residual_inf);
  if (max_relative_error_inf > tolerance || max_relative_residual_inf > tolerance) {
    throw std::runtime_error("laplacian_3d_streaming_fp64 validation failed");
  }
}

const ssg::benchmark_registration laplacian_3d_streaming_fp64_registration({
    "laplacian_3d_streaming_fp64",
    {"light", "streaming", "factorization", "fp64", "laplacian", "3d"},
    run_laplacian_3d_streaming_fp64,
    [] {
      return std::vector<ssi::sparse_problem_properties_t>{
          ssg::benchmarks::make_laplacian_properties(streaming_laplacian.n())};
    },
});

}  // namespace
