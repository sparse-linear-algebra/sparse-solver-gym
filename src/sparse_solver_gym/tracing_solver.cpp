#include "sparse_solver_gym/tracing_solver.hpp"

#include "sparse_solver_gym/perfetto_trace.hpp"

#include <stdexcept>
#include <utility>

namespace ssg {
namespace {

class tracing_context_t;
class tracing_matrix_t;
class tracing_sparse_problem_t;
class tracing_graph_t;
class tracing_sparse_matrix_t;
class tracing_symbolic_t;
class tracing_numeric_factorization_t;

const char* dtype_name(ssi::dtype_t dtype) {
  switch (dtype) {
    case ssi::dtype_t::fp32:
      return "fp32";
    case ssi::dtype_t::fp64:
      return "fp64";
    case ssi::dtype_t::c64:
      return "c64";
    case ssi::dtype_t::c128:
      return "c128";
  }
  return "unknown";
}

const char* itype_name(ssi::itype_t itype) {
  switch (itype) {
    case ssi::itype_t::i32:
      return "i32";
    case ssi::itype_t::i64:
      return "i64";
  }
  return "unknown";
}

const char* orientation_name(ssi::graph_orientation_t orientation) {
  switch (orientation) {
    case ssi::graph_orientation_t::row:
      return "row";
    case ssi::graph_orientation_t::column:
      return "column";
  }
  return "unknown";
}

class tracing_context_t final :
    public ssi::context_t,
    public std::enable_shared_from_this<tracing_context_t> {
 public:
  tracing_context_t(std::shared_ptr<ssi::context_t> inner, trace_metadata metadata)
      : inner_(std::move(inner)), metadata_(std::move(metadata)) {
    if (inner_ == nullptr) {
      throw std::runtime_error("null traced context");
    }
  }

  std::shared_ptr<ssi::matrix_t> make_matrix(ssi::dtype_t dtype) override;
  std::shared_ptr<ssi::graph_t> make_graph(ssi::itype_t itype) override;
  std::shared_ptr<ssi::sparse_problem_t> make_sparse_problem(
      const ssi::sparse_problem_properties_t& properties) override;
  ssi::support_result_t check_support(
      const ssi::sparse_problem_properties_t& properties) const override;

  const trace_metadata& metadata() const { return metadata_; }

 private:
  std::shared_ptr<ssi::context_t> inner_;
  trace_metadata metadata_;
};

class tracing_matrix_t final : public ssi::matrix_t {
 public:
  tracing_matrix_t(
      std::shared_ptr<ssi::context_t> context,
      std::shared_ptr<ssi::matrix_t> inner,
      trace_metadata metadata)
      : matrix_t(std::move(context)),
        inner_(std::move(inner)),
        metadata_(std::move(metadata)) {}

  ssi::matrix_t& inner() const { return *inner_; }

  int64_t nrows() const override {
    TRACE_EVENT("ssg.ssi.matrix", "matrix.nrows");
    return inner_->nrows();
  }

  int64_t ncols() const override {
    TRACE_EVENT("ssg.ssi.matrix", "matrix.ncols");
    return inner_->ncols();
  }

  ssi::dtype_t dtype() const override {
    TRACE_EVENT("ssg.ssi.matrix", "matrix.dtype");
    return inner_->dtype();
  }

  void preallocate(int64_t nrows, int64_t ncols) override {
    TRACE_EVENT(
        "ssg.ssi.matrix", "matrix.preallocate",
        "benchmark", metadata_.benchmark_name,
        "nrows", nrows,
        "ncols", ncols);
    inner_->preallocate(nrows, ncols);
  }

  void borrow_matrix_view(const ssi::placement_t& placement, const ssi::matrix_view_t& view) override {
    TRACE_EVENT(
        "ssg.ssi.matrix", "matrix.borrow_matrix_view",
        "benchmark", metadata_.benchmark_name,
        "dtype", dtype_name(view.dtype),
        "nrows", view.rend - view.rbeg,
        "ncols", view.cend - view.cbeg);
    inner_->borrow_matrix_view(placement, view);
  }

  void build_from_host(std::function<void(ssi::matrix_view_t&)>& builder) override {
    TRACE_EVENT("ssg.ssi.matrix", "matrix.build_from_host", "benchmark", metadata_.benchmark_name);
    std::function<void(ssi::matrix_view_t&)> traced_builder =
        [&](ssi::matrix_view_t& view) {
          TRACE_EVENT(
              "ssg.callback", "matrix.build_from_host.callback",
              "benchmark", metadata_.benchmark_name,
              "dtype", dtype_name(view.dtype),
              "nrows", view.rend - view.rbeg,
              "ncols", view.cend - view.cbeg);
          builder(view);
        };
    inner_->build_from_host(traced_builder);
  }

  void build_from_placement(
      std::function<void(const ssi::placement_t&, ssi::matrix_view_t&)>& builder) override {
    TRACE_EVENT("ssg.ssi.matrix", "matrix.build_from_placement", "benchmark", metadata_.benchmark_name);
    inner_->build_from_placement(builder);
  }

  void read_to_host(std::function<void(const ssi::matrix_view_t&)>& reader) const override {
    TRACE_EVENT("ssg.ssi.matrix", "matrix.read_to_host", "benchmark", metadata_.benchmark_name);
    std::function<void(const ssi::matrix_view_t&)> traced_reader =
        [&](const ssi::matrix_view_t& view) {
          TRACE_EVENT(
              "ssg.callback", "matrix.read_to_host.callback",
              "benchmark", metadata_.benchmark_name,
              "dtype", dtype_name(view.dtype),
              "nrows", view.rend - view.rbeg,
              "ncols", view.cend - view.cbeg);
          reader(view);
        };
    inner_->read_to_host(traced_reader);
  }

  void read_to_placement(
      const ssi::placement_t& placement,
      std::function<void(const ssi::matrix_view_t&)>& reader) const override {
    TRACE_EVENT("ssg.ssi.matrix", "matrix.read_to_placement", "benchmark", metadata_.benchmark_name);
    inner_->read_to_placement(placement, reader);
  }

 private:
  std::shared_ptr<ssi::matrix_t> inner_;
  trace_metadata metadata_;
};

class tracing_sparse_problem_t final :
    public ssi::sparse_problem_t,
    public std::enable_shared_from_this<tracing_sparse_problem_t> {
 public:
  tracing_sparse_problem_t(
      std::shared_ptr<ssi::context_t> context,
      std::shared_ptr<ssi::sparse_problem_t> inner,
      trace_metadata metadata,
      ssi::sparse_problem_properties_t properties)
      : sparse_problem_t(context, properties),
        context_(std::move(context)),
        inner_(std::move(inner)),
        metadata_(std::move(metadata)) {}

  ssi::sparse_problem_properties_t properties() const override {
    TRACE_EVENT(
        "ssg.ssi.sparse_problem", "sparse_problem.properties",
        "benchmark", metadata_.benchmark_name);
    return inner_->properties();
  }

  void assert_properties(const ssi::sparse_problem_properties_t& properties) override {
    TRACE_EVENT(
        "ssg.ssi.sparse_problem", "sparse_problem.assert_properties",
        "benchmark", metadata_.benchmark_name,
        "nrows", properties.nrows,
        "ncols", properties.ncols,
        "dtype", dtype_name(properties.dtype),
        "itype", itype_name(properties.itype));
    inner_->assert_properties(properties);
  }

  void compute_missing_properties() override {
    TRACE_EVENT(
        "ssg.ssi.sparse_problem", "sparse_problem.compute_missing_properties",
        "benchmark", metadata_.benchmark_name);
    inner_->compute_missing_properties();
  }

  std::shared_ptr<ssi::graph_t> make_graph() override;
  std::shared_ptr<ssi::sparse_matrix_t> make_sparse_matrix() override;
  std::shared_ptr<ssi::symbolic_t> make_symbolic_analysis() override;

 private:
  std::shared_ptr<ssi::context_t> context_;
  std::shared_ptr<ssi::sparse_problem_t> inner_;
  trace_metadata metadata_;
  std::shared_ptr<ssi::graph_t> graph_;
  std::shared_ptr<ssi::sparse_matrix_t> matrix_;
  std::shared_ptr<ssi::symbolic_t> symbolic_;
};

class tracing_graph_t final :
    public ssi::graph_t,
    public std::enable_shared_from_this<tracing_graph_t> {
 public:
  tracing_graph_t(
      std::shared_ptr<ssi::context_t> context,
      std::shared_ptr<ssi::graph_t> inner,
      trace_metadata metadata)
      : graph_t(std::move(context)),
        inner_(std::move(inner)),
        metadata_(std::move(metadata)) {}

  ssi::graph_t& inner() const { return *inner_; }

  ssi::itype_t itype() const override {
    TRACE_EVENT("ssg.ssi.graph", "graph.itype");
    return inner_->itype();
  }

  int64_t nrows() const override {
    TRACE_EVENT("ssg.ssi.graph", "graph.nrows");
    return inner_->nrows();
  }

  int64_t ncols() const override {
    TRACE_EVENT("ssg.ssi.graph", "graph.ncols");
    return inner_->ncols();
  }

  int64_t nedges() const override {
    TRACE_EVENT("ssg.ssi.graph", "graph.nedges");
    return inner_->nedges();
  }

  void build_from_host(
      int64_t nrows,
      int64_t ncols,
      ssi::graph_orientation_t orientation,
      std::function<void(ssi::graph_count_builder_t&)>& count_builder,
      std::function<void(ssi::graph_edge_builder_t&)>& edge_builder) override {
    TRACE_EVENT(
        "ssg.ssi.graph", "graph.build_from_host",
        "benchmark", metadata_.benchmark_name,
        "nrows", nrows,
        "ncols", ncols,
        "orientation", orientation_name(orientation));
    std::function<void(ssi::graph_count_builder_t&)> traced_count =
        [&](ssi::graph_count_builder_t& builder) {
          TRACE_EVENT(
              "ssg.callback", "graph.count_builder",
              "benchmark", metadata_.benchmark_name,
              "beg", builder.beg,
              "end", builder.end);
          count_builder(builder);
        };
    std::function<void(ssi::graph_edge_builder_t&)> traced_edges =
        [&](ssi::graph_edge_builder_t& builder) {
          TRACE_EVENT(
              "ssg.callback", "graph.edge_builder",
              "benchmark", metadata_.benchmark_name,
              "beg", builder.beg,
              "end", builder.end);
          edge_builder(builder);
        };
    inner_->build_from_host(nrows, ncols, orientation, traced_count, traced_edges);
  }

  void borrow_compressed_graph_view(const ssi::compressed_graph_view_t& view) override {
    TRACE_EVENT(
        "ssg.ssi.graph", "graph.borrow_compressed_graph_view",
        "benchmark", metadata_.benchmark_name,
        "nrows", view.nrows,
        "ncols", view.ncols);
    inner_->borrow_compressed_graph_view(view);
  }

  std::shared_ptr<ssi::sparse_matrix_t> make_sparse_matrix() override;
  std::shared_ptr<ssi::symbolic_t> make_symbolic_analysis() override;

 private:
  std::shared_ptr<ssi::graph_t> inner_;
  trace_metadata metadata_;
};

class tracing_sparse_matrix_t final : public ssi::sparse_matrix_t {
 public:
  tracing_sparse_matrix_t(
      std::shared_ptr<ssi::graph_t> graph,
      std::shared_ptr<ssi::sparse_matrix_t> inner,
      trace_metadata metadata)
      : sparse_matrix_t(std::move(graph)),
        inner_(std::move(inner)),
        metadata_(std::move(metadata)) {}

  ssi::sparse_matrix_t& inner() const { return *inner_; }

  int64_t nrows() const override {
    TRACE_EVENT("ssg.ssi.sparse_matrix", "sparse_matrix.nrows");
    return inner_->nrows();
  }

  int64_t ncols() const override {
    TRACE_EVENT("ssg.ssi.sparse_matrix", "sparse_matrix.ncols");
    return inner_->ncols();
  }

  ssi::dtype_t dtype() const override {
    TRACE_EVENT("ssg.ssi.sparse_matrix", "sparse_matrix.dtype");
    return inner_->dtype();
  }

  void build_from_host(
      ssi::dtype_t dtype,
      ssi::graph_orientation_t orientation,
      std::function<void(ssi::sparse_value_builder_t&)>& builder) override {
    TRACE_EVENT(
        "ssg.ssi.sparse_matrix", "sparse_matrix.build_from_host",
        "benchmark", metadata_.benchmark_name,
        "dtype", dtype_name(dtype),
        "orientation", orientation_name(orientation));
    std::function<void(ssi::sparse_value_builder_t&)> traced_builder =
        [&](ssi::sparse_value_builder_t& value_builder) {
          TRACE_EVENT(
              "ssg.callback", "sparse_matrix.value_builder",
              "benchmark", metadata_.benchmark_name,
              "beg", value_builder.beg,
              "end", value_builder.end);
          builder(value_builder);
        };
    inner_->build_from_host(dtype, orientation, traced_builder);
  }

  void read_to_host(
      ssi::graph_orientation_t orientation,
      std::function<void(const ssi::sparse_value_builder_t&)>& reader) const override {
    TRACE_EVENT("ssg.ssi.sparse_matrix", "sparse_matrix.read_to_host", "orientation", orientation_name(orientation));
    std::function<void(const ssi::sparse_value_builder_t&)> traced_reader =
        [&](const ssi::sparse_value_builder_t& value_builder) {
          TRACE_EVENT(
              "ssg.callback", "sparse_matrix.value_reader",
              "benchmark", metadata_.benchmark_name,
              "beg", value_builder.beg,
              "end", value_builder.end);
          reader(value_builder);
        };
    inner_->read_to_host(orientation, traced_reader);
  }

  void borrow_sparse_values_view(const ssi::sparse_values_view_t& view) override {
    TRACE_EVENT(
        "ssg.ssi.sparse_matrix", "sparse_matrix.borrow_sparse_values_view",
        "benchmark", metadata_.benchmark_name,
        "dtype", dtype_name(view.dtype),
        "nedges", view.nedges);
    inner_->borrow_sparse_values_view(view);
  }

 private:
  std::shared_ptr<ssi::sparse_matrix_t> inner_;
  trace_metadata metadata_;
};

class tracing_symbolic_t final :
    public ssi::symbolic_t,
    public std::enable_shared_from_this<tracing_symbolic_t> {
 public:
  tracing_symbolic_t(
      std::shared_ptr<ssi::graph_t> graph,
      std::shared_ptr<ssi::symbolic_t> inner,
      trace_metadata metadata)
      : symbolic_t(std::move(graph)),
        inner_(std::move(inner)),
        metadata_(std::move(metadata)) {}

  ssi::symbolic_t& inner() const { return *inner_; }

  std::shared_ptr<ssi::numeric_factorization_t>
  make_numeric_factorization(std::shared_ptr<ssi::sparse_matrix_t> matrix) override;

 private:
  std::shared_ptr<ssi::symbolic_t> inner_;
  trace_metadata metadata_;
};

class tracing_numeric_factorization_t final : public ssi::numeric_factorization_t {
 public:
  tracing_numeric_factorization_t(
      std::shared_ptr<ssi::symbolic_t> symbolic,
      std::shared_ptr<ssi::sparse_matrix_t> matrix,
      std::shared_ptr<ssi::numeric_factorization_t> inner,
      trace_metadata metadata)
      : numeric_factorization_t(std::move(symbolic), std::move(matrix)),
        inner_(std::move(inner)),
        metadata_(std::move(metadata)) {}

  ssi::dtype_t dtype() const override {
    TRACE_EVENT("ssg.ssi.numeric", "numeric_factorization.dtype");
    return inner_->dtype();
  }

  ssi::solve_result_t solve(const ssi::matrix_t& rhs, ssi::matrix_t& solution) const override {
    TRACE_EVENT("ssg.ssi.numeric", "numeric_factorization.solve", "benchmark", metadata_.benchmark_name);
    const auto* traced_rhs = dynamic_cast<const tracing_matrix_t*>(&rhs);
    auto* traced_solution = dynamic_cast<tracing_matrix_t*>(&solution);
    const ssi::matrix_t& inner_rhs = traced_rhs != nullptr ? traced_rhs->inner() : rhs;
    ssi::matrix_t& inner_solution = traced_solution != nullptr ? traced_solution->inner() : solution;
    return inner_->solve(inner_rhs, inner_solution);
  }

 private:
  std::shared_ptr<ssi::numeric_factorization_t> inner_;
  trace_metadata metadata_;
};

std::shared_ptr<ssi::matrix_t> tracing_context_t::make_matrix(ssi::dtype_t dtype) {
  TRACE_EVENT(
      "ssg.ssi.context", "context.make_matrix",
      "benchmark", metadata_.benchmark_name,
      "dtype", dtype_name(dtype));
  return std::make_shared<tracing_matrix_t>(
      shared_from_this(), inner_->make_matrix(dtype), metadata_);
}

std::shared_ptr<ssi::graph_t> tracing_context_t::make_graph(ssi::itype_t itype) {
  TRACE_EVENT(
      "ssg.ssi.context", "context.make_graph",
      "benchmark", metadata_.benchmark_name,
      "itype", itype_name(itype));
  return std::make_shared<tracing_graph_t>(
      shared_from_this(), inner_->make_graph(itype), metadata_);
}

std::shared_ptr<ssi::sparse_problem_t> tracing_context_t::make_sparse_problem(
    const ssi::sparse_problem_properties_t& properties) {
  TRACE_EVENT(
      "ssg.ssi.context", "context.make_sparse_problem",
      "benchmark", metadata_.benchmark_name,
      "nrows", properties.nrows,
      "ncols", properties.ncols,
      "orientation", orientation_name(properties.orientation),
      "dtype", dtype_name(properties.dtype),
      "itype", itype_name(properties.itype));
  return std::make_shared<tracing_sparse_problem_t>(
      shared_from_this(),
      inner_->make_sparse_problem(properties),
      metadata_,
      properties);
}

ssi::support_result_t tracing_context_t::check_support(
    const ssi::sparse_problem_properties_t& properties) const {
  TRACE_EVENT(
      "ssg.ssi.context", "context.check_support",
      "benchmark", metadata_.benchmark_name,
      "nrows", properties.nrows,
      "ncols", properties.ncols,
      "orientation", orientation_name(properties.orientation),
      "dtype", dtype_name(properties.dtype),
      "itype", itype_name(properties.itype));
  return inner_->check_support(properties);
}

std::shared_ptr<ssi::graph_t> tracing_sparse_problem_t::make_graph() {
  TRACE_EVENT(
      "ssg.ssi.sparse_problem", "sparse_problem.make_graph",
      "benchmark", metadata_.benchmark_name);
  if (graph_ == nullptr) {
    graph_ = std::make_shared<tracing_graph_t>(
        context_,
        inner_->make_graph(),
        metadata_);
  }
  return graph_;
}

std::shared_ptr<ssi::sparse_matrix_t> tracing_sparse_problem_t::make_sparse_matrix() {
  TRACE_EVENT(
      "ssg.ssi.sparse_problem", "sparse_problem.make_sparse_matrix",
      "benchmark", metadata_.benchmark_name);
  if (matrix_ == nullptr) {
    matrix_ = std::make_shared<tracing_sparse_matrix_t>(
        make_graph(),
        inner_->make_sparse_matrix(),
        metadata_);
  }
  return matrix_;
}

std::shared_ptr<ssi::symbolic_t> tracing_sparse_problem_t::make_symbolic_analysis() {
  TRACE_EVENT(
      "ssg.ssi.sparse_problem", "sparse_problem.make_symbolic_analysis",
      "benchmark", metadata_.benchmark_name);
  if (symbolic_ == nullptr) {
    symbolic_ = std::make_shared<tracing_symbolic_t>(
        make_graph(),
        inner_->make_symbolic_analysis(),
        metadata_);
  }
  return symbolic_;
}

std::shared_ptr<ssi::sparse_matrix_t> tracing_graph_t::make_sparse_matrix() {
  TRACE_EVENT("ssg.ssi.graph", "graph.make_sparse_matrix", "benchmark", metadata_.benchmark_name);
  return std::make_shared<tracing_sparse_matrix_t>(
      shared_from_this(), inner_->make_sparse_matrix(), metadata_);
}

std::shared_ptr<ssi::symbolic_t> tracing_graph_t::make_symbolic_analysis() {
  TRACE_EVENT("ssg.ssi.graph", "graph.make_symbolic_analysis", "benchmark", metadata_.benchmark_name);
  return std::make_shared<tracing_symbolic_t>(
      shared_from_this(), inner_->make_symbolic_analysis(), metadata_);
}

std::shared_ptr<ssi::numeric_factorization_t>
tracing_symbolic_t::make_numeric_factorization(std::shared_ptr<ssi::sparse_matrix_t> matrix) {
  TRACE_EVENT("ssg.ssi.symbolic", "symbolic.make_numeric_factorization", "benchmark", metadata_.benchmark_name);
  const auto* traced_matrix = dynamic_cast<const tracing_sparse_matrix_t*>(matrix.get());
  std::shared_ptr<ssi::sparse_matrix_t> inner_matrix = matrix;
  if (traced_matrix != nullptr) {
    inner_matrix = std::shared_ptr<ssi::sparse_matrix_t>(matrix, &traced_matrix->inner());
  }
  return std::make_shared<tracing_numeric_factorization_t>(
      std::shared_ptr<ssi::symbolic_t>(shared_from_this(), this),
      matrix,
      inner_->make_numeric_factorization(inner_matrix),
      metadata_);
}

}  // namespace

std::shared_ptr<ssi::context_t> make_tracing_context(
    std::shared_ptr<ssi::context_t> inner,
    trace_metadata metadata) {
  return std::make_shared<tracing_context_t>(std::move(inner), std::move(metadata));
}

}  // namespace ssg
