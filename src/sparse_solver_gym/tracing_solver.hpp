#ifndef SPARSE_SOLVER_GYM_TRACING_SOLVER_HPP
#define SPARSE_SOLVER_GYM_TRACING_SOLVER_HPP

#include "sparse_solver_interface.hpp"

#include <memory>
#include <string>

namespace ssg {

struct trace_metadata {
  std::string benchmark_name;
  std::string benchmark_tags;
  std::string solver_path;
};

std::shared_ptr<ssi::context_t> make_tracing_context(
    std::shared_ptr<ssi::context_t> inner,
    trace_metadata metadata);

}  // namespace ssg

#endif
