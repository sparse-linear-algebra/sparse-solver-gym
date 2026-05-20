#ifndef SPARSE_SOLVER_GYM_ORCHESTRATOR_HPP
#define SPARSE_SOLVER_GYM_ORCHESTRATOR_HPP

#include "sparse_solver_gym/cli.hpp"

namespace ssg {

int run_orchestrator(const cli_options& options);
int list_benchmarks(const cli_options& options);

}  // namespace ssg

#endif
