#ifndef SPARSE_SOLVER_GYM_WORKER_HPP
#define SPARSE_SOLVER_GYM_WORKER_HPP

#include "sparse_solver_gym/cli.hpp"

namespace ssg {

inline constexpr int worker_unsupported_exit_code = 10;

int run_worker(const cli_options& options);

}  // namespace ssg

#endif
