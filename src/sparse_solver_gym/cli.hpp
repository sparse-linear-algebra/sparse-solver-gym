#ifndef SPARSE_SOLVER_GYM_CLI_HPP
#define SPARSE_SOLVER_GYM_CLI_HPP

#include "sparse_solver_gym/benchmark.hpp"

#include <optional>
#include <string>
#include <vector>

namespace ssg {

enum class command {
  help,
  list,
  run,
  worker,
};

struct cli_options {
  command cmd = command::help;
  std::string executable_path;
  std::string solver_path;
  std::string output_dir;
  std::string trace_path;
  std::string worker_benchmark;
  int timeout_seconds = 0;
  benchmark_filter filter;
};

cli_options parse_cli(int argc, char** argv);
void print_usage(const char* executable);

}  // namespace ssg

#endif
