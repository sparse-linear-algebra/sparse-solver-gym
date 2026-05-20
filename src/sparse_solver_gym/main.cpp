#include "sparse_solver_gym/cli.hpp"
#include "sparse_solver_gym/orchestrator.hpp"
#include "sparse_solver_gym/worker.hpp"

#include <exception>

#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
  spdlog::set_pattern("[%H:%M:%S.%e] [%l] %v");

  try {
    const auto options = ssg::parse_cli(argc, argv);
    switch (options.cmd) {
      case ssg::command::help:
        ssg::print_usage(argc > 0 ? argv[0] : "sparse-solver-gym");
        return 0;
      case ssg::command::list:
        return ssg::list_benchmarks(options);
      case ssg::command::run:
        return ssg::run_orchestrator(options);
      case ssg::command::worker:
        return ssg::run_worker(options);
    }
  } catch (const std::exception& error) {
    spdlog::error("{}", error.what());
    ssg::print_usage(argc > 0 ? argv[0] : "sparse-solver-gym");
    return 2;
  }
  return 2;
}
