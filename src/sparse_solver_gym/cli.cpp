#include "sparse_solver_gym/cli.hpp"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace ssg {
namespace {

bool is_option(std::string_view value) {
  return value.starts_with("--");
}

std::string require_value(int& index, int argc, char** argv, std::string_view option) {
  if (index + 1 >= argc || is_option(argv[index + 1])) {
    throw std::runtime_error("missing value for " + std::string(option));
  }
  ++index;
  return argv[index];
}

void parse_common_option(cli_options& options, int& index, int argc, char** argv) {
  const std::string_view arg = argv[index];
  if (arg == "--solver") {
    options.solver_path = require_value(index, argc, argv, arg);
  } else if (arg == "--tag") {
    options.filter.tags.push_back(require_value(index, argc, argv, arg));
  } else if (arg == "--benchmark") {
    options.filter.names.push_back(require_value(index, argc, argv, arg));
  } else {
    throw std::runtime_error("unknown option: " + std::string(arg));
  }
}

}  // namespace

cli_options parse_cli(int argc, char** argv) {
  cli_options options;
  if (argc > 0) {
    options.executable_path = argv[0];
  }
  if (argc < 2) {
    return options;
  }

  const std::string_view cmd = argv[1];
  if (cmd == "help" || cmd == "--help" || cmd == "-h") {
    options.cmd = command::help;
    return options;
  }
  if (cmd == "list") {
    options.cmd = command::list;
  } else if (cmd == "run") {
    options.cmd = command::run;
  } else if (cmd == "worker") {
    options.cmd = command::worker;
  } else {
    throw std::runtime_error("unknown command: " + std::string(cmd));
  }

  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--output-dir") {
      options.output_dir = require_value(i, argc, argv, arg);
    } else if (arg == "--trace-out") {
      options.trace_path = require_value(i, argc, argv, arg);
    } else if (arg == "--worker-benchmark") {
      options.worker_benchmark = require_value(i, argc, argv, arg);
    } else if (arg == "--timeout-seconds") {
      options.timeout_seconds = std::stoi(require_value(i, argc, argv, arg));
      if (options.timeout_seconds < 0) {
        throw std::runtime_error("--timeout-seconds must be non-negative");
      }
    } else {
      parse_common_option(options, i, argc, argv);
    }
  }

  return options;
}

void print_usage(const char* executable) {
  std::cout
      << "usage:\n"
      << "  " << executable << " list [--tag TAG] [--benchmark NAME]\n"
      << "  " << executable << " run --solver SOLVER_SO [--output-dir DIR] [--timeout-seconds N] [--tag TAG] [--benchmark NAME]\n"
      << "  " << executable << " worker --solver SOLVER_SO --worker-benchmark NAME --trace-out TRACE\n";
}

}  // namespace ssg
