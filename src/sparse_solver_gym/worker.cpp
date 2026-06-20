#include "sparse_solver_gym/worker.hpp"

#include "sparse_solver_gym/benchmark.hpp"
#include "sparse_solver_gym/perfetto_trace.hpp"
#include "sparse_solver_gym/tracing_solver.hpp"

#include "sparse_solver_interface_plugin.hpp"

#include <stdexcept>

#include <spdlog/spdlog.h>

namespace ssg {
namespace {

std::string join_tags(const std::vector<std::string>& tags) {
  std::string joined;
  for (std::size_t i = 0; i < tags.size(); ++i) {
    if (i != 0) {
      joined += ",";
    }
    joined += tags[i];
  }
  return joined;
}

void validate_worker_options(const cli_options& options) {
  if (options.solver_path.empty()) {
    throw std::runtime_error("worker requires --solver");
  }
  if (options.worker_benchmark.empty()) {
    throw std::runtime_error("worker requires --worker-benchmark");
  }
  if (options.trace_path.empty()) {
    throw std::runtime_error("worker requires --trace-out");
  }
}

}  // namespace

int run_worker(const cli_options& options) {
  try {
    validate_worker_options(options);

    const auto* benchmark =
        benchmark_registry::instance().find(options.worker_benchmark);
    if (benchmark == nullptr) {
      throw std::runtime_error("unknown benchmark: " + options.worker_benchmark);
    }

    spdlog::info("worker starting benchmark {}", benchmark->name);
    perfetto_trace_session trace(options.trace_path);

    trace_metadata metadata{
        benchmark->name,
        join_tags(benchmark->tags),
        options.solver_path,
    };

    auto loaded_context =
        ssi::load_context_from_shared_object(options.solver_path.c_str());
    benchmark_context context{
        make_tracing_context(std::move(loaded_context), metadata),
        options.solver_path,
        options.trace_path,
    };

    TRACE_EVENT(
        "ssg.benchmark", "benchmark.run",
        "benchmark", benchmark->name,
        "tags", metadata.benchmark_tags,
        "solver", options.solver_path);
    benchmark->run(context);

    trace.stop();
    spdlog::info("worker finished benchmark {}; trace {}", benchmark->name, options.trace_path);
    return 0;
  } catch (const ssi::unsupported_error_t& error) {
    spdlog::warn("worker unsupported: {}", error.what());
    return worker_unsupported_exit_code;
  } catch (const std::exception& error) {
    spdlog::error("worker failed: {}", error.what());
    return 1;
  } catch (...) {
    spdlog::error("worker failed with an unknown exception");
    return 1;
  }
}

}  // namespace ssg
