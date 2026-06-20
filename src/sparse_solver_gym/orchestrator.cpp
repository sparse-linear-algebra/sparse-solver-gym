#include "sparse_solver_gym/orchestrator.hpp"

#include "sparse_solver_gym/benchmark.hpp"
#include "sparse_solver_gym/worker.hpp"

#include "sparse_solver_interface_plugin.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <spdlog/spdlog.h>

namespace ssg {
namespace {

enum class child_status {
  passed,
  unsupported,
  failed,
  crashed,
  timed_out,
};

struct child_result {
  std::string benchmark_name;
  std::filesystem::path trace_path;
  child_status status = child_status::failed;
  int code = 0;
  std::string reason;
};

std::string shell_safe_name(std::string_view value) {
  std::string out;
  for (char c : value) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out;
}

std::filesystem::path default_output_dir() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&time, &tm);

  std::ostringstream name;
  name << std::put_time(&tm, "%Y%m%d-%H%M%S") << "-" << getpid();
  return std::filesystem::temp_directory_path() / "sparse-solver-gym" / name.str();
}

child_result run_child(
    const cli_options& options,
    const benchmark_info& benchmark,
    const std::filesystem::path& trace_path) {
  const pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
  }

  if (pid == 0) {
    std::vector<std::string> args{
        options.executable_path,
        "worker",
        "--solver",
        options.solver_path,
        "--worker-benchmark",
        benchmark.name,
        "--trace-out",
        trace_path.string(),
    };

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& arg : args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    execv(options.executable_path.c_str(), argv.data());
    spdlog::critical("execv failed: {}", std::strerror(errno));
    _exit(127);
  }

  int status = 0;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(options.timeout_seconds);
  while (true) {
    const pid_t wait_result = waitpid(
        pid,
        &status,
        options.timeout_seconds > 0 ? WNOHANG : 0);
    if (wait_result == pid) {
      break;
    }
    if (wait_result < 0) {
      throw std::runtime_error(std::string("waitpid failed: ") + std::strerror(errno));
    }
    if (options.timeout_seconds > 0 && std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGTERM);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (waitpid(pid, &status, WNOHANG) == 0) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
      }
      child_result result;
      result.benchmark_name = benchmark.name;
      result.trace_path = trace_path;
      result.status = child_status::timed_out;
      result.code = options.timeout_seconds;
      return result;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  child_result result;
  result.benchmark_name = benchmark.name;
  result.trace_path = trace_path;
  if (WIFEXITED(status)) {
    result.code = WEXITSTATUS(status);
    if (result.code == 0) {
      result.status = child_status::passed;
    } else if (result.code == worker_unsupported_exit_code) {
      result.status = child_status::unsupported;
      result.reason = "worker reported unsupported benchmark requirements";
    } else {
      result.status = child_status::failed;
    }
  } else if (WIFSIGNALED(status)) {
    result.code = WTERMSIG(status);
    result.status = child_status::crashed;
  }
  return result;
}

const char* ssi_status_name(ssi::status_t status) {
  switch (status) {
    case ssi::status_t::ok:
      return "ok";
    case ssi::status_t::invalid_argument:
      return "invalid_argument";
    case ssi::status_t::out_of_range:
      return "out_of_range";
    case ssi::status_t::unsupported:
      return "unsupported";
    case ssi::status_t::singular:
      return "singular";
    case ssi::status_t::rank_deficient:
      return "rank_deficient";
    case ssi::status_t::indefinite:
      return "indefinite";
    case ssi::status_t::zero_pivot:
      return "zero_pivot";
    case ssi::status_t::breakdown:
      return "breakdown";
    case ssi::status_t::not_converged:
      return "not_converged";
    case ssi::status_t::exception:
      return "exception";
  }
  return "unknown";
}

std::string support_reason(const ssi::support_result_t& support) {
  std::string reason = std::string("status=") + ssi_status_name(support.status);
  if (!support.reason.empty()) {
    reason += " reason=" + support.reason;
  }
  return reason;
}

bool benchmark_supported(
    const ssi::context_t& solver,
    const benchmark_info& benchmark,
    std::string& reason) {
  if (!benchmark.support_requirements) {
    return true;
  }
  const auto requirements = benchmark.support_requirements();
  for (const auto& properties : requirements) {
    const auto support = solver.check_support(properties);
    if (!support.supported()) {
      reason = support_reason(support);
      return false;
    }
  }
  return true;
}

const char* status_name(child_status status) {
  switch (status) {
    case child_status::passed:
      return "passed";
    case child_status::unsupported:
      return "unsupported";
    case child_status::failed:
      return "failed";
    case child_status::crashed:
      return "crashed";
    case child_status::timed_out:
      return "timed_out";
  }
  return "unknown";
}

}  // namespace

int list_benchmarks(const cli_options& options) {
  const auto selected = select_benchmarks(options.filter);
  for (const auto* benchmark : selected) {
    std::ostringstream tags;
    for (std::size_t i = 0; i < benchmark->tags.size(); ++i) {
      if (i != 0) {
        tags << ",";
      }
      tags << benchmark->tags[i];
    }
    spdlog::info("{} [{}]", benchmark->name, tags.str());
  }
  return 0;
}

int run_orchestrator(const cli_options& options) {
  if (options.solver_path.empty()) {
    throw std::runtime_error("run requires --solver");
  }

  const auto selected = select_benchmarks(options.filter);
  if (selected.empty()) {
    spdlog::warn("no benchmarks selected");
    return 0;
  }

  std::filesystem::path output_dir;
  bool single_trace_path = false;
  if (!options.output_dir.empty()) {
    output_dir = options.output_dir;
  } else if (!options.trace_path.empty()) {
    if (selected.size() != 1) {
      throw std::runtime_error("--trace-out can only be used with a single selected benchmark");
    }
    output_dir = std::filesystem::path(options.trace_path).parent_path();
    if (output_dir.empty()) {
      output_dir = ".";
    }
    single_trace_path = true;
  } else {
    output_dir = default_output_dir();
  }

  std::filesystem::create_directories(output_dir);
  spdlog::info("writing benchmark traces to {}", output_dir.string());

  auto support_context =
      ssi::load_context_from_shared_object(options.solver_path.c_str());

  std::vector<child_result> results;
  for (const auto* benchmark : selected) {
    std::string unsupported_reason;
    if (!benchmark_supported(*support_context, *benchmark, unsupported_reason)) {
      child_result result;
      result.benchmark_name = benchmark->name;
      result.status = child_status::unsupported;
      result.reason = std::move(unsupported_reason);
      spdlog::warn(
          "benchmark {} unsupported: {}",
          result.benchmark_name,
          result.reason);
      results.push_back(std::move(result));
      continue;
    }

    const auto trace_path = single_trace_path
                                ? std::filesystem::path(options.trace_path)
                                : output_dir / (shell_safe_name(benchmark->name) + ".pftrace");
    spdlog::info("running benchmark {}", benchmark->name);
    auto result = run_child(options, *benchmark, trace_path);
    if (result.status == child_status::unsupported) {
      spdlog::warn(
          "benchmark {} unsupported: {}",
          result.benchmark_name,
          result.reason);
    } else {
      spdlog::info(
          "benchmark {} {}; trace {}",
          result.benchmark_name,
          status_name(result.status),
          result.trace_path.string());
    }
    results.push_back(std::move(result));
  }

  bool all_supported_benchmarks_passed = true;
  for (const auto& result : results) {
    if (result.status == child_status::passed) {
      continue;
    }
    if (result.status == child_status::unsupported) {
      spdlog::warn(
          "benchmark {} ended as unsupported ({})",
          result.benchmark_name,
          result.reason);
      continue;
    }
    all_supported_benchmarks_passed = false;
    spdlog::error(
        "benchmark {} ended as {} ({})",
        result.benchmark_name,
        status_name(result.status),
        result.code);
  }
  return all_supported_benchmarks_passed ? 0 : 1;
}

}  // namespace ssg
