#include "sparse_solver_gym/orchestrator.hpp"

#include "sparse_solver_gym/benchmark.hpp"

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
  failed,
  crashed,
  timed_out,
};

struct child_result {
  std::string benchmark_name;
  std::filesystem::path trace_path;
  child_status status = child_status::failed;
  int code = 0;
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
    result.status = result.code == 0 ? child_status::passed : child_status::failed;
  } else if (WIFSIGNALED(status)) {
    result.code = WTERMSIG(status);
    result.status = child_status::crashed;
  }
  return result;
}

const char* status_name(child_status status) {
  switch (status) {
    case child_status::passed:
      return "passed";
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

  std::vector<child_result> results;
  for (const auto* benchmark : selected) {
    const auto trace_path = single_trace_path
                                ? std::filesystem::path(options.trace_path)
                                : output_dir / (shell_safe_name(benchmark->name) + ".pftrace");
    spdlog::info("running benchmark {}", benchmark->name);
    auto result = run_child(options, *benchmark, trace_path);
    spdlog::info(
        "benchmark {} {}; trace {}",
        result.benchmark_name,
        status_name(result.status),
        result.trace_path.string());
    results.push_back(std::move(result));
  }

  bool all_passed = true;
  for (const auto& result : results) {
    if (result.status != child_status::passed) {
      all_passed = false;
      spdlog::error(
          "benchmark {} ended as {} ({})",
          result.benchmark_name,
          status_name(result.status),
          result.code);
    }
  }
  return all_passed ? 0 : 1;
}

}  // namespace ssg
