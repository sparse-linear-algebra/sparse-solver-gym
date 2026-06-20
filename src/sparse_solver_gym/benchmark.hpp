#ifndef SPARSE_SOLVER_GYM_BENCHMARK_HPP
#define SPARSE_SOLVER_GYM_BENCHMARK_HPP

#include "sparse_solver_interface.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ssg {

struct benchmark_context {
  std::shared_ptr<ssi::context_t> solver;
  std::string solver_path;
  std::string trace_path;
};

struct benchmark_info {
  std::string name;
  std::vector<std::string> tags;
  std::function<void(benchmark_context&)> run;
  std::function<std::vector<ssi::sparse_problem_properties_t>()> support_requirements;
};

class benchmark_registry {
 public:
  static benchmark_registry& instance();

  void add(benchmark_info info);
  const std::vector<benchmark_info>& benchmarks() const;
  const benchmark_info* find(std::string_view name) const;

 private:
  std::vector<benchmark_info> benchmarks_;
};

class benchmark_registration {
 public:
  explicit benchmark_registration(benchmark_info info);
};

struct benchmark_filter {
  std::vector<std::string> names;
  std::vector<std::string> tags;
};

std::vector<const benchmark_info*> select_benchmarks(const benchmark_filter& filter);
bool has_tag(const benchmark_info& benchmark, std::string_view tag);

}  // namespace ssg

#endif
