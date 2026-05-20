#include "sparse_solver_gym/benchmark.hpp"

#include <algorithm>
#include <stdexcept>

namespace ssg {

benchmark_registry& benchmark_registry::instance() {
  static benchmark_registry registry;
  return registry;
}

void benchmark_registry::add(benchmark_info info) {
  if (find(info.name) != nullptr) {
    throw std::runtime_error("duplicate benchmark registration: " + info.name);
  }
  benchmarks_.push_back(std::move(info));
}

const std::vector<benchmark_info>& benchmark_registry::benchmarks() const {
  return benchmarks_;
}

const benchmark_info* benchmark_registry::find(std::string_view name) const {
  for (const auto& benchmark : benchmarks_) {
    if (benchmark.name == name) {
      return &benchmark;
    }
  }
  return nullptr;
}

benchmark_registration::benchmark_registration(benchmark_info info) {
  benchmark_registry::instance().add(std::move(info));
}

bool has_tag(const benchmark_info& benchmark, std::string_view tag) {
  return std::find(benchmark.tags.begin(), benchmark.tags.end(), tag) !=
         benchmark.tags.end();
}

std::vector<const benchmark_info*> select_benchmarks(const benchmark_filter& filter) {
  std::vector<const benchmark_info*> selected;
  const auto& registry = benchmark_registry::instance();

  if (!filter.names.empty()) {
    for (const auto& name : filter.names) {
      const auto* benchmark = registry.find(name);
      if (benchmark == nullptr) {
        throw std::runtime_error("unknown benchmark: " + name);
      }
      selected.push_back(benchmark);
    }
  } else {
    for (const auto& benchmark : registry.benchmarks()) {
      selected.push_back(&benchmark);
    }
  }

  if (filter.tags.empty()) {
    return selected;
  }

  std::vector<const benchmark_info*> filtered;
  for (const auto* benchmark : selected) {
    const bool matches_all_tags = std::all_of(
        filter.tags.begin(), filter.tags.end(),
        [&](const std::string& tag) { return has_tag(*benchmark, tag); });
    if (matches_all_tags) {
      filtered.push_back(benchmark);
    }
  }
  return filtered;
}

}  // namespace ssg
