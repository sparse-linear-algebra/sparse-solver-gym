#ifndef SPARSE_SOLVER_GYM_PERFETTO_TRACE_HPP
#define SPARSE_SOLVER_GYM_PERFETTO_TRACE_HPP

#include <memory>
#include <string>

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("ssg.benchmark").SetDescription("Sparse Solver Gym benchmarks"),
    perfetto::Category("ssg.callback").SetDescription("Sparse Solver Gym benchmark callbacks"),
    perfetto::Category("ssg.ssi.context").SetDescription("SSI context calls"),
    perfetto::Category("ssg.ssi.matrix").SetDescription("SSI dense matrix calls"),
    perfetto::Category("ssg.ssi.sparse_problem").SetDescription("SSI sparse problem calls"),
    perfetto::Category("ssg.ssi.graph").SetDescription("SSI graph calls"),
    perfetto::Category("ssg.ssi.sparse_matrix").SetDescription("SSI sparse matrix calls"),
    perfetto::Category("ssg.ssi.symbolic").SetDescription("SSI symbolic calls"),
    perfetto::Category("ssg.ssi.numeric").SetDescription("SSI numeric factorization calls"));

namespace ssg {

void initialize_perfetto_once();

class perfetto_trace_session {
 public:
  explicit perfetto_trace_session(std::string output_path);
  ~perfetto_trace_session();

  perfetto_trace_session(const perfetto_trace_session&) = delete;
  perfetto_trace_session& operator=(const perfetto_trace_session&) = delete;

  void stop();

 private:
  std::string output_path_;
  std::unique_ptr<perfetto::TracingSession> session_;
  bool stopped_ = false;
};

}  // namespace ssg

#endif
