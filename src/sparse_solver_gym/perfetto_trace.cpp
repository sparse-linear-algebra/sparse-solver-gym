#include "sparse_solver_gym/perfetto_trace.hpp"

#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <spdlog/spdlog.h>

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace ssg {
namespace {

std::once_flag perfetto_init_once;

}  // namespace

void initialize_perfetto_once() {
  std::call_once(perfetto_init_once, [] {
    perfetto::TracingInitArgs args;
    args.backends = perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
  });
}

perfetto_trace_session::perfetto_trace_session(std::string output_path)
    : output_path_(std::move(output_path)) {
  initialize_perfetto_once();

  perfetto::TraceConfig config;
  config.add_buffers()->set_size_kb(64 * 1024);
  auto* data_source = config.add_data_sources()->mutable_config();
  data_source->set_name("track_event");

  perfetto::protos::gen::TrackEventConfig track_event_config;
  track_event_config.add_disabled_categories("*");
  track_event_config.add_enabled_categories("ssg.benchmark");
  track_event_config.add_enabled_categories("ssg.callback");
  track_event_config.add_enabled_categories("ssg.ssi.context");
  track_event_config.add_enabled_categories("ssg.ssi.matrix");
  track_event_config.add_enabled_categories("ssg.ssi.sparse_problem");
  track_event_config.add_enabled_categories("ssg.ssi.graph");
  track_event_config.add_enabled_categories("ssg.ssi.sparse_matrix");
  track_event_config.add_enabled_categories("ssg.ssi.symbolic");
  track_event_config.add_enabled_categories("ssg.ssi.numeric");
  data_source->set_track_event_config_raw(track_event_config.SerializeAsString());

  session_ = perfetto::Tracing::NewTrace();
  session_->Setup(config);
  session_->StartBlocking();
}

perfetto_trace_session::~perfetto_trace_session() {
  if (!stopped_) {
    try {
      stop();
    } catch (const std::exception& error) {
      spdlog::error("failed to stop Perfetto trace: {}", error.what());
    }
  }
}

void perfetto_trace_session::stop() {
  if (stopped_) {
    return;
  }
  perfetto::TrackEvent::Flush();
  session_->StopBlocking();
  std::vector<char> trace_data(session_->ReadTraceBlocking());

  std::ofstream output(output_path_, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to open trace output: " + output_path_);
  }
  output.write(trace_data.data(), static_cast<std::streamsize>(trace_data.size()));
  if (!output) {
    throw std::runtime_error("failed to write trace output: " + output_path_);
  }
  stopped_ = true;
}

}  // namespace ssg
