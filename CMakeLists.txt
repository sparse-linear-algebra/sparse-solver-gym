cmake_minimum_required(VERSION 3.20)

project(sparse_solver_gym LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SPARSE_SOLVER_INTERFACE_ROOT
    "/home/reidatcheson/sparse_linear_algebra/installs/sparse_solver_interface/0.1.0"
    CACHE PATH "Path to the sparse solver interface install root")
list(APPEND CMAKE_PREFIX_PATH
  "/home/reidatcheson/sparse_linear_algebra/installs/perfetto"
  "/home/reidatcheson/sparse_linear_algebra/installs/spdlog"
)

find_package(Perfetto CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

add_executable(sparse-solver-gym
  src/benchmarks/laplacian_3d_streaming.cpp
  src/benchmarks/laplacian_1d.cpp
  src/sparse_solver_gym/benchmark.cpp
  src/sparse_solver_gym/cli.cpp
  src/sparse_solver_gym/main.cpp
  src/sparse_solver_gym/orchestrator.cpp
  src/sparse_solver_gym/perfetto_trace.cpp
  src/sparse_solver_gym/tracing_solver.cpp
  src/sparse_solver_gym/worker.cpp
)

target_include_directories(sparse-solver-gym
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${SPARSE_SOLVER_INTERFACE_ROOT}/include"
)

target_link_libraries(sparse-solver-gym
  PRIVATE
    Perfetto::perfetto
    spdlog::spdlog
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

install(TARGETS sparse-solver-gym RUNTIME DESTINATION bin)
