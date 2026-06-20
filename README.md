# sparse-solver-gym
Test suite for sparse linear solvers

## Build

```sh
cmake -S . -B build
cmake --build build
```

The default CMake configuration looks for the currently installed local copies
of sparse solver interface, Perfetto, and spdlog.

## Usage

List benchmarks:

```sh
./build/sparse-solver-gym list
./build/sparse-solver-gym list --tag light --tag fp64
```

Run benchmarks against an SSI solver plugin shared object:

```sh
./build/sparse-solver-gym run --solver /path/to/solver_plugin.so
```

By default, trace files are written under `/tmp/sparse-solver-gym/<timestamp>-<pid>/`
and the location is logged. Use `--output-dir DIR` for a run directory, or
`--trace-out FILE` when selecting exactly one benchmark.

Each selected benchmark is run in a worker subprocess. The orchestrator records
failed, crashed, and timed-out workers and continues running the remaining
benchmarks. Tag filters currently use AND semantics.

When a solver reports unsupported sparse problem properties through SSI
`check_support`, the orchestrator marks that benchmark `unsupported` and skips
the worker process. Unsupported benchmarks are reported in the log with the
solver-provided reason, but do not make the overall run fail.
