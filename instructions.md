I want a cmake project here that will depend on "sparse solver interface" an initial installation exists here

```
 Installed here:

  /home/reidatcheson/sparse_linear_algebra/installs/sparse_solver_interface/0.1.0

  Headers are under:

  /home/reidatcheson/sparse_linear_algebra/installs/sparse_solver_interface/0.1.0/include

  To use it from another CMake repo for now:

  set(SPARSE_SOLVER_INTERFACE_ROOT
      "/home/reidatcheson/sparse_linear_algebra/installs/sparse_solver_interface/0.1.0")

  target_include_directories(
    your_target
    PRIVATE
      "${SPARSE_SOLVER_INTERFACE_ROOT}/include"
  )

  The install command sequence was:

  cmake -S . -B /tmp/ssi-shared-install-build \
    -DBUILD_TESTING=ON \
    -DCMAKE_INSTALL_PREFIX=/home/reidatcheson/sparse_linear_algebra/installs/sparse_solver_interface/0.1.0

  cmake --build /tmp/ssi-shared-install-build
  ctest --test-dir /tmp/ssi-shared-install-build --output-on-failure
  cmake --install /tmp/ssi-shared-install-build

  Tests passed before install.
```


I want this to primarily export a binary in its install target. the binary will take a shared object (defined by sparse solver interface) and that's the creature that it will benchmark and test. I want it to be easy to 
configure and set up different benchmarks. So we probably want benchmark to be a plugin-able thing (not with the hourglass pattern like sparse solver interface, but possibly with a static registry and benchmark "tags" we can query against at cli runtime). the default behavior is to just run all benchmarks, but tags can help filter. I think some initial tags could be like "light,medium,heavy" for example.


sparse solver interface provides a feature complete interface and it is _only_ these that I wish to test. I want to dlopen a solver shared object and then shove it into our own implementation that instruments every method with perfetto which we installed like this

```
 Installed Perfetto C++ tracing SDK here:

  /home/reidatcheson/sparse_linear_algebra/installs/perfetto

  Contents include:

  - include/perfetto.h
  - include/perfetto.cc
  - lib/libperfetto.a
  - CMake package configs under lib/cmake/perfetto and lib/cmake/Perfetto

  Use it from another CMake project with either spelling:

  find_package(Perfetto CONFIG REQUIRED)
  # or: find_package(perfetto CONFIG REQUIRED)

  target_link_libraries(your_target PRIVATE Perfetto::perfetto)

  Configure downstream projects with:

  cmake -S . -B build -DCMAKE_PREFIX_PATH=/home/reidatcheson/sparse_linear_algebra/installs/perfetto

  I verified this by configuring, building, and running a minimal downstream CMake consumer that includes <perfetto.h>, initializes tracing, registers track events, and links Perfetto::perfetto.
```


we should be able to add benchmark-specific information to these traces like benchmark name, benchmark tags. 
