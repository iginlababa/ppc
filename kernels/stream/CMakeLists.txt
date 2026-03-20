cmake_minimum_required(VERSION 3.20)

# HIP-only builds (AMD) must not require the CUDA toolkit.
# -DSTREAM_ENABLE_HIP=ON is set in the cache before project() runs, so this
# check correctly skips the CUDA language when hipcc is the sole compiler.
if((DEFINED STREAM_ENABLE_HIP AND STREAM_ENABLE_HIP) OR
   (DEFINED STREAM_ENABLE_SYCL AND STREAM_ENABLE_SYCL))
    project(stream_kernels LANGUAGES CXX)
else()
    project(stream_kernels LANGUAGES CXX CUDA)
endif()

# Kokkos must be found before project() declares languages if using the
# Kokkos compiler-launcher approach.  Here we use find_package post-project,
# which works for Kokkos installed via cmake --install (the recommended path).
# Set -DKokkos_ROOT=<prefix> or -DCMAKE_PREFIX_PATH=<prefix> at configure time.

# ── Standards ─────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
# ROCm's hipcc/clang may not be in CMake's known-features list; disable the
# hard REQUIRED check so we fall back to the -std=c++17 flag instead.
if(DEFINED STREAM_ENABLE_HIP AND STREAM_ENABLE_HIP)
    set(CMAKE_CXX_STANDARD_REQUIRED OFF)
    # CMake 3.28 has no compiler-feature database entry for ROCm Clang 17/18.
    # RAJA's BLT cmake calls target_compile_features(blt_hip INTERFACE cxx_std_XX),
    # which aborts the Generate step with "No known features".  Pre-populate the
    # feature lists so CMake accepts the requirement without querying the DB.
    set(CMAKE_CXX_COMPILE_FEATURES
        cxx_std_11 cxx_std_14 cxx_std_17 cxx_std_20
        cxx_alias_templates cxx_auto_type cxx_constexpr cxx_decltype
        cxx_defaulted_functions cxx_deleted_functions cxx_final
        cxx_lambdas cxx_noexcept cxx_nullptr cxx_override
        cxx_range_for cxx_rvalue_references cxx_static_assert
        cxx_thread_local cxx_trailing_return_types cxx_variadic_templates)
    set(CMAKE_CXX11_COMPILE_FEATURES cxx_std_11)
    set(CMAKE_CXX14_COMPILE_FEATURES cxx_std_14)
    set(CMAKE_CXX17_COMPILE_FEATURES cxx_std_17)
    set(CMAKE_CXX20_COMPILE_FEATURES cxx_std_20)
else()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU Architecture ──────────────────────────────────────────────────────────
# Set via -DCMAKE_CUDA_ARCHITECTURES=<arch> or the CUDA_ARCH variable.
# Defaults: sm_80 = A100 (Ampere), sm_90 = H100 (Hopper)
# For AMD (HIP) implementations: set in build_stream.sh via hipcc flags.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)   # A100 default
    message(STATUS "CUDA arch not specified — defaulting to sm_80 (A100)")
endif()
message(STATUS "Building for CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")

# ── Optional: single-precision build ─────────────────────────────────────────
option(STREAM_USE_FLOAT "Use single precision (default: double)" OFF)

# ── CUDA native baseline (abstraction=native on NVIDIA) ──────────────────────
# Skipped when building HIP-only (AMD): no CUDA language is enabled.
if(NOT (DEFINED STREAM_ENABLE_HIP AND STREAM_ENABLE_HIP) AND
   NOT (DEFINED STREAM_ENABLE_SYCL AND STREAM_ENABLE_SYCL))
    add_executable(stream-cuda kernel_stream_cuda.cu)

    target_include_directories(stream-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    set_target_properties(stream-cuda PROPERTIES
        OUTPUT_NAME     "stream-cuda"
        CUDA_SEPARABLE_COMPILATION OFF   # no device linking needed
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    target_compile_options(stream-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            --generate-line-info
            -Xcompiler=-Wall,-Wextra
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
            -Wall
            -Wextra
        >
    )

    if(STREAM_USE_FLOAT)
        target_compile_definitions(stream-cuda PRIVATE STREAM_USE_FLOAT)
        message(STATUS "STREAM precision: float")
    else()
        message(STATUS "STREAM precision: double")
    endif()

    install(TARGETS stream-cuda DESTINATION bin)
else()
    if(STREAM_USE_FLOAT)
        message(STATUS "STREAM precision: float")
    else()
        message(STATUS "STREAM precision: double")
    endif()
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "E1 STREAM Triad — build summary")
message(STATUS "  Targets:    stream-cuda  (+ stream-kokkos if Kokkos found)")
message(STATUS "  CUDA arch:  sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Precision:  ${STREAM_USE_FLOAT}")
message(STATUS "")
message(STATUS "CUDA build:")
message(STATUS "  cmake -S kernels/stream -B build/stream/cuda_nvidia_a100 \\")
message(STATUS "        -DCMAKE_CUDA_ARCHITECTURES=80")
message(STATUS "  cmake --build build/stream/cuda_nvidia_a100 --parallel")
message(STATUS "")
message(STATUS "Kokkos/CUDA build:")
message(STATUS "  cmake -S kernels/stream -B build/stream/kokkos_nvidia_a100 \\")
message(STATUS "        -DKokkos_ROOT=<prefix> \\")
message(STATUS "        -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON")
message(STATUS "  cmake --build build/stream/kokkos_nvidia_a100 --parallel")
message(STATUS "")
message(STATUS "RAJA/CUDA build:")
message(STATUS "  cmake -S kernels/stream -B build/stream/raja_nvidia_a100 \\")
message(STATUS "        -DRAJA_DIR=<prefix>/lib/cmake/raja \\")
message(STATUS "        -DRAJA_ENABLE_CUDA=ON -DRAJA_CUDA_ARCH=sm_80")
message(STATUS "  cmake --build build/stream/raja_nvidia_a100 --parallel")
message(STATUS "")
message(STATUS "HIP build (AMD MI250X, opt-in):")
message(STATUS "  cmake -S kernels/stream -B build/stream/hip_amd_mi250x \\")
message(STATUS "        -DCMAKE_CXX_COMPILER=hipcc \\")
message(STATUS "        -DSTREAM_ENABLE_HIP=ON -DHIP_ARCH=gfx90a")
message(STATUS "  cmake --build build/stream/hip_amd_mi250x --parallel")
message(STATUS "")
message(STATUS "SYCL build (Intel PVC, opt-in):")
message(STATUS "  cmake -S kernels/stream -B build/stream/sycl_intel_pvc \\")
message(STATUS "        -DCMAKE_CXX_COMPILER=icpx -DSTREAM_ENABLE_SYCL=ON")
message(STATUS "  cmake --build build/stream/sycl_intel_pvc --parallel")
message(STATUS "")
message(STATUS "Run with:")
message(STATUS "  ./stream-cuda   --arraysize 268435456 --numtimes 30 --warmup 10")
message(STATUS "  ./stream-kokkos --arraysize 268435456 --numtimes 30 --warmup 10")
message(STATUS "  ./stream-raja   --arraysize 268435456 --numtimes 30 --warmup 10")
message(STATUS "  ./stream-sycl   --arraysize 268435456 --numtimes 30 --warmup 10")
message(STATUS "  ./stream-hip    --arraysize 268435456 --numtimes 30 --warmup 10")
message(STATUS "  ./stream-julia  --arraysize 268435456 --numtimes 30 --warmup 10")
message(STATUS "  ./stream-numba  --arraysize 268435456 --numtimes 30 --warmup 10")
message(STATUS "")

# ── Kokkos abstraction (abstraction=kokkos) ───────────────────────────────────
# Requires Kokkos ≥ 4.0 installed with cmake --install.
# Configure with:
#   cmake -S kernels/stream -B build/stream/kokkos_nvidia_a100       \
#         -DKokkos_ROOT=/path/to/kokkos/install                      \
#         -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON          \
#         -DCMAKE_CXX_COMPILER=<kokkos-install>/bin/nvcc_wrapper
#
# For OpenMP (CPU):
#   cmake ... -DKokkos_ENABLE_OPENMP=ON
#
# For HIP (AMD):
#   cmake ... -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    message(STATUS "Kokkos ${Kokkos_VERSION} found — building stream-kokkos")
    add_executable(stream-kokkos kernel_stream_kokkos.cpp)
    target_link_libraries(stream-kokkos PRIVATE Kokkos::kokkos)
    target_include_directories(stream-kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(stream-kokkos PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>: -O3 -Wall -Wextra>
    )
    if(STREAM_USE_FLOAT)
        target_compile_definitions(stream-kokkos PRIVATE STREAM_USE_FLOAT)
    endif()
    install(TARGETS stream-kokkos DESTINATION bin)
else()
    message(STATUS "Kokkos not found — stream-kokkos target skipped")
    message(STATUS "  Install Kokkos and re-run with -DKokkos_ROOT=<prefix>")
endif()

# ── RAJA abstraction (abstraction=raja) ──────────────────────────────────────
# Requires RAJA ≥ 2022.10 installed with cmake --install.
# Configure with:
#   cmake -S kernels/stream -B build/stream/raja_nvidia_a100   \
#         -DRAJA_DIR=/path/to/raja/install/lib/cmake/raja      \
#         -DRAJA_ENABLE_CUDA=ON -DRAJA_CUDA_ARCH=sm_80
#
# RAJA propagates RAJA_ENABLE_CUDA (and the --expt-extended-lambda flag)
# through its interface target, so no extra flags are needed here.
#
# For OpenMP (CPU):
#   cmake ... -DRAJA_ENABLE_OPENMP=ON
#
# For HIP (AMD):
#   cmake ... -DRAJA_ENABLE_HIP=ON -DRAJA_HIP_ARCH=gfx90a
find_package(RAJA QUIET)
if(RAJA_FOUND)
    message(STATUS "RAJA ${RAJA_VERSION} found — building stream-raja")

    # When RAJA's CUDA backend is active, kernel_stream_raja.cpp must be
    # compiled by nvcc so that __CUDACC__ is defined and RAJA_LAMBDA expands
    # to [=] __device__ __host__ (extended-lambda syntax).  Detect this by
    # inspecting RAJA's interface link libraries for the "cuda" target.
    get_target_property(_raja_iface_libs RAJA INTERFACE_LINK_LIBRARIES)
    if("cuda" IN_LIST _raja_iface_libs)
        set_source_files_properties(kernel_stream_raja.cpp PROPERTIES LANGUAGE CUDA)
        message(STATUS "  RAJA CUDA backend detected — compiling kernel as CUDA TU")
    endif()

    add_executable(stream-raja kernel_stream_raja.cpp)
    target_link_libraries(stream-raja PRIVATE RAJA)
    target_include_directories(stream-raja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(stream-raja PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>: -O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            --generate-line-info
            --expt-extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-Wall,-Wextra
        >
    )
    if(STREAM_USE_FLOAT)
        target_compile_definitions(stream-raja PRIVATE STREAM_USE_FLOAT)
    endif()
    install(TARGETS stream-raja DESTINATION bin)
else()
    message(STATUS "RAJA not found — stream-raja target skipped")
    message(STATUS "  Install RAJA and re-run with -DRAJA_DIR=<prefix>/lib/cmake/raja")
endif()

# ── SYCL abstraction (abstraction=sycl) ──────────────────────────────────────
# Requires a SYCL 2020-conformant toolchain:
#   Intel DPC++ (oneAPI ≥ 2022.3):
#     source /opt/intel/oneapi/setvars.sh
#     cmake -S kernels/stream -B build/stream/sycl_intel_pvc \
#           -DCMAKE_CXX_COMPILER=icpx \
#           -DSYCL_TARGET_BACKEND=opencl
#
#   Intel DPC++ targeting NVIDIA A100 (CUDA plugin):
#     cmake -S kernels/stream -B build/stream/sycl_nvidia_a100 \
#           -DCMAKE_CXX_COMPILER=clang++ \
#           -DSYCL_TARGETS="nvptx64-nvidia-cuda" \
#           -DSYCL_CUDA_ARCH=sm_80
#
#   AdaptiveCpp / hipSYCL targeting AMD:
#     cmake -S kernels/stream -B build/stream/sycl_amd_mi250x \
#           -DCMAKE_CXX_COMPILER=acpp \
#           -DHIPSYCL_TARGETS="hip:gfx90a"
#
# SYCL does not have a standard CMake find_package target; detection is done
# by checking for the compiler flag or the presence of the sycl header.
# We expose two CMake options to opt in:
#   -DSTREAM_ENABLE_SYCL=ON         (required; off by default)
#   -DSTREAM_SYCL_COMPAT=ON         (use <CL/sycl.hpp> for older DPC++)
#   -DSYCL_FLAGS="-fsycl;-fsycl-targets=nvptx64-nvidia-cuda"  (extra flags)

option(STREAM_ENABLE_SYCL "Build stream-sycl (requires SYCL compiler)" OFF)
set(SYCL_FLAGS "" CACHE STRING "Extra SYCL compiler/linker flags (semicolon-separated)")

if(STREAM_ENABLE_SYCL)
    message(STATUS "SYCL enabled — building stream-sycl")
    add_executable(stream-sycl kernel_stream_sycl.cpp)
    target_include_directories(stream-sycl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Base SYCL flag — required for DPC++ (icpx/clang++) but rejected by AdaptiveCpp (acpp).
    # Detect acpp by compiler name; acpp manages SYCL translation via --acpp-targets instead.
    get_filename_component(_sycl_compiler_name "${CMAKE_CXX_COMPILER}" NAME)
    if(_sycl_compiler_name MATCHES "acpp")
        message(STATUS "  SYCL: AdaptiveCpp compiler detected — omitting -fsycl (use --acpp-targets in SYCL_FLAGS)")
        target_compile_options(stream-sycl PRIVATE -O3 -Wall -Wextra)
    else()
        target_compile_options(stream-sycl PRIVATE -O3 -Wall -Wextra -fsycl)
        target_link_options(stream-sycl PRIVATE -fsycl)
    endif()

    # Extra backend-targeting flags (e.g. -fsycl-targets=nvptx64-nvidia-cuda)
    if(SYCL_FLAGS)
        separate_arguments(sycl_flag_list UNIX_COMMAND "${SYCL_FLAGS}")
        target_compile_options(stream-sycl PRIVATE ${sycl_flag_list})
        target_link_options(stream-sycl PRIVATE ${sycl_flag_list})
    endif()

    if(STREAM_SYCL_COMPAT)
        target_compile_definitions(stream-sycl PRIVATE STREAM_SYCL_COMPAT)
        message(STATUS "  SYCL: using legacy <CL/sycl.hpp> header")
    endif()

    if(STREAM_USE_FLOAT)
        target_compile_definitions(stream-sycl PRIVATE STREAM_USE_FLOAT)
    endif()

    install(TARGETS stream-sycl DESTINATION bin)
else()
    message(STATUS "SYCL disabled — stream-sycl target skipped")
    message(STATUS "  Enable with -DSTREAM_ENABLE_SYCL=ON and set CXX compiler to icpx/acpp")
endif()

# ── HIP native baseline (abstraction=native on AMD) ──────────────────────────
# Requires ROCm ≥ 5.0 and hipcc on PATH.  The CUDA and HIP targets share
# the same CMakeLists.txt; HIP is opt-in to avoid breaking CUDA-only builds.
#
# Configure with:
#   cmake -S kernels/stream -B build/stream/hip_amd_mi250x \
#         -DCMAKE_CXX_COMPILER=hipcc \
#         -DSTREAM_ENABLE_HIP=ON \
#         -DHIP_ARCH=gfx90a
#
# For MI100 (gfx908):
#   cmake ... -DHIP_ARCH=gfx908
#
# For MI300X (gfx942):
#   cmake ... -DHIP_ARCH=gfx942

option(STREAM_ENABLE_HIP "Build stream-hip (requires hipcc)" OFF)
set(HIP_ARCH "gfx90a" CACHE STRING "AMD GPU architecture target (default: MI250X)")

if(STREAM_ENABLE_HIP)
    message(STATUS "HIP enabled — building stream-hip (arch=${HIP_ARCH})")

    # Enable HIP language support; CMake ≥ 3.21 supports LANGUAGES HIP natively.
    # For older CMake, we rely on hipcc being the CXX compiler directly.
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        enable_language(HIP)
        set_source_files_properties(kernel_stream_hip.cpp PROPERTIES LANGUAGE HIP)
    endif()

    add_executable(stream-hip kernel_stream_hip.cpp)
    target_include_directories(stream-hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # --offload-arch: tells hipcc which AMD GPU ISA to generate
    target_compile_options(stream-hip PRIVATE
        -O3
        --offload-arch=${HIP_ARCH}
        -Wall
        -Wextra
    )
    target_link_options(stream-hip PRIVATE
        --offload-arch=${HIP_ARCH}
    )

    if(STREAM_USE_FLOAT)
        target_compile_definitions(stream-hip PRIVATE STREAM_USE_FLOAT)
    endif()

    install(TARGETS stream-hip DESTINATION bin)
else()
    message(STATUS "HIP disabled — stream-hip target skipped")
    message(STATUS "  Enable with -DSTREAM_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc")
endif()

# ── Julia abstraction (abstraction=julia) ────────────────────────────────────
# Julia does not compile to a native binary.  The "target" is a shell wrapper
# (stream-julia) that invokes julia --project=... kernel_stream_julia.jl.
# The wrapper is copied into the build directory so that run_stream.sh finds
# it at the expected path: build/stream/julia_<platform>/stream-julia.
#
# Requires:
#   - Julia ≥ 1.9 on PATH (or set JULIA_BIN environment variable)
#   - CUDA.jl ≥ 5.0 installed in the project environment:
#       julia --project=kernels/stream/ -e 'using Pkg; Pkg.instantiate()'
#
# Enable with:
#   cmake -S kernels/stream -B build/stream/julia_nvidia_a100 \
#         -DSTREAM_ENABLE_JULIA=ON

option(STREAM_ENABLE_JULIA "Install stream-julia wrapper (requires Julia + CUDA.jl)" OFF)

if(STREAM_ENABLE_JULIA)
    message(STATUS "Julia enabled — installing stream-julia wrapper")

    # Copy the wrapper and script into the build directory together so that
    # the wrapper's relative path resolution (SCRIPT_DIR) finds the .jl file.
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/stream-julia
        ${CMAKE_CURRENT_BINARY_DIR}/stream-julia
        COPYONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_stream_julia.jl
        ${CMAKE_CURRENT_BINARY_DIR}/kernel_stream_julia.jl
        COPYONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Project.toml
        ${CMAKE_CURRENT_BINARY_DIR}/Project.toml
        COPYONLY
    )

    # Ensure the wrapper is executable after configure_file copies it
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/stream-julia DESTINATION bin)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_stream_julia.jl
        ${CMAKE_CURRENT_SOURCE_DIR}/Project.toml
        DESTINATION bin)

    message(STATUS "  Julia: run 'julia --project=kernels/stream/ -e \"using Pkg; Pkg.instantiate()\"' to install deps")
else()
    message(STATUS "Julia disabled — stream-julia target skipped")
    message(STATUS "  Enable with -DSTREAM_ENABLE_JULIA=ON (requires Julia ≥ 1.9)")
endif()

# ── Numba abstraction (abstraction=numba) ────────────────────────────────────
# Python does not compile to a native binary.  The "target" is a shell wrapper
# (stream-numba) that invokes python3 kernel_stream_numba.py.
#
# Requires:
#   - Python ≥ 3.9 with numba ≥ 0.57 and numpy on PATH
#       pip install numba numpy
#   - CUDA toolkit accessible to Numba (set CUDA_HOME if not auto-detected)
#
# Enable with:
#   cmake -S kernels/stream -B build/stream/numba_nvidia_a100 \
#         -DSTREAM_ENABLE_NUMBA=ON

option(STREAM_ENABLE_NUMBA "Install stream-numba wrapper (requires Python + Numba)" OFF)

if(STREAM_ENABLE_NUMBA)
    message(STATUS "Numba enabled — installing stream-numba wrapper")

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/stream-numba
        ${CMAKE_CURRENT_BINARY_DIR}/stream-numba
        COPYONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_stream_numba.py
        ${CMAKE_CURRENT_BINARY_DIR}/kernel_stream_numba.py
        COPYONLY
    )

    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/stream-numba DESTINATION bin)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernel_stream_numba.py DESTINATION bin)

    message(STATUS "  Numba: install deps with 'pip install numba numpy'")
else()
    message(STATUS "Numba disabled — stream-numba target skipped")
    message(STATUS "  Enable with -DSTREAM_ENABLE_NUMBA=ON (requires Python ≥ 3.9 + numba)")
endif()
