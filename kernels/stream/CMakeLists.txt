cmake_minimum_required(VERSION 3.20)

project(stream_kernels LANGUAGES CXX CUDA)

# Kokkos must be found before project() declares languages if using the
# Kokkos compiler-launcher approach.  Here we use find_package post-project,
# which works for Kokkos installed via cmake --install (the recommended path).
# Set -DKokkos_ROOT=<prefix> or -DCMAKE_PREFIX_PATH=<prefix> at configure time.

# ── Standards ─────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS stream-cuda DESTINATION bin)

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
    add_executable(stream-raja kernel_stream_raja.cpp)
    target_link_libraries(stream-raja PRIVATE RAJA)
    target_include_directories(stream-raja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(stream-raja PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>: -O3 -Wall -Wextra>
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

    # Base SYCL flag — works for DPC++ (icpx/clang++) and AdaptiveCpp (acpp)
    target_compile_options(stream-sycl PRIVATE -O3 -Wall -Wextra -fsycl)
    target_link_options(stream-sycl PRIVATE -fsycl)

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

# ── Placeholder targets for future implementations ────────────────────────────
# Each target follows the naming convention: stream-<abstraction>

# stream-hip       → kernel_stream_hip.cpp        (AMD HIP backend)
# stream-julia     → run via Julia script          (CUDA.jl / AMDGPU.jl)
# stream-numba     → run via Python script         (Numba CUDA)
