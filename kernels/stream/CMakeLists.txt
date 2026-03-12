cmake_minimum_required(VERSION 3.20)

project(stream_kernels LANGUAGES CXX CUDA)

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
message(STATUS "E1 STREAM Triad — CUDA build summary")
message(STATUS "  Target:     stream-cuda")
message(STATUS "  CUDA arch:  sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Precision:  ${STREAM_USE_FLOAT}")
message(STATUS "")
message(STATUS "Build with:")
message(STATUS "  cmake -S kernels/stream -B build/stream/cuda_nvidia_a100 \\")
message(STATUS "        -DCMAKE_CUDA_ARCHITECTURES=80")
message(STATUS "  cmake --build build/stream/cuda_nvidia_a100 --parallel")
message(STATUS "")
message(STATUS "Run with:")
message(STATUS "  ./stream-cuda --arraysize 268435456 --numtimes 30 --warmup 10")
message(STATUS "")

# ── Placeholder targets for future implementations ────────────────────────────
# These will be added as separate source files are implemented.
# Each target follows the naming convention: stream-<abstraction>

# stream-hip       → kernel_stream_hip.cpp        (AMD HIP backend)
# stream-kokkos    → kernel_stream_kokkos.cpp      (Kokkos backend)
# stream-raja      → kernel_stream_raja.cpp        (RAJA backend)
# stream-sycl      → kernel_stream_sycl.cpp        (SYCL / DPC++ backend)
# stream-julia     → run via Julia script          (CUDA.jl / AMDGPU.jl)
# stream-numba     → run via Python script         (Numba CUDA)
