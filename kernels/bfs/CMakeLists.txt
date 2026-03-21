cmake_minimum_required(VERSION 3.20)
project(e6_bfs LANGUAGES CXX)

# ── E6 DESIGN DECISIONS ───────────────────────────────────────────────────────
# [D4] One thread per frontier vertex (scatter); compact via Thrust/parallel_scan.
# [D7] Adaptive warmup logic is inside each kernel binary.
# HIP: conditional on -DBFS_ENABLE_HIP=ON; disables CUDA language entirely.
# CUDA: built when HIP not requested.
# SYCL: conditional on -DBFS_ENABLE_SYCL=ON; detects acpp vs icpx/clang++.
# Kokkos/RAJA: conditional on find_package.
# numba: UNSUPPORTED_CC120 (Blackwell); not built on AMD (numba-hip experimental).
# ─────────────────────────────────────────────────────────────────────────────

option(BFS_ENABLE_HIP   "Build HIP bfs target (AMD/ROCm)"  OFF)
option(BFS_ENABLE_SYCL  "Build SYCL bfs target"            OFF)
option(BFS_ENABLE_JULIA "Build Julia bfs wrapper"           ON)

# ── CUDA (skipped when HIP requested) ─────────────────────────────────────────
if(NOT BFS_ENABLE_HIP)
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
        find_program(_GXX g++)
        if(_GXX)
            set(CMAKE_CUDA_HOST_COMPILER "${_GXX}" CACHE STRING "")
        endif()
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(_BFS_HAVE_CUDA TRUE)
else()
    set(_BFS_HAVE_CUDA FALSE)
endif()

set(BFS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ── bfs-cuda (native baseline, NVIDIA + Thrust) ────────────────────────────────
if(_BFS_HAVE_CUDA)
    add_executable(bfs-cuda kernel_bfs_cuda.cu)
    target_include_directories(bfs-cuda PRIVATE ${BFS_INCLUDE_DIR})
    target_link_libraries(bfs-cuda PRIVATE CUDA::cudart)
    target_compile_options(bfs-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-O3,-march=native>)
    set_target_properties(bfs-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ── bfs-hip (native baseline, AMD + ROCThrust) ─────────────────────────────────
if(BFS_ENABLE_HIP)
    find_package(hip REQUIRED)
    add_executable(bfs-hip kernel_bfs_hip.cpp)
    target_include_directories(bfs-hip PRIVATE ${BFS_INCLUDE_DIR})
    target_link_libraries(bfs-hip PRIVATE hip::device)
    target_compile_options(bfs-hip PRIVATE -O3 -ffast-math -Wall -Wextra)
endif()

# ── bfs-kokkos ────────────────────────────────────────────────────────────────
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    get_target_property(_kokkos_iface Kokkos::kokkos INTERFACE_LINK_LIBRARIES)
    if(_BFS_HAVE_CUDA)
        if("cuda" IN_LIST _kokkos_iface OR "CUDA::cudart" IN_LIST _kokkos_iface)
            set_source_files_properties(kernel_bfs_kokkos.cpp PROPERTIES LANGUAGE CUDA)
            message(STATUS "  BFS: Kokkos CUDA backend — compiling as CUDA TU")
        endif()
    elseif(BFS_ENABLE_HIP)
        message(STATUS "  BFS: Kokkos HIP backend — compiling as CXX")
    endif()
    add_executable(bfs-kokkos kernel_bfs_kokkos.cpp)
    target_include_directories(bfs-kokkos PRIVATE ${BFS_INCLUDE_DIR})
    if(_BFS_HAVE_CUDA)
        target_link_libraries(bfs-kokkos PRIVATE Kokkos::kokkos CUDA::cudart)
        target_compile_options(bfs-kokkos PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:
                -O3 --use_fast_math --extended-lambda
                --expt-relaxed-constexpr
                -Xcompiler=-Wall,-Wextra>)
    else()
        target_link_libraries(bfs-kokkos PRIVATE Kokkos::kokkos hip::device)
        target_compile_options(bfs-kokkos PRIVATE -O3 -Wall -Wextra)
    endif()
else()
    message(STATUS "  BFS: Kokkos not found — skipping bfs-kokkos")
endif()

# ── bfs-raja ──────────────────────────────────────────────────────────────────
find_package(RAJA QUIET)
if(RAJA_FOUND)
    get_target_property(_raja_iface RAJA INTERFACE_LINK_LIBRARIES)
    if(_BFS_HAVE_CUDA)
        if("cuda" IN_LIST _raja_iface OR "CUDA::cudart" IN_LIST _raja_iface)
            set_source_files_properties(kernel_bfs_raja.cpp PROPERTIES LANGUAGE CUDA)
            message(STATUS "  BFS: RAJA CUDA backend — compiling as CUDA TU")
        endif()
    elseif(BFS_ENABLE_HIP)
        message(STATUS "  BFS: RAJA HIP backend — compiling as CXX")
    endif()
    add_executable(bfs-raja kernel_bfs_raja.cpp)
    target_include_directories(bfs-raja PRIVATE ${BFS_INCLUDE_DIR})
    if(_BFS_HAVE_CUDA)
        target_link_libraries(bfs-raja PRIVATE RAJA CUDA::cudart)
        target_compile_options(bfs-raja PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:
                -O3 --use_fast_math --extended-lambda
                --expt-relaxed-constexpr
                -allow-unsupported-compiler
                -Xcompiler=-Wall,-Wextra>)
    else()
        target_link_libraries(bfs-raja PRIVATE RAJA hip::device)
        target_compile_options(bfs-raja PRIVATE -O3 -Wall -Wextra)
    endif()
else()
    message(STATUS "  BFS: RAJA not found — skipping bfs-raja")
endif()

# ── bfs-sycl ──────────────────────────────────────────────────────────────────
if(BFS_ENABLE_SYCL)
    add_executable(bfs-sycl kernel_bfs_sycl.cpp)
    target_include_directories(bfs-sycl PRIVATE ${BFS_INCLUDE_DIR})
    # AdaptiveCpp (acpp/syclcc) uses --acpp-targets instead of -fsycl
    if(CMAKE_CXX_COMPILER MATCHES "acpp|syclcc")
        target_compile_options(bfs-sycl PRIVATE -O3)
    else()
        target_compile_options(bfs-sycl PRIVATE -fsycl -O3)
        target_link_options(bfs-sycl PRIVATE -fsycl)
    endif()
    if(DEFINED SYCL_FLAGS)
        separate_arguments(_sycl_flags UNIX_COMMAND "${SYCL_FLAGS}")
        target_compile_options(bfs-sycl PRIVATE ${_sycl_flags})
        target_link_options(bfs-sycl PRIVATE ${_sycl_flags})
    endif()
else()
    message(STATUS "  BFS: SYCL disabled (pass -DBFS_ENABLE_SYCL=ON to enable)")
endif()

# ── bfs-julia wrapper ─────────────────────────────────────────────────────────
if(BFS_ENABLE_JULIA)
    set(_julia_src  "${CMAKE_CURRENT_SOURCE_DIR}/kernel_bfs_julia.jl")
    set(_julia_wrap "${CMAKE_CURRENT_BINARY_DIR}/bfs-julia")
    file(WRITE "${_julia_wrap}"
"#!/usr/bin/env bash
# Auto-generated by CMake — do not edit.
JULIA_BIN=\"\${JULIA_BIN:-julia}\"
exec \"\${JULIA_BIN}\" --startup-file=no --project=@. \\
     \"${_julia_src}\" \"\$@\"
")
    file(CHMOD "${_julia_wrap}" PERMISSIONS
         OWNER_READ OWNER_WRITE OWNER_EXECUTE
         GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    message(STATUS "  BFS: Julia wrapper → ${_julia_wrap}")
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "E6 BFS build summary:")
if(_BFS_HAVE_CUDA)
    message(STATUS "  bfs-cuda    → native CSR scatter + Thrust compact (CUDA)")
endif()
if(BFS_ENABLE_HIP)
    message(STATUS "  bfs-hip     → native CSR scatter + ROCThrust compact (HIP/AMD)")
endif()
if(Kokkos_FOUND)
    message(STATUS "  bfs-kokkos  → RangePolicy scatter + parallel_scan compact")
endif()
if(RAJA_FOUND)
    message(STATUS "  bfs-raja    → forall scatter + exclusive_scan compact")
endif()
if(BFS_ENABLE_SYCL)
    message(STATUS "  bfs-sycl    → nd_range<1> scatter + atomic counter compact")
endif()
if(BFS_ENABLE_JULIA)
    message(STATUS "  bfs-julia   → @cuda/@roc scatter + cumsum! compact")
endif()
