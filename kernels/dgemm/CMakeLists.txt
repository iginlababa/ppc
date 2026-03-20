cmake_minimum_required(VERSION 3.20)
project(e2_dgemm LANGUAGES CXX)

# ── E2 DESIGN DECISIONS ───────────────────────────────────────────────────────
# [D2] TILE=32 baked into dgemm_common.h; no CMake option needed.
# [D5] native_cublas: same binary as native (dgemm-cuda) with --mode cublas.
#      native_rocblas: same binary as native (dgemm-hip) with --mode rocblas.
# CUDA enabled only when at least one CUDA-backend target is requested.
# RAJA compiled as CUDA TU when RAJA CUDA backend detected (same logic as stream).
# SYCL: conditional on -DDGEMM_ENABLE_SYCL=ON and a working SYCL compiler.
# HIP: conditional on -DDGEMM_ENABLE_HIP=ON; disables CUDA language entirely.
# ─────────────────────────────────────────────────────────────────────────────

# ── Options ───────────────────────────────────────────────────────────────────
option(DGEMM_ENABLE_HIP   "Build HIP DGEMM target (AMD/ROCm)"  OFF)
option(DGEMM_ENABLE_SYCL  "Build SYCL DGEMM target"            OFF)
option(DGEMM_ENABLE_JULIA "Build Julia DGEMM wrapper"           ON)
option(DGEMM_ENABLE_NUMBA "Build Numba DGEMM wrapper"           ON)

# ── CUDA (skipped when HIP requested or nvcc not found) ───────────────────────
# Julia/Numba-only builds on non-NVIDIA machines should not fail due to missing
# nvcc.  Check for nvcc first; CUDA targets are only added when it is found.
if(NOT DGEMM_ENABLE_HIP)
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        find_program(CMAKE_CUDA_COMPILER nvcc
            PATHS /usr/local/cuda/bin /usr/local/cuda-12/bin /usr/local/cuda-11/bin
            DOC "NVIDIA CUDA compiler")
    endif()
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
            find_program(_GXX g++)
            if(_GXX)
                set(CMAKE_CUDA_HOST_COMPILER "${_GXX}" CACHE STRING "")
            endif()
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(_DGEMM_HAVE_CUDA TRUE)
    else()
        message(STATUS "  DGEMM: nvcc not found — CUDA targets skipped")
        set(_DGEMM_HAVE_CUDA FALSE)
    endif()
else()
    set(_DGEMM_HAVE_CUDA FALSE)
endif()

# ── Common include ────────────────────────────────────────────────────────────
set(DGEMM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ── dgemm-cuda (native tiled + --mode cublas ceiling) ────────────────────────
if(_DGEMM_HAVE_CUDA)
    add_executable(dgemm-cuda kernel_dgemm_cuda.cu)
    target_include_directories(dgemm-cuda PRIVATE ${DGEMM_INCLUDE_DIR})
    target_link_libraries(dgemm-cuda PRIVATE CUDA::cublas CUDA::cudart)
    target_compile_options(dgemm-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -Xcompiler=-Wall,-Wextra>)
    set_target_properties(dgemm-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ── dgemm-hip (native tiled + --mode rocblas ceiling) ────────────────────────
if(DGEMM_ENABLE_HIP)
    find_package(hip REQUIRED)
    find_package(rocblas REQUIRED)
    add_executable(dgemm-hip kernel_dgemm_hip.cpp)
    target_include_directories(dgemm-hip PRIVATE ${DGEMM_INCLUDE_DIR})
    target_link_libraries(dgemm-hip PRIVATE hip::device roc::rocblas)
    target_compile_options(dgemm-hip PRIVATE -O3 -Wall -Wextra)
endif()

# ── dgemm-kokkos ──────────────────────────────────────────────────────────────
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    get_target_property(_kokkos_iface Kokkos::kokkos INTERFACE_LINK_LIBRARIES)
    if(_DGEMM_HAVE_CUDA)
        if("cuda" IN_LIST _kokkos_iface OR "CUDA::cudart" IN_LIST _kokkos_iface)
            set_source_files_properties(kernel_dgemm_kokkos.cpp PROPERTIES LANGUAGE CUDA)
            message(STATUS "  DGEMM: Kokkos CUDA backend — compiling as CUDA TU")
        endif()
    elseif(DGEMM_ENABLE_HIP)
        message(STATUS "  DGEMM: Kokkos HIP backend — compiling as CXX")
    endif()
    add_executable(dgemm-kokkos kernel_dgemm_kokkos.cpp)
    target_include_directories(dgemm-kokkos PRIVATE ${DGEMM_INCLUDE_DIR})
    if(_DGEMM_HAVE_CUDA)
        target_link_libraries(dgemm-kokkos PRIVATE Kokkos::kokkos CUDA::cudart)
    else()
        target_link_libraries(dgemm-kokkos PRIVATE Kokkos::kokkos)
    endif()
    target_compile_options(dgemm-kokkos PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -Xcompiler=-Wall,-Wextra>)
else()
    message(STATUS "  DGEMM: Kokkos not found — skipping dgemm-kokkos")
endif()

# ── dgemm-raja ────────────────────────────────────────────────────────────────
# Try cmake config first; fall back to direct-path compilation if camp/BLT
# cmake configs are broken (common with build-tree RAJA installs on new GCC).
# CMAKE_DISABLE_FIND_PACKAGE_RAJA also disables the fallback search below.
find_package(RAJA QUIET)
if(RAJA_FOUND)
    get_target_property(_raja_iface RAJA INTERFACE_LINK_LIBRARIES)
    if(_DGEMM_HAVE_CUDA)
        if("cuda" IN_LIST _raja_iface OR "CUDA::cudart" IN_LIST _raja_iface)
            set_source_files_properties(kernel_dgemm_raja.cpp PROPERTIES LANGUAGE CUDA)
            message(STATUS "  DGEMM: RAJA CUDA backend — compiling as CUDA TU")
        endif()
    elseif(DGEMM_ENABLE_HIP)
        message(STATUS "  DGEMM: RAJA HIP backend — compiling as CXX")
    endif()
    add_executable(dgemm-raja kernel_dgemm_raja.cpp)
    target_include_directories(dgemm-raja PRIVATE ${DGEMM_INCLUDE_DIR})
    if(_DGEMM_HAVE_CUDA)
        target_link_libraries(dgemm-raja PRIVATE RAJA CUDA::cudart)
    else()
        target_link_libraries(dgemm-raja PRIVATE RAJA)
    endif()
    target_compile_options(dgemm-raja PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -Xcompiler=-Wall,-Wextra>)
elseif(NOT CMAKE_DISABLE_FIND_PACKAGE_RAJA)
    # Fallback: direct path compilation (bypasses broken camp/BLT cmake configs).
    if(NOT DEFINED RAJA_INSTALL_PREFIX)
        foreach(_p
                "/home/obalola/projects/raja/install"
                "/usr/local"
                "/opt/raja")
            if(EXISTS "${_p}/include/RAJA/RAJA.hpp")
                set(RAJA_INSTALL_PREFIX "${_p}")
                break()
            endif()
        endforeach()
    endif()
    if(DEFINED RAJA_INSTALL_PREFIX AND EXISTS "${RAJA_INSTALL_PREFIX}/include/RAJA/RAJA.hpp")
        message(STATUS "  DGEMM: RAJA cmake config unavailable — using direct path ${RAJA_INSTALL_PREFIX}")
        if(_DGEMM_HAVE_CUDA)
            set_source_files_properties(kernel_dgemm_raja.cpp PROPERTIES LANGUAGE CUDA)
        endif()
        add_executable(dgemm-raja kernel_dgemm_raja.cpp)
        target_include_directories(dgemm-raja PRIVATE
            ${DGEMM_INCLUDE_DIR}
            ${RAJA_INSTALL_PREFIX}/include)
        if(_DGEMM_HAVE_CUDA)
            target_link_libraries(dgemm-raja PRIVATE
                ${RAJA_INSTALL_PREFIX}/lib/libRAJA.a
                ${RAJA_INSTALL_PREFIX}/lib/libcamp.a
                CUDA::cudart)
            target_compile_options(dgemm-raja PRIVATE
                -O3 --use_fast_math --generate-line-info
                --expt-extended-lambda --expt-relaxed-constexpr
                -allow-unsupported-compiler
                -Xcompiler=-Wall,-Wextra)
        else()
            target_link_libraries(dgemm-raja PRIVATE
                ${RAJA_INSTALL_PREFIX}/lib/libRAJA.a
                ${RAJA_INSTALL_PREFIX}/lib/libcamp.a)
            target_compile_options(dgemm-raja PRIVATE -O3 -Wall -Wextra)
        endif()
    else()
        message(STATUS "  DGEMM: RAJA not found — skipping dgemm-raja")
    endif()
endif()

# ── dgemm-sycl ────────────────────────────────────────────────────────────────
if(DGEMM_ENABLE_SYCL)
    # Detect acpp (AdaptiveCpp) — it rejects -fsycl (Intel DPC++ flag)
    get_filename_component(_sycl_compiler_name "${CMAKE_CXX_COMPILER}" NAME)
    if(_sycl_compiler_name MATCHES "acpp|syclcc")
        set(_is_acpp TRUE)
    else()
        set(_is_acpp FALSE)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
            message(WARNING "  DGEMM SYCL: CMAKE_CXX_COMPILER may not support -fsycl. "
                            "Pass -DCMAKE_CXX_COMPILER=icpx or clang++.")
        endif()
    endif()
    add_executable(dgemm-sycl kernel_dgemm_sycl.cpp)
    target_include_directories(dgemm-sycl PRIVATE ${DGEMM_INCLUDE_DIR})
    if(NOT _is_acpp)
        target_compile_options(dgemm-sycl PRIVATE -fsycl -O3)
        target_link_options(dgemm-sycl PRIVATE -fsycl)
    else()
        target_compile_options(dgemm-sycl PRIVATE -O3)
    endif()
    # SYCL_FLAGS from build script adds backend-specific flags (e.g. --acpp-targets=...)
    if(DEFINED SYCL_FLAGS)
        separate_arguments(_sycl_flags UNIX_COMMAND "${SYCL_FLAGS}")
        target_compile_options(dgemm-sycl PRIVATE ${_sycl_flags})
        target_link_options(dgemm-sycl PRIVATE ${_sycl_flags})
    endif()
else()
    message(STATUS "  DGEMM: SYCL disabled (pass -DDGEMM_ENABLE_SYCL=ON to enable)")
endif()

# ── dgemm-julia wrapper ───────────────────────────────────────────────────────
if(DGEMM_ENABLE_JULIA)
    set(_julia_src "${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_julia.jl")
    set(_julia_wrapper "${CMAKE_CURRENT_BINARY_DIR}/dgemm-julia")
    file(WRITE "${_julia_wrapper}"
"#!/usr/bin/env bash
# Auto-generated by CMake — do not edit.
JULIA_BIN=\"\${JULIA_BIN:-julia}\"
exec \"\${JULIA_BIN}\" --startup-file=no --project=\"${CMAKE_CURRENT_SOURCE_DIR}\" \\
     \"${_julia_src}\" \"\$@\"
")
    file(CHMOD "${_julia_wrapper}" PERMISSIONS
         OWNER_READ OWNER_WRITE OWNER_EXECUTE
         GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    message(STATUS "  DGEMM: Julia wrapper → ${_julia_wrapper}")
endif()

# ── dgemm-numba wrapper ───────────────────────────────────────────────────────
if(DGEMM_ENABLE_NUMBA)
    set(_numba_src "${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_numba.py")
    set(_numba_wrapper "${CMAKE_CURRENT_BINARY_DIR}/dgemm-numba")
    file(WRITE "${_numba_wrapper}"
"#!/usr/bin/env bash
# Auto-generated by CMake — do not edit.
PYTHON_BIN=\"\${PYTHON_BIN:-python3}\"
exec \"\${PYTHON_BIN}\" \"${_numba_src}\" \"\$@\"
")
    file(CHMOD "${_numba_wrapper}" PERMISSIONS
         OWNER_READ OWNER_WRITE OWNER_EXECUTE
         GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    message(STATUS "  DGEMM: Numba wrapper → ${_numba_wrapper}")
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "E2 DGEMM build summary:")
if(_DGEMM_HAVE_CUDA)
    message(STATUS "  dgemm-cuda    → always (CUDA tiled + --mode cublas ceiling)")
elseif(DGEMM_ENABLE_HIP)
    message(STATUS "  dgemm-hip     → always (HIP tiled + --mode rocblas ceiling)")
endif()
if(Kokkos_FOUND)
    message(STATUS "  dgemm-kokkos  → Kokkos TeamPolicy tiled")
endif()
if(RAJA_FOUND OR (DEFINED RAJA_INSTALL_PREFIX AND EXISTS "${RAJA_INSTALL_PREFIX}/include/RAJA/RAJA.hpp"))
    message(STATUS "  dgemm-raja    → RAJA naïve (API Limitation pattern, no shmem)")
endif()
if(DGEMM_ENABLE_SYCL)
    message(STATUS "  dgemm-sycl    → SYCL nd_range tiled")
endif()
if(DGEMM_ENABLE_JULIA)
    message(STATUS "  dgemm-julia   → Julia script wrapper (--mode naive|cublas|rocblas)")
endif()
if(DGEMM_ENABLE_NUMBA)
    message(STATUS "  dgemm-numba   → Numba script wrapper (tiled)")
endif()
