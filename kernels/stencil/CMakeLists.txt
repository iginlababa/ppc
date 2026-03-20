cmake_minimum_required(VERSION 3.20)
project(e3_stencil LANGUAGES CXX)

# ── E3 DESIGN DECISIONS ───────────────────────────────────────────────────────
# [D4] Block=(32,4,2)=256 threads; baked into stencil_common.h — no CMake option.
# [D7] Adaptive warmup logic is inside each kernel binary.
# CUDA built when nvcc found AND STENCIL_ENABLE_HIP=OFF.
# HIP: conditional on -DSTENCIL_ENABLE_HIP=ON; disables CUDA language entirely.
# SYCL: conditional on -DSTENCIL_ENABLE_SYCL=ON; detects acpp vs icpx/clang++.
# Kokkos/RAJA: compiled as CUDA TU on NVIDIA; plain CXX with hipcc on AMD.
# ─────────────────────────────────────────────────────────────────────────────

option(STENCIL_ENABLE_HIP   "Build HIP stencil target (AMD/ROCm)"  OFF)
option(STENCIL_ENABLE_SYCL  "Build SYCL stencil target"            OFF)
option(STENCIL_ENABLE_JULIA "Build Julia stencil wrapper"           ON)
option(STENCIL_ENABLE_NUMBA "Build Numba stencil wrapper"           ON)

# ── CUDA (skipped when HIP requested or nvcc not found) ───────────────────────
if(NOT STENCIL_ENABLE_HIP)
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
        set(_STENCIL_HAVE_CUDA TRUE)
    else()
        message(STATUS "  Stencil: nvcc not found — CUDA targets skipped")
        set(_STENCIL_HAVE_CUDA FALSE)
    endif()
else()
    set(_STENCIL_HAVE_CUDA FALSE)
endif()

set(STENCIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ── stencil-cuda (native baseline, NVIDIA) ───────────────────────────────────
if(_STENCIL_HAVE_CUDA)
    add_executable(stencil-cuda kernel_stencil_cuda.cu)
    target_include_directories(stencil-cuda PRIVATE ${STENCIL_INCLUDE_DIR})
    target_link_libraries(stencil-cuda PRIVATE CUDA::cudart)
    target_compile_options(stencil-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -Xcompiler=-Wall,-Wextra>)
    set_target_properties(stencil-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ── stencil-hip (native baseline, AMD) ───────────────────────────────────────
if(STENCIL_ENABLE_HIP)
    find_package(hip REQUIRED)
    add_executable(stencil-hip kernel_stencil_hip.cpp)
    target_include_directories(stencil-hip PRIVATE ${STENCIL_INCLUDE_DIR})
    target_link_libraries(stencil-hip PRIVATE hip::device)
    target_compile_options(stencil-hip PRIVATE -O3 -ffast-math -Wall -Wextra)
endif()

# ── stencil-kokkos ────────────────────────────────────────────────────────────
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    get_target_property(_kokkos_iface Kokkos::kokkos INTERFACE_LINK_LIBRARIES)
    if(_STENCIL_HAVE_CUDA)
        if("cuda" IN_LIST _kokkos_iface OR "CUDA::cudart" IN_LIST _kokkos_iface)
            set_source_files_properties(kernel_stencil_kokkos.cpp PROPERTIES LANGUAGE CUDA)
            message(STATUS "  Stencil: Kokkos CUDA backend — compiling as CUDA TU")
        endif()
    elseif(STENCIL_ENABLE_HIP)
        message(STATUS "  Stencil: Kokkos HIP backend — compiling as CXX")
    endif()
    add_executable(stencil-kokkos kernel_stencil_kokkos.cpp)
    target_include_directories(stencil-kokkos PRIVATE ${STENCIL_INCLUDE_DIR})
    if(_STENCIL_HAVE_CUDA)
        target_link_libraries(stencil-kokkos PRIVATE Kokkos::kokkos CUDA::cudart)
    else()
        target_link_libraries(stencil-kokkos PRIVATE Kokkos::kokkos)
    endif()
    target_compile_options(stencil-kokkos PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -Xcompiler=-Wall,-Wextra>)
else()
    message(STATUS "  Stencil: Kokkos not found — skipping stencil-kokkos")
endif()

# ── stencil-raja ──────────────────────────────────────────────────────────────
find_package(RAJA QUIET)
if(RAJA_FOUND)
    get_target_property(_raja_iface RAJA INTERFACE_LINK_LIBRARIES)
    if(_STENCIL_HAVE_CUDA)
        if("cuda" IN_LIST _raja_iface OR "CUDA::cudart" IN_LIST _raja_iface)
            set_source_files_properties(kernel_stencil_raja.cpp PROPERTIES LANGUAGE CUDA)
            message(STATUS "  Stencil: RAJA CUDA backend — compiling as CUDA TU")
        endif()
    elseif(STENCIL_ENABLE_HIP)
        message(STATUS "  Stencil: RAJA HIP backend — compiling as CXX")
    endif()
    add_executable(stencil-raja kernel_stencil_raja.cpp)
    target_include_directories(stencil-raja PRIVATE ${STENCIL_INCLUDE_DIR})
    if(_STENCIL_HAVE_CUDA)
        target_link_libraries(stencil-raja PRIVATE RAJA CUDA::cudart)
    else()
        target_link_libraries(stencil-raja PRIVATE RAJA)
    endif()
    target_compile_options(stencil-raja PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -allow-unsupported-compiler
            -Xcompiler=-Wall,-Wextra>)
elseif(NOT CMAKE_DISABLE_FIND_PACKAGE_RAJA)
    # Fallback: direct path compilation (bypasses broken camp/BLT cmake configs)
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
        message(STATUS "  Stencil: RAJA cmake config unavailable — using direct path ${RAJA_INSTALL_PREFIX}")
        if(_STENCIL_HAVE_CUDA)
            set_source_files_properties(kernel_stencil_raja.cpp PROPERTIES LANGUAGE CUDA)
        endif()
        add_executable(stencil-raja kernel_stencil_raja.cpp)
        target_include_directories(stencil-raja PRIVATE
            ${STENCIL_INCLUDE_DIR}
            ${RAJA_INSTALL_PREFIX}/include)
        if(_STENCIL_HAVE_CUDA)
            target_link_libraries(stencil-raja PRIVATE
                ${RAJA_INSTALL_PREFIX}/lib/libRAJA.a
                ${RAJA_INSTALL_PREFIX}/lib/libcamp.a
                CUDA::cudart)
            target_compile_options(stencil-raja PRIVATE
                -O3 --use_fast_math --generate-line-info
                --expt-extended-lambda --expt-relaxed-constexpr
                -allow-unsupported-compiler
                -Xcompiler=-Wall,-Wextra)
        else()
            target_link_libraries(stencil-raja PRIVATE
                ${RAJA_INSTALL_PREFIX}/lib/libRAJA.a
                ${RAJA_INSTALL_PREFIX}/lib/libcamp.a)
            target_compile_options(stencil-raja PRIVATE -O3 -Wall -Wextra)
        endif()
    else()
        message(STATUS "  Stencil: RAJA not found — skipping stencil-raja")
    endif()
endif()

# ── stencil-sycl ──────────────────────────────────────────────────────────────
if(STENCIL_ENABLE_SYCL)
    # Detect acpp (AdaptiveCpp) — it rejects -fsycl (Intel DPC++ flag)
    get_filename_component(_sycl_compiler_name "${CMAKE_CXX_COMPILER}" NAME)
    if(_sycl_compiler_name MATCHES "acpp|syclcc")
        set(_is_acpp TRUE)
    else()
        set(_is_acpp FALSE)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
            message(WARNING "  Stencil SYCL: CMAKE_CXX_COMPILER may not support -fsycl. "
                            "Pass -DCMAKE_CXX_COMPILER=icpx or clang++.")
        endif()
    endif()
    add_executable(stencil-sycl kernel_stencil_sycl.cpp)
    target_include_directories(stencil-sycl PRIVATE ${STENCIL_INCLUDE_DIR})
    if(NOT _is_acpp)
        target_compile_options(stencil-sycl PRIVATE -fsycl -O3)
        target_link_options(stencil-sycl PRIVATE -fsycl)
    else()
        target_compile_options(stencil-sycl PRIVATE -O3)
    endif()
    if(DEFINED SYCL_FLAGS)
        separate_arguments(_sycl_flags UNIX_COMMAND "${SYCL_FLAGS}")
        target_compile_options(stencil-sycl PRIVATE ${_sycl_flags})
        target_link_options(stencil-sycl PRIVATE ${_sycl_flags})
    endif()
else()
    message(STATUS "  Stencil: SYCL disabled (pass -DSTENCIL_ENABLE_SYCL=ON to enable)")
endif()

# ── stencil-julia wrapper ─────────────────────────────────────────────────────
if(STENCIL_ENABLE_JULIA)
    set(_julia_src "${CMAKE_CURRENT_SOURCE_DIR}/kernel_stencil_julia.jl")
    set(_julia_wrapper "${CMAKE_CURRENT_BINARY_DIR}/stencil-julia")
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
    message(STATUS "  Stencil: Julia wrapper → ${_julia_wrapper}")
endif()

# ── stencil-numba wrapper ─────────────────────────────────────────────────────
if(STENCIL_ENABLE_NUMBA)
    set(_numba_src "${CMAKE_CURRENT_SOURCE_DIR}/kernel_stencil_numba.py")
    set(_numba_wrapper "${CMAKE_CURRENT_BINARY_DIR}/stencil-numba")
    file(WRITE "${_numba_wrapper}"
"#!/usr/bin/env bash
# Auto-generated by CMake — do not edit.
PYTHON_BIN=\"\${PYTHON_BIN:-python3}\"
exec \"\${PYTHON_BIN}\" \"${_numba_src}\" \"\$@\"
")
    file(CHMOD "${_numba_wrapper}" PERMISSIONS
         OWNER_READ OWNER_WRITE OWNER_EXECUTE
         GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    message(STATUS "  Stencil: Numba wrapper → ${_numba_wrapper}")
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "E3 Stencil build summary:")
if(_STENCIL_HAVE_CUDA)
    message(STATUS "  stencil-cuda    → native 7-pt Jacobi (CUDA, no shmem, coalesced)")
elseif(STENCIL_ENABLE_HIP)
    message(STATUS "  stencil-hip     → native 7-pt Jacobi (HIP, AMD ROCm)")
endif()
if(Kokkos_FOUND)
    message(STATUS "  stencil-kokkos  → MDRangePolicy<Rank<3>> tiling {2,4,32}")
endif()
if(RAJA_FOUND OR (DEFINED RAJA_INSTALL_PREFIX AND EXISTS "${RAJA_INSTALL_PREFIX}/include/RAJA/RAJA.hpp"))
    message(STATUS "  stencil-raja    → RAJA::kernel 3D (iz→block_z, iy→block_y, ix→thread_x)")
endif()
if(STENCIL_ENABLE_SYCL)
    message(STATUS "  stencil-sycl    → SYCL nd_range<3> USM (work-group 32×4×2)")
endif()
if(STENCIL_ENABLE_JULIA)
    message(STATUS "  stencil-julia   → Julia GPU backend-aware kernel wrapper")
endif()
if(STENCIL_ENABLE_NUMBA)
    message(STATUS "  stencil-numba   → Numba @cuda.jit 3D wrapper")
endif()
