cmake_minimum_required(VERSION 3.20)
project(e3_stencil LANGUAGES CXX)

# ── E3 DESIGN DECISIONS ───────────────────────────────────────────────────────
# [D4] Block=(32,4,2)=256 threads; baked into stencil_common.h — no CMake option.
# [D7] Adaptive warmup logic is inside each kernel binary.
# CUDA always built (native stencil). Kokkos/RAJA conditional on find_package.
# SYCL: conditional on -DSTENCIL_ENABLE_SYCL=ON.
# ─────────────────────────────────────────────────────────────────────────────

option(STENCIL_ENABLE_SYCL  "Build SYCL stencil target"   OFF)
option(STENCIL_ENABLE_JULIA "Build Julia stencil wrapper"  ON)
option(STENCIL_ENABLE_NUMBA "Build Numba stencil wrapper"  ON)

# ── CUDA ──────────────────────────────────────────────────────────────────────
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(_GXX g++)
    if(_GXX)
        set(CMAKE_CUDA_HOST_COMPILER "${_GXX}" CACHE STRING "")
    endif()
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(STENCIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ── stencil-cuda (native baseline) ───────────────────────────────────────────
add_executable(stencil-cuda kernel_stencil_cuda.cu)
target_include_directories(stencil-cuda PRIVATE ${STENCIL_INCLUDE_DIR})
target_link_libraries(stencil-cuda PRIVATE CUDA::cudart)
target_compile_options(stencil-cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math --generate-line-info
        --expt-extended-lambda --expt-relaxed-constexpr
        -Xcompiler=-Wall,-Wextra>)
set_target_properties(stencil-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# ── stencil-kokkos ────────────────────────────────────────────────────────────
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    get_target_property(_kokkos_iface Kokkos::kokkos INTERFACE_LINK_LIBRARIES)
    if("cuda" IN_LIST _kokkos_iface OR "CUDA::cudart" IN_LIST _kokkos_iface)
        set_source_files_properties(kernel_stencil_kokkos.cpp PROPERTIES LANGUAGE CUDA)
        message(STATUS "  Stencil: Kokkos CUDA backend — compiling as CUDA TU")
    endif()
    add_executable(stencil-kokkos kernel_stencil_kokkos.cpp)
    target_include_directories(stencil-kokkos PRIVATE ${STENCIL_INCLUDE_DIR})
    target_link_libraries(stencil-kokkos PRIVATE Kokkos::kokkos CUDA::cudart)
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
    if("cuda" IN_LIST _raja_iface OR "CUDA::cudart" IN_LIST _raja_iface)
        set_source_files_properties(kernel_stencil_raja.cpp PROPERTIES LANGUAGE CUDA)
        message(STATUS "  Stencil: RAJA CUDA backend — compiling as CUDA TU")
    endif()
    add_executable(stencil-raja kernel_stencil_raja.cpp)
    target_include_directories(stencil-raja PRIVATE ${STENCIL_INCLUDE_DIR})
    target_link_libraries(stencil-raja PRIVATE RAJA CUDA::cudart)
    target_compile_options(stencil-raja PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -allow-unsupported-compiler
            -Xcompiler=-Wall,-Wextra>)
else()
    # Fallback: direct path compilation
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
        set_source_files_properties(kernel_stencil_raja.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(stencil-raja kernel_stencil_raja.cpp)
        target_include_directories(stencil-raja PRIVATE
            ${STENCIL_INCLUDE_DIR}
            ${RAJA_INSTALL_PREFIX}/include)
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
        message(STATUS "  Stencil: RAJA not found — skipping stencil-raja")
    endif()
endif()

# ── stencil-sycl ──────────────────────────────────────────────────────────────
if(STENCIL_ENABLE_SYCL)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
        message(WARNING "  Stencil SYCL: compiler may not support -fsycl.")
    endif()
    add_executable(stencil-sycl kernel_stencil_sycl.cpp)
    target_include_directories(stencil-sycl PRIVATE ${STENCIL_INCLUDE_DIR})
    target_compile_options(stencil-sycl PRIVATE -fsycl -O3)
    target_link_options(stencil-sycl PRIVATE -fsycl)
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
exec \"\${JULIA_BIN}\" --startup-file=no --project=@. \\
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
message(STATUS "  stencil-cuda    → always (native 7-pt Jacobi, no shmem, coalesced)")
if(Kokkos_FOUND)
    message(STATUS "  stencil-kokkos  → MDRangePolicy<Rank<3>> with tiling hint {2,4,32}")
endif()
if(RAJA_FOUND OR (DEFINED RAJA_INSTALL_PREFIX AND EXISTS "${RAJA_INSTALL_PREFIX}/include/RAJA/RAJA.hpp"))
    message(STATUS "  stencil-raja    → RAJA::kernel 3D (iz→block_z, iy→block_y, ix→thread_x)")
endif()
if(STENCIL_ENABLE_SYCL)
    message(STATUS "  stencil-sycl    → SYCL nd_range 3D (pending)")
endif()
if(STENCIL_ENABLE_JULIA)
    message(STATUS "  stencil-julia   → Julia @cuda 3D kernel wrapper")
endif()
if(STENCIL_ENABLE_NUMBA)
    message(STATUS "  stencil-numba   → Numba @cuda.jit 3D wrapper")
endif()
