cmake_minimum_required(VERSION 3.20)
project(e4_spmv LANGUAGES CXX)

# ── E4 DESIGN DECISIONS ───────────────────────────────────────────────────────
# [D3] One thread per row. No warp-per-row optimization — honest baseline.
# [D7] Adaptive warmup logic is inside each kernel binary.
# CUDA always built (native spmv). Kokkos/RAJA conditional on find_package.
# SYCL: conditional on -DSPMV_ENABLE_SYCL=ON.
# ─────────────────────────────────────────────────────────────────────────────

option(SPMV_ENABLE_SYCL  "Build SYCL spmv target"    OFF)
option(SPMV_ENABLE_JULIA "Build Julia spmv wrapper"   ON)
option(SPMV_ENABLE_NUMBA "Build Numba spmv wrapper"   ON)

# ── CUDA ──────────────────────────────────────────────────────────────────────
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(_GXX g++)
    if(_GXX)
        set(CMAKE_CUDA_HOST_COMPILER "${_GXX}" CACHE STRING "")
    endif()
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(SPMV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ── spmv-cuda (native baseline) ───────────────────────────────────────────────
add_executable(spmv-cuda kernel_spmv_cuda.cu)
target_include_directories(spmv-cuda PRIVATE ${SPMV_INCLUDE_DIR})
target_link_libraries(spmv-cuda PRIVATE CUDA::cudart)
target_compile_options(spmv-cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math --generate-line-info
        --expt-extended-lambda --expt-relaxed-constexpr
        -Xcompiler=-Wall,-Wextra>)
set_target_properties(spmv-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# ── spmv-kokkos ───────────────────────────────────────────────────────────────
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    get_target_property(_kokkos_iface Kokkos::kokkos INTERFACE_LINK_LIBRARIES)
    if("cuda" IN_LIST _kokkos_iface OR "CUDA::cudart" IN_LIST _kokkos_iface)
        set_source_files_properties(kernel_spmv_kokkos.cpp PROPERTIES LANGUAGE CUDA)
        message(STATUS "  SpMV: Kokkos CUDA backend — compiling as CUDA TU")
    endif()
    add_executable(spmv-kokkos kernel_spmv_kokkos.cpp)
    target_include_directories(spmv-kokkos PRIVATE ${SPMV_INCLUDE_DIR})
    target_link_libraries(spmv-kokkos PRIVATE Kokkos::kokkos CUDA::cudart)
    target_compile_options(spmv-kokkos PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -Xcompiler=-Wall,-Wextra>)
else()
    message(STATUS "  SpMV: Kokkos not found — skipping spmv-kokkos")
endif()

# ── spmv-raja ─────────────────────────────────────────────────────────────────
find_package(RAJA QUIET)
if(RAJA_FOUND)
    get_target_property(_raja_iface RAJA INTERFACE_LINK_LIBRARIES)
    if("cuda" IN_LIST _raja_iface OR "CUDA::cudart" IN_LIST _raja_iface)
        set_source_files_properties(kernel_spmv_raja.cpp PROPERTIES LANGUAGE CUDA)
        message(STATUS "  SpMV: RAJA CUDA backend — compiling as CUDA TU")
    endif()
    add_executable(spmv-raja kernel_spmv_raja.cpp)
    target_include_directories(spmv-raja PRIVATE ${SPMV_INCLUDE_DIR})
    target_link_libraries(spmv-raja PRIVATE RAJA CUDA::cudart)
    target_compile_options(spmv-raja PRIVATE
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
        message(STATUS "  SpMV: RAJA cmake config unavailable — using direct path ${RAJA_INSTALL_PREFIX}")
        set_source_files_properties(kernel_spmv_raja.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(spmv-raja kernel_spmv_raja.cpp)
        target_include_directories(spmv-raja PRIVATE
            ${SPMV_INCLUDE_DIR}
            ${RAJA_INSTALL_PREFIX}/include)
        target_link_libraries(spmv-raja PRIVATE
            ${RAJA_INSTALL_PREFIX}/lib/libRAJA.a
            ${RAJA_INSTALL_PREFIX}/lib/libcamp.a
            CUDA::cudart)
        target_compile_options(spmv-raja PRIVATE
            -O3 --use_fast_math --generate-line-info
            --expt-extended-lambda --expt-relaxed-constexpr
            -allow-unsupported-compiler
            -Xcompiler=-Wall,-Wextra)
    else()
        message(STATUS "  SpMV: RAJA not found — skipping spmv-raja")
    endif()
endif()

# ── spmv-sycl ─────────────────────────────────────────────────────────────────
if(SPMV_ENABLE_SYCL)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
        message(WARNING "  SpMV SYCL: compiler may not support -fsycl.")
    endif()
    add_executable(spmv-sycl kernel_spmv_sycl.cpp)
    target_include_directories(spmv-sycl PRIVATE ${SPMV_INCLUDE_DIR})
    target_compile_options(spmv-sycl PRIVATE -fsycl -O3)
    target_link_options(spmv-sycl PRIVATE -fsycl)
    if(DEFINED SYCL_FLAGS)
        separate_arguments(_sycl_flags UNIX_COMMAND "${SYCL_FLAGS}")
        target_compile_options(spmv-sycl PRIVATE ${_sycl_flags})
        target_link_options(spmv-sycl PRIVATE ${_sycl_flags})
    endif()
else()
    message(STATUS "  SpMV: SYCL disabled (pass -DSPMV_ENABLE_SYCL=ON to enable)")
endif()

# ── spmv-julia wrapper ────────────────────────────────────────────────────────
if(SPMV_ENABLE_JULIA)
    set(_julia_src  "${CMAKE_CURRENT_SOURCE_DIR}/kernel_spmv_julia.jl")
    set(_julia_wrap "${CMAKE_CURRENT_BINARY_DIR}/spmv-julia")
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
    message(STATUS "  SpMV: Julia wrapper → ${_julia_wrap}")
endif()

# ── spmv-numba wrapper ────────────────────────────────────────────────────────
if(SPMV_ENABLE_NUMBA)
    set(_numba_src  "${CMAKE_CURRENT_SOURCE_DIR}/kernel_spmv_numba.py")
    set(_numba_wrap "${CMAKE_CURRENT_BINARY_DIR}/spmv-numba")
    file(WRITE "${_numba_wrap}"
"#!/usr/bin/env bash
# Auto-generated by CMake — do not edit.
PYTHON_BIN=\"\${PYTHON_BIN:-python3}\"
exec \"\${PYTHON_BIN}\" \"${_numba_src}\" \"\$@\"
")
    file(CHMOD "${_numba_wrap}" PERMISSIONS
         OWNER_READ OWNER_WRITE OWNER_EXECUTE
         GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    message(STATUS "  SpMV: Numba wrapper → ${_numba_wrap}")
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "E4 SpMV build summary:")
message(STATUS "  spmv-cuda    → always (native CSR, one thread per row)")
if(Kokkos_FOUND)
    message(STATUS "  spmv-kokkos  → RangePolicy over rows")
endif()
if(RAJA_FOUND OR (DEFINED RAJA_INSTALL_PREFIX AND EXISTS "${RAJA_INSTALL_PREFIX}/include/RAJA/RAJA.hpp"))
    message(STATUS "  spmv-raja    → RAJA::forall<cuda_exec<256>> over rows")
endif()
if(SPMV_ENABLE_SYCL)
    message(STATUS "  spmv-sycl    → SYCL parallel_for over rows (pending)")
endif()
if(SPMV_ENABLE_JULIA)
    message(STATUS "  spmv-julia   → Julia @cuda kernel wrapper")
endif()
if(SPMV_ENABLE_NUMBA)
    message(STATUS "  spmv-numba   → Numba @cuda.jit wrapper")
endif()
