cmake_minimum_required(VERSION 3.18)
project(e7_nbody LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CXX_STANDARD 17)

# ── Kokkos paths ───────────────────────────────────────────────────────────────
set(KOKKOS_INSTALL_DIR "/home/obalola/projects/kokkos-cuda-install"
    CACHE PATH "Kokkos installation prefix")
set(KOKKOS_INCLUDE_DIR "${KOKKOS_INSTALL_DIR}/include")
set(KOKKOS_LIB_DIR     "${KOKKOS_INSTALL_DIR}/lib")

# ── RAJA paths ─────────────────────────────────────────────────────────────────
set(RAJA_INSTALL_DIR "/home/obalola/projects/raja/install"
    CACHE PATH "RAJA installation prefix")
set(RAJA_INCLUDE_DIR "${RAJA_INSTALL_DIR}/include")
set(RAJA_LIB_DIR     "${RAJA_INSTALL_DIR}/lib")

# ── Native targets ─────────────────────────────────────────────────────────────
add_executable(nbody_native nbody_native.cu)
target_include_directories(nbody_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(nbody_native PROPERTIES
    CUDA_ARCHITECTURES 120
    CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(nbody_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# ── Kokkos target ──────────────────────────────────────────────────────────────
if(EXISTS "${KOKKOS_INCLUDE_DIR}/Kokkos_Core.hpp")
    # Treat .cpp as CUDA source for Kokkos device lambdas
    set_source_files_properties(nbody_kokkos.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(nbody_kokkos nbody_kokkos.cpp)
    target_include_directories(nbody_kokkos PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${KOKKOS_INCLUDE_DIR})
    set_target_properties(nbody_kokkos PROPERTIES
        CUDA_ARCHITECTURES 120
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(nbody_kokkos PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -std=c++20
            --expt-extended-lambda
            --expt-relaxed-constexpr
            -O3>)
    target_link_libraries(nbody_kokkos PRIVATE
        "${KOKKOS_LIB_DIR}/libkokkoscore.a"
        cuda dl)
    message(STATUS "E7 Kokkos target enabled")
else()
    message(STATUS "E7 Kokkos SKIPPED: ${KOKKOS_INCLUDE_DIR}/Kokkos_Core.hpp not found")
endif()

# ── RAJA target ────────────────────────────────────────────────────────────────
if(EXISTS "${RAJA_INCLUDE_DIR}/RAJA/RAJA.hpp")
    set_source_files_properties(nbody_raja.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(nbody_raja nbody_raja.cpp)
    target_include_directories(nbody_raja PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${RAJA_INCLUDE_DIR})
    set_target_properties(nbody_raja PROPERTIES
        CUDA_ARCHITECTURES 120
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(nbody_raja PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-extended-lambda
            --expt-relaxed-constexpr
            -allow-unsupported-compiler
            -O3>)
    target_link_libraries(nbody_raja PRIVATE
        "${RAJA_LIB_DIR}/libRAJA.a"
        "${RAJA_LIB_DIR}/libcamp.a"
        cuda dl)
    message(STATUS "E7 RAJA target enabled")
else()
    message(STATUS "E7 RAJA SKIPPED: ${RAJA_INCLUDE_DIR}/RAJA/RAJA.hpp not found")
endif()
