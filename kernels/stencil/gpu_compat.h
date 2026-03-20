// gpu_compat.h — HIP/CUDA portability shim for E3 stencil kernels.
//
// Maps gpu* names to either HIP or CUDA API depending on the build target.
// Included by kernel_stencil_kokkos.cpp, kernel_stencil_raja.cpp, and
// kernel_stencil_hip.cpp (which defines its own HIP_CHECK but still benefits
// from the alias macros for consistency).
//
// Select backend: -D__HIP_PLATFORM_AMD__ is set automatically by hipcc.

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_runtime.h>
#  define gpuMalloc             hipMalloc
#  define gpuFree               hipFree
#  define gpuMemcpy             hipMemcpy
#  define gpuMemset             hipMemset
#  define gpuDeviceSynchronize  hipDeviceSynchronize
#  define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#  define gpuSuccess            hipSuccess
#else
#  include <cuda_runtime.h>
#  define gpuMalloc             cudaMalloc
#  define gpuFree               cudaFree
#  define gpuMemcpy             cudaMemcpy
#  define gpuMemset             cudaMemset
#  define gpuDeviceSynchronize  cudaDeviceSynchronize
#  define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#  define gpuSuccess            cudaSuccess
#endif
