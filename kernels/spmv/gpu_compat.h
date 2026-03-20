// gpu_compat.h — HIP/CUDA portability shim for E4 SpMV kernels.
//
// Maps gpu* names to either HIP or CUDA API depending on the build target.
// Included by kernel_spmv_kokkos.cpp, kernel_spmv_raja.cpp.
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
