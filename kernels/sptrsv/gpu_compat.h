// gpu_compat.h — HIP/CUDA portability shim for E5 SpTRSV.
// Maps gpu* names → hip* (if __HIP_PLATFORM_AMD__) or cuda* otherwise.
// Mirrors kernels/spmv/gpu_compat.h; adds gpuDeviceSynchronize for SpTRSV.

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
#  define gpuError_t            hipError_t
#  define gpuGetErrorString     hipGetErrorString
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
#  define gpuError_t            cudaError_t
#  define gpuGetErrorString     cudaGetErrorString
#endif
