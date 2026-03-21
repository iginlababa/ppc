// gpu_compat.h — HIP/CUDA portability shim for E6 BFS.
// Maps gpu* names → hip* (if __HIP_PLATFORM_AMD__) or cuda* otherwise.
// Identical in structure to kernels/sptrsv/gpu_compat.h; duplicated per-kernel
// to keep each kernel directory self-contained.

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_runtime.h>
#  define gpuMalloc                hipMalloc
#  define gpuFree                  hipFree
#  define gpuMemcpy                hipMemcpy
#  define gpuMemset                hipMemset
#  define gpuDeviceSynchronize     hipDeviceSynchronize
#  define gpuMemcpyHostToDevice    hipMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost    hipMemcpyDeviceToHost
#  define gpuMemcpyDeviceToDevice  hipMemcpyDeviceToDevice
#  define gpuSuccess               hipSuccess
#  define gpuError_t               hipError_t
#  define gpuGetErrorString        hipGetErrorString
#else
#  include <cuda_runtime.h>
#  define gpuMalloc                cudaMalloc
#  define gpuFree                  cudaFree
#  define gpuMemcpy                cudaMemcpy
#  define gpuMemset                cudaMemset
#  define gpuDeviceSynchronize     cudaDeviceSynchronize
#  define gpuMemcpyHostToDevice    cudaMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost    cudaMemcpyDeviceToHost
#  define gpuMemcpyDeviceToDevice  cudaMemcpyDeviceToDevice
#  define gpuSuccess               cudaSuccess
#  define gpuError_t               cudaError_t
#  define gpuGetErrorString        cudaGetErrorString
#endif
