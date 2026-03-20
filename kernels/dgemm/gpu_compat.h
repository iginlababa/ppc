// gpu_compat.h — GPU API compatibility shim for CUDA/HIP portability.
//
// Maps gpu* → hip* when building with hipcc (__HIP_PLATFORM_AMD__ defined),
// cuda* when building with nvcc/clang CUDA.  Include this instead of
// cuda_runtime.h / hip_runtime.h directly in host code that only needs
// memory management (malloc/free/memcpy).

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_runtime.h>
#  define gpuMalloc              hipMalloc
#  define gpuFree                hipFree
#  define gpuMemcpy              hipMemcpy
#  define gpuMemcpyHostToDevice  hipMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost  hipMemcpyDeviceToHost
#  define gpuSuccess             hipSuccess
#else
#  include <cuda_runtime.h>
#  define gpuMalloc              cudaMalloc
#  define gpuFree                cudaFree
#  define gpuMemcpy              cudaMemcpy
#  define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
#  define gpuSuccess             cudaSuccess
#endif
