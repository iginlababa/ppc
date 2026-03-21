// gpu_compat.h — HIP/CUDA portability shim for E7 N-Body kernels.
//
// Maps gpu* names to either HIP or CUDA API depending on the build target.
// Included by nbody_kokkos.cpp, nbody_raja.cpp.
//
// Select backend: -D__HIP_PLATFORM_AMD__ is set automatically by hipcc.

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_runtime.h>
#  define gpuMalloc              hipMalloc
#  define gpuFree                hipFree
#  define gpuMemcpy              hipMemcpy
#  define gpuMemset              hipMemset
#  define gpuDeviceSynchronize   hipDeviceSynchronize
#  define gpuMemcpyHostToDevice  hipMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost  hipMemcpyDeviceToHost
#  define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#  define gpuSuccess             hipSuccess
#  define gpuError_t             hipError_t
#  define gpuGetErrorString      hipGetErrorString
#  define gpuMemGetInfo          hipMemGetInfo
#  define gpuEvent_t             hipEvent_t
#  define gpuEventCreate         hipEventCreate
#  define gpuEventRecord         hipEventRecord
#  define gpuEventSynchronize    hipEventSynchronize
#  define gpuEventElapsedTime    hipEventElapsedTime
#  define gpuEventDestroy        hipEventDestroy
#else
#  include <cuda_runtime.h>
#  define gpuMalloc              cudaMalloc
#  define gpuFree                cudaFree
#  define gpuMemcpy              cudaMemcpy
#  define gpuMemset              cudaMemset
#  define gpuDeviceSynchronize   cudaDeviceSynchronize
#  define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
#  define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
#  define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#  define gpuSuccess             cudaSuccess
#  define gpuError_t             cudaError_t
#  define gpuGetErrorString      cudaGetErrorString
#  define gpuMemGetInfo          cudaMemGetInfo
#  define gpuEvent_t             cudaEvent_t
#  define gpuEventCreate         cudaEventCreate
#  define gpuEventRecord         cudaEventRecord
#  define gpuEventSynchronize    cudaEventSynchronize
#  define gpuEventElapsedTime    cudaEventElapsedTime
#  define gpuEventDestroy        cudaEventDestroy
#endif

#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        gpuError_t _e = (call);                                                \
        if (_e != gpuSuccess) {                                                \
            std::fprintf(stderr, "GPU error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, gpuGetErrorString(_e));           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
