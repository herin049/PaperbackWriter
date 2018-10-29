#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

#ifdef __JETBRAINS_IDE__
    #include <math.h>
    #define __CUDACC__ 1
    #define __host__
    #define __device__
    #define __global__
    #define __noinline__
    #define __forceinline__
    #define __shared__
    #define __constant__
    #define __managed__
    #define __restrict__

    inline void __syncthreads() {};
    inline void __threadfence_block() {};
    inline void __threadfence() {};
    inline void __threadfence_system();
    inline int __syncthreads_count(int predicate) {return predicate};
    inline int __syncthreads_and(int predicate) {return predicate};
    inline int __syncthreads_or(int predicate) {return predicate};
    template<class T> inline T __clz(const T val) { return val; }
    template<class T> inline T __ldg(const T* address){return *address};

    typedef unsigned short uchar;
    typedef unsigned short ushort;
    typedef unsigned int uint;
    typedef unsigned long ulong;
    typedef unsigned long long ulonglong;
    typedef long long longlong;
#endif

#ifdef __CUDACC__
#  define PBW_ANNOTATE__ __host__ __device__
#else
#  define PBW_ANNOTATE__
#endif
#define PBW_KERNEL__ __global__

