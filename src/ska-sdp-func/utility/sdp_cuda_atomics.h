/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_CUDA_ATOMICS_H_
#define SDP_CUDA_ATOMICS_H_

#ifndef __CUDACC__
#define __device__
#define __forceinline__
#endif


__device__ __forceinline__ void sdp_atomic_add(float* addr, float value)
{
    atomicAdd(addr, value);
}


__device__ __forceinline__ void sdp_atomic_add(double* addr, double value)
{
#if __CUDA_ARCH__ >= 600
    // Supports native double precision atomic add.
    atomicAdd(addr, value);
#else
    unsigned long long int* laddr = (unsigned long long int*)(addr);
    unsigned long long int assumed, old_ = *laddr;
    do
    {
        assumed = old_;
        old_ = atomicCAS(laddr,
                assumed,
                __double_as_longlong(value +
                __longlong_as_double(assumed)
                )
        );
    }
    while (assumed != old_);
#endif
}


__device__ __forceinline__ double sdp_atomic_min(double* address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long*) address, old,
                __double_as_longlong(val)
                )) == old)
            break;
    }
    return __longlong_as_double(ret);
}


__device__ __forceinline__ double sdp_atomic_max(double* address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long*) address, old,
                __double_as_longlong(val)
                )) == old)
            break;
    }
    return __longlong_as_double(ret);
}


#endif /* include guard */
