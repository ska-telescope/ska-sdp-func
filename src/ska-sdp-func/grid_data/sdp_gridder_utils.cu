/* See the LICENSE file at the top-level directory of this distribution. */

#include <thrust/complex.h>

#include "ska-sdp-func/fourier_transforms/private_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_cuda_atomics.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using thrust::complex;


template<typename OUT_TYPE, typename IN1_TYPE, typename IN2_TYPE>
__global__ void sdp_gridder_accum_scale_array(
        sdp_MemViewGpu<OUT_TYPE, 2> out,
        const sdp_MemViewGpu<const IN1_TYPE, 2> in1,
        const sdp_MemViewGpu<const IN2_TYPE, 2> in2,
        const int exponent,
        const int use_in2
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= out.shape[0] || j >= out.shape[1]) return;
    if (use_in2)
    {
        if (exponent == 1)
        {
            out(i, j) += (IN2_TYPE) in1(i, j) * in2(i, j);
        }
        else
        {
            out(i, j) += (IN2_TYPE) in1(i, j) * pow(in2(i, j), exponent);
        }
    }
    else
    {
        out(i, j) += in1(i, j);
    }
}


template<typename T>
__global__ void sdp_gridder_grid_correct_pswf(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        const sdp_MemViewGpu<const double, 1> pswf,
        const double* const __restrict__ pswf_n_coeff,
        double pswf_n_c,
        sdp_MemViewGpu<T, 2> facet,
        int facet_offset_l,
        int facet_offset_m
)
{
    const int64_t il = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t im = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t num_l = facet.shape[0];
    const int64_t num_m = facet.shape[1];
    if (il >= num_l || im >= num_m) return;
    const int pl = il - num_l / 2 + facet_offset_l;
    const int pm = im - num_m / 2 + facet_offset_m;
    const double l_ = pl * theta / image_size;
    const double m_ = pm * theta / image_size;
    const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
    const double pswf_l = pswf(pl + image_size / 2);
    const double pswf_m = pswf(pm + image_size / 2);
    const double pswf_n_x = fabs(n_ * 2.0 * w_step);
    const double pswf_n = (pswf_n_x < 1.0) ?
                sdp_pswf_aswfa(0, 0, pswf_n_c, pswf_n_coeff, pswf_n_x) : 1.0;
    const double scale = 1.0 / (pswf_l * pswf_m * pswf_n);
    facet(il, im) *= (T) scale;
}


template<typename T>
__global__ void sdp_gridder_grid_correct_w_stack(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_MemViewGpu<T, 2> facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        int inverse
)
{
    const int64_t il = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t im = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t num_l = facet.shape[0];
    const int64_t num_m = facet.shape[1];
    if (il >= num_l || im >= num_m) return;
    const int pl = il - num_l / 2 + facet_offset_l;
    const int pm = im - num_m / 2 + facet_offset_m;
    const double l_ = pl * theta / image_size;
    const double m_ = pm * theta / image_size;
    const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
    const double phase = 2.0 * M_PI * w_step * n_;
    complex<double> w = complex<double>(cos(phase), sin(phase));
    w = pow(w, w_offset);
    w = !inverse ? 1.0 / w : w;
    facet(il, im) *= (T) w;
}


template<typename OUT_TYPE, typename IN1_TYPE, typename IN2_TYPE>
__global__ void sdp_gridder_scale_inv_array(
        sdp_MemViewGpu<OUT_TYPE, 2> out,
        const sdp_MemViewGpu<const IN1_TYPE, 2> in1,
        const sdp_MemViewGpu<const IN2_TYPE, 2> in2,
        const int exponent
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= out.shape[0] || j >= out.shape[1]) return;
    if (exponent == 1)
    {
        out(i, j) = (IN2_TYPE) in1(i, j) / in2(i, j);
    }
    else
    {
        out(i, j) = (IN2_TYPE) in1(i, j) / pow(in2(i, j), exponent);
    }
}


template<typename GRID_TYPE, typename SUBGRID_TYPE, typename FACTOR_TYPE>
__global__ void sdp_gridder_subgrid_add(
        sdp_MemViewGpu<GRID_TYPE, 2> grid,
        int offset_u,
        int offset_v,
        sdp_MemViewGpu<const SUBGRID_TYPE, 2> subgrid,
        FACTOR_TYPE factor
)
{
    const int64_t sub_size_u = subgrid.shape[0], sub_size_v = subgrid.shape[1];
    const int64_t grid_size_u = grid.shape[0], grid_size_v = grid.shape[1];
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= sub_size_u || j >= sub_size_v) return;
    int64_t i1 = i + grid_size_u / 2 - sub_size_u / 2 - offset_u;
    int64_t j1 = j + grid_size_v / 2 - sub_size_v / 2 - offset_v;
    while (i1 < 0)
    {
        i1 += grid_size_u;
    }
    while (i1 >= grid_size_u)
    {
        i1 -= grid_size_u;
    }
    while (j1 < 0)
    {
        j1 += grid_size_v;
    }
    while (j1 >= grid_size_v)
    {
        j1 -= grid_size_v;
    }
    grid(i1, j1) += subgrid(i, j) * factor;
}


template<typename GRID_TYPE, typename SUBGRID_TYPE>
__global__ void sdp_gridder_subgrid_cut_out(
        sdp_MemViewGpu<const GRID_TYPE, 2> grid,
        int offset_u,
        int offset_v,
        sdp_MemViewGpu<SUBGRID_TYPE, 2> subgrid
)
{
    const int64_t sub_size_u = subgrid.shape[0], sub_size_v = subgrid.shape[1];
    const int64_t grid_size_u = grid.shape[0], grid_size_v = grid.shape[1];
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= sub_size_u || j >= sub_size_v) return;
    int64_t i1 = i + grid_size_u / 2 - sub_size_u / 2 - offset_u;
    int64_t j1 = j + grid_size_v / 2 - sub_size_v / 2 - offset_v;
    while (i1 < 0)
    {
        i1 += grid_size_u;
    }
    while (i1 >= grid_size_u)
    {
        i1 -= grid_size_u;
    }
    while (j1 < 0)
    {
        j1 += grid_size_v;
    }
    while (j1 >= grid_size_v)
    {
        j1 -= grid_size_v;
    }
    subgrid(i, j) = grid(i1, j1);
}


template<typename T, int BLOCK_SIZE>
__device__ void warp_reduce(volatile T* smem, int thread_id)
{
    if (BLOCK_SIZE >= 64) smem[thread_id] += smem[thread_id + 32];
    if (BLOCK_SIZE >= 32) smem[thread_id] += smem[thread_id + 16];
    if (BLOCK_SIZE >= 16) smem[thread_id] += smem[thread_id + 8];
    if (BLOCK_SIZE >= 8) smem[thread_id] += smem[thread_id + 4];
    if (BLOCK_SIZE >= 4) smem[thread_id] += smem[thread_id + 2];
    if (BLOCK_SIZE >= 2) smem[thread_id] += smem[thread_id + 1];
}


template<typename T, int BLOCK_SIZE>
__global__ void sdp_gridder_sum_diff(
        sdp_MemViewGpu<const T, 1> a,
        sdp_MemViewGpu<const T, 1> b,
        T* result
)
{
    extern __shared__ T smem[];
    const int64_t n = MIN(a.shape[0], b.shape[0]);
    const int thread_id = threadIdx.x;
    const int64_t grid_size = BLOCK_SIZE * 2 * gridDim.x;
    int64_t i = blockIdx.x * (BLOCK_SIZE * 2) + thread_id;
    smem[thread_id] = (T) 0;
    while (i < n)
    {
        smem[thread_id] += a(i) - b(i);
        if (i + BLOCK_SIZE < n)
        {
            smem[thread_id] += a(i + BLOCK_SIZE) - b(i + BLOCK_SIZE);
        }
        i += grid_size;
    }
    __syncthreads();
    if (BLOCK_SIZE >= 512)
    {
        if (thread_id < 256) smem[thread_id] += smem[thread_id + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256)
    {
        if (thread_id < 128) smem[thread_id] += smem[thread_id + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128)
    {
        if (thread_id < 64) smem[thread_id] += smem[thread_id + 64];
        __syncthreads();
    }
    if (thread_id < warpSize) warp_reduce<T, BLOCK_SIZE>(smem, thread_id);
    if (thread_id == 0) atomicAdd(result, smem[0]);
}


template<typename UVW_TYPE>
__global__ void sdp_gridder_uvw_bounds_all(
        const sdp_MemViewGpu<const UVW_TYPE, 2> uvws,
        const double freq0_hz,
        const double dfreq_hz,
        const sdp_MemViewGpu<const int, 1> start_chs,
        const sdp_MemViewGpu<const int, 1> end_chs,
        double* uvw_min,
        double* uvw_max
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t num_uvw = uvws.shape[0];
    if (i >= num_uvw)
        return;
    const int start_ch = start_chs(i), end_ch = end_chs(i);
    if (start_ch >= end_ch)
        return;
    const double uvw[] = {uvws(i, 0), uvws(i, 1), uvws(i, 2)};
    #pragma unroll
    for (int j = 0; j < 3; ++j)
    {
        const double u0 = freq0_hz * uvw[j] / C_0;
        const double du = dfreq_hz * uvw[j] / C_0;
        if (uvw[j] >= 0)
        {
            (void)sdp_atomic_min(&uvw_min[j], u0 + start_ch * du);
            (void)sdp_atomic_max(&uvw_max[j], u0 + (end_ch - 1) * du);
        }
        else
        {
            (void)sdp_atomic_max(&uvw_max[j], u0 + start_ch * du);
            (void)sdp_atomic_min(&uvw_min[j], u0 + (end_ch - 1) * du);
        }
    }
}


// *INDENT-OFF*
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<double>, complex<double>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<float>, complex<float>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<double>, complex<float>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<float>, complex<double>, complex<double> >)

SDP_CUDA_KERNEL(sdp_gridder_grid_correct_pswf<double>)
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_pswf<float>)
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_pswf<complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_pswf<complex<float> >)

SDP_CUDA_KERNEL(sdp_gridder_grid_correct_w_stack<complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_w_stack<complex<float> >)

SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<double>, double, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<float>, float, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<double>, complex<double>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<float>, complex<float>, complex<double> >)

SDP_CUDA_KERNEL(sdp_gridder_subgrid_add<complex<double>, complex<double>, double>)
SDP_CUDA_KERNEL(sdp_gridder_subgrid_add<complex<float>, complex<float>, double>)

SDP_CUDA_KERNEL(sdp_gridder_subgrid_cut_out<complex<double>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_subgrid_cut_out<complex<float>, complex<float> >)

SDP_CUDA_KERNEL(sdp_gridder_sum_diff<int, 512>)

SDP_CUDA_KERNEL(sdp_gridder_uvw_bounds_all<double>)
SDP_CUDA_KERNEL(sdp_gridder_uvw_bounds_all<float>)
// *INDENT-ON*
