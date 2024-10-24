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
        const int exponent
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= out.shape[0] || j >= out.shape[1]) return;
    if (exponent == 0)
    {
        out(i, j) += in1(i, j);
    }
    else if (exponent == 1)
    {
        out(i, j) += (IN2_TYPE) in1(i, j) * in2(i, j);
    }
    else
    {
        out(i, j) += (IN2_TYPE) in1(i, j) * pow(in2(i, j), exponent);
    }
}


template<typename OUT_TYPE, typename IN_TYPE>
__global__ void sdp_gridder_accum_complex_real_array(
        sdp_MemViewGpu<OUT_TYPE, 2> out,
        const sdp_MemViewGpu<const IN_TYPE, 2> in
)
{
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= out.shape[0] || j >= out.shape[1]) return;
    out(i, j) += in(i, j).real();
}


template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
__global__ void sdp_gridder_idft(
        const sdp_MemViewGpu<const UVW_TYPE, 2> uvw,
        const sdp_MemViewGpu<const complex<VIS_TYPE>, 2> vis,
        const sdp_MemViewGpu<const int, 1> start_chs,
        const sdp_MemViewGpu<const int, 1> end_chs,
        const sdp_MemViewGpu<const DIR_TYPE, 2> lmn,
        const sdp_MemViewGpu<const double, 1> image_taper_1d,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double theta,
        double w_step,
        double freq0_hz,
        double dfreq_hz,
        sdp_MemViewGpu<FLUX_TYPE, 2> image,
        int use_start_end_chs,
        int use_taper
)
{
    const int64_t il = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t im = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t image_size = image.shape[0];
    if (il >= image_size || im >= image_size) return;
    const int64_t num_uvw = uvw.shape[0];
    const int num_chan = (int) vis.shape[1];

    // Scale subgrid offset values.
    double du = 0, dv = 0, dw = 0;
    if (theta > 0)
    {
        du = (double) subgrid_offset_u / theta;
        dv = (double) subgrid_offset_v / theta;
        dw = (double) subgrid_offset_w * w_step;
    }

    FLUX_TYPE local_pix = 0;
    const int64_t s = il * image_size + im; // Linearised pixel index.

    // Loop over uvw values.
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        // Skip if there's no visibility to grid.
        if (use_start_end_chs && start_chs(i) >= end_chs(i))
        {
            continue;
        }

        // Loop over channels.
        for (int c = 0; c < num_chan; ++c)
        {
            const double inv_wave = (freq0_hz + dfreq_hz * c) / C_0;

            // Scale and shift uvws.
            const double u = uvw(i, 0) * inv_wave - du;
            const double v = uvw(i, 1) * inv_wave - dv;
            const double w = uvw(i, 2) * inv_wave - dw;

            const double phase = 2.0 * M_PI *
                    (lmn(s, 0) * u + lmn(s, 1) * v + lmn(s, 2) * w);
            const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
            local_pix += vis(i, c) * phasor;
        }
    }

    // Store local pixel value, appropriately tapered.
    // We can't taper afterwards, as the input image may be nonzero.
    double taper_val = use_taper ? image_taper_1d(il) * image_taper_1d(im) : 1.;
    image(il, im) += local_pix * (FLUX_TYPE) taper_val;
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


template<typename T>
__global__ void sdp_gridder_shift_subgrids(
        sdp_MemViewGpu<T, 3> subgrids
)
{
    const int w_support = (int) subgrids.shape[0];
    const int subgrid_size = (int) subgrids.shape[1];
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= subgrid_size || j >= subgrid_size) return;
    for (int k = 0; k < w_support - 1; ++k)
    {
        subgrids(k, j, i) = subgrids(k + 1, j, i);
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
    if (i1 < 0) i1 += grid_size_u;
    if (i1 >= grid_size_u) i1 -= grid_size_u;
    if (j1 < 0) j1 += grid_size_v;
    if (j1 >= grid_size_v) j1 -= grid_size_v;
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
    int64_t i1 = i + grid_size_u / 2 - sub_size_u / 2 + offset_u;
    int64_t j1 = j + grid_size_v / 2 - sub_size_v / 2 + offset_v;
    if (i1 < 0) i1 += grid_size_u;
    if (i1 >= grid_size_u) i1 -= grid_size_u;
    if (j1 < 0) j1 += grid_size_v;
    if (j1 >= grid_size_v) j1 -= grid_size_v;
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

SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<double>, double, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<float>, float, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<double>, float, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_scale_array<complex<float>, double, complex<double> >)

SDP_CUDA_KERNEL(sdp_gridder_accum_complex_real_array<double, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_complex_real_array<float, complex<float> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_complex_real_array<double, complex<float> >)
SDP_CUDA_KERNEL(sdp_gridder_accum_complex_real_array<float, complex<double> >)

SDP_CUDA_KERNEL(sdp_gridder_idft<double, complex<double>, double, double>)
SDP_CUDA_KERNEL(sdp_gridder_idft<float, complex<float>, float, float>)

SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<double>, double, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<float>, float, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<double>, complex<double>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_scale_inv_array<complex<float>, complex<float>, complex<double> >)

SDP_CUDA_KERNEL(sdp_gridder_shift_subgrids<complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_shift_subgrids<complex<float> >)

SDP_CUDA_KERNEL(sdp_gridder_subgrid_add<complex<double>, complex<double>, double>)
SDP_CUDA_KERNEL(sdp_gridder_subgrid_add<complex<float>, complex<float>, double>)

SDP_CUDA_KERNEL(sdp_gridder_subgrid_cut_out<complex<double>, complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_subgrid_cut_out<complex<float>, complex<float> >)

SDP_CUDA_KERNEL(sdp_gridder_sum_diff<int, 512>)

SDP_CUDA_KERNEL(sdp_gridder_uvw_bounds_all<double>)
SDP_CUDA_KERNEL(sdp_gridder_uvw_bounds_all<float>)
// *INDENT-ON*
