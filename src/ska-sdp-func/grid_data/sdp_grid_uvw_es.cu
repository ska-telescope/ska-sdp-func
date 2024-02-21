/* See the LICENSE file at the top-level directory of this distribution. */

/* CUDA kernels are based on those from the NZAPP gridder. */

#include <cmath>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"

#define KERNEL_SUPPORT_BOUND 16

template<typename FP>
__device__ __forceinline__ void my_atomic_add(FP* addr, FP value);


template<>
__device__ __forceinline__ void my_atomic_add(float* addr, float value)
{
    atomicAdd(addr, value);
}


template<>
__device__ __forceinline__ void my_atomic_add(double* addr, double value)
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


template<typename VFP>
__device__ __forceinline__ VFP exp_semicircle(const VFP beta, const VFP x)
{
    const VFP xx = x * x;
    return (xx > VFP(1)) ? VFP(0) : exp(beta * (sqrt(VFP(1) - xx) - VFP(1)));
}


template<typename VFP, typename VFP2, typename FP, typename FP2, typename FP3>
__global__ void sdp_grid_uvw_es_cuda_3d
(
        const int num_vis_rows,
        const int num_vis_chan,
        const VFP2* const __restrict__ visibilities, // Input visibilities
        const VFP* const __restrict__ vis_weights, // Input visibility weights
        const FP3* const __restrict__ uvw_coords, // (u, v, w) coordinates
        const FP* const __restrict__ freq_hz, // Frequency per channel
        const int support, // full support for gridding kernel
        const VFP beta, // beta value used in exponential of semicircle kernel
        const FP uv_scale, // factor to convert uv coords to grid coordinates (full_grid_size * cell_size)
        const FP w_scale, // factor to convert w coord to signed w grid index
        const FP min_plane_w, // minimum w coordinate of w plane
        const int sub_grid_u_min, // zero-based sub-grid start offset
        const int sub_grid_v_min, // zero-based sub-grid start offset
        const int sub_grid_w, // Index of w grid
        const int full_grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int sub_grid_size,
        VFP2* __restrict__ sub_grid, // Output grid
        int* vis_count // If not NULL, maintain count of visibilities gridded
)
{
    const int i_chan = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_row  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;
    if (i_chan >= num_vis_chan || i_row >= num_vis_rows)
        return;

    // Calculate bounds of where gridding kernel will be applied for this visibility
    const int origin_offset_uv = full_grid_size / 2; // offset of origin along u or v axes
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const FP inv_half_support = (VFP)1.0 / (VFP)half_support;
    const int grid_min_uv = -full_grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (full_grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP flip = (uvw_coords[i_row].z < 0.0) ? -1.0 : 1.0;
    const FP inv_wavelength = flip * freq_hz[i_chan] / (FP)299792458.0;
    const FP pos_u = uvw_coords[i_row].x * inv_wavelength * uv_scale;
    const FP pos_v = uvw_coords[i_row].y * inv_wavelength * uv_scale;
    const FP pos_w = (
        uvw_coords[i_row].z * inv_wavelength - min_plane_w) * w_scale;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    const int grid_w_min = max((int)ceil(pos_w - half_support), sub_grid_w);
    const int grid_w_max = min((int)floor(pos_w + half_support), sub_grid_w);
    if (grid_w_min > grid_w_max ||
            grid_u_min > grid_u_max ||
            grid_v_min > grid_v_max)
    {
        // This visibility does not intersect the current grid.
        return;
    }

    // Calculate and check sub-grid bounds.
    const int sub_grid_u_max = sub_grid_u_min + sub_grid_size - 1;
    const int sub_grid_v_max = sub_grid_v_min + sub_grid_size - 1;
    if (grid_u_min + origin_offset_uv > sub_grid_u_max ||
            grid_u_max + origin_offset_uv < sub_grid_u_min ||
            grid_v_min + origin_offset_uv > sub_grid_v_max ||
            grid_v_max + origin_offset_uv < sub_grid_v_min)
    {
        // This visibility does not intersect the current sub-grid.
        return;
    }

    // Get the weighted visibility.
    VFP2 vis_weighted;
    vis_weighted.x = visibilities[i_vis].x * vis_weights[i_vis];
    vis_weighted.y = visibilities[i_vis].y * vis_weights[i_vis];
    vis_weighted.y *= flip; // complex conjugate for negative w coords

    // There is only one w-grid active at any time.
    const VFP kernel_w = exp_semicircle(beta,
            (VFP)(sub_grid_w - pos_w) * inv_half_support
    );

    // Cache the kernel in v and w directions.
    VFP kernel_vw[KERNEL_SUPPORT_BOUND];
    for (int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
    {
        kernel_vw[grid_v - grid_v_min] = kernel_w * exp_semicircle(beta,
                (VFP)(grid_v - pos_v) * inv_half_support
        );
    }

    // Swapped u and v for consistency with original nifty gridder.
    for (int grid_u = grid_u_min; grid_u <= grid_u_max; grid_u++)
    {
        const VFP kernel_u = exp_semicircle(beta,
                (VFP)(grid_u - pos_u) * inv_half_support
        );
        for (int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
        {
            // Get the convolution kernel value.
            VFP kernel_value = kernel_u * kernel_vw[grid_v - grid_v_min];
#if 0
            // We probably don't want this here.
            // It would be better to put it in the FFT shift instead.
            const bool odd_grid_coordinate = ((grid_u + grid_v) & 1) != 0;
            kernel_value = odd_grid_coordinate ? -kernel_value : kernel_value;
#endif
            // Update the sub-grid.
            const int sub_grid_u = grid_u + origin_offset_uv - sub_grid_u_min;
            const int sub_grid_v = grid_v + origin_offset_uv - sub_grid_v_min;
            if (sub_grid_u >= 0 && sub_grid_u < sub_grid_size &&
                    sub_grid_v >= 0 && sub_grid_v < sub_grid_size)
            {
                const size_t i_grid =
                        size_t(sub_grid_u) * sub_grid_size + size_t(sub_grid_v);
                my_atomic_add<VFP>(
                        &sub_grid[i_grid].x, vis_weighted.x * kernel_value
                );
                my_atomic_add<VFP>(
                        &sub_grid[i_grid].y, vis_weighted.y * kernel_value
                );
            }
        }
    }
    if (vis_count) atomicAdd(vis_count, 1);
}


// *INDENT-OFF*
// register kernels
SDP_CUDA_KERNEL(sdp_grid_uvw_es_cuda_3d<double, double2, double, double2, double3>)
SDP_CUDA_KERNEL(sdp_grid_uvw_es_cuda_3d<float,  float2,  float,  float2,  float3>)
// *INDENT-ON*
