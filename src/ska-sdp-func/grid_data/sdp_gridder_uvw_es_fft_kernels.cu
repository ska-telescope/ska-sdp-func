/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"

#define KERNEL_SUPPORT_BOUND 16

#ifndef PI
#define PI 3.1415926535897931
#endif

#define C_0 299792458.0

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


/**********************************************************************
 * Evaluates the convolutional correction C(k) in one dimension.
 * As the exponential of semicircle gridding kernel psi(u,v,w) is separable,
 * its Fourier transform Fourier(psi)(l,m,n) is likewise separable into
 * one-dimensional components C(l)C(m)C(n).
 *
 * As psi is even and zero outside its support, the convolutional correction
 * is given by:
 *   C(k) = 2\integral_{0}^{supp/2} psi(u)cos(2\pi ku) du =
 *       supp\integral_{0}^{1}psi(supp*x)cos(\pi*k*supp*x) dx
 * by change of variables.
 *
 * This integral from 0 to 1 can be numerically approximated via a 2p-node
 * Gauss-Legendre quadrature, as recommended in equation 3.10 of
 * 'A Parallel Non-uniform Fast Fourier Transform Library
 * Based on an "Exponential of Semicircle" Kernel'
 * by Barnett, Magland, Klintenberg and only using the p positive nodes):
 *   C(k) ~ Sum_{i=1}^{p} weight_i*psi(supp*node_i)*cos(\pi*k*supp*node_i)
 * Note this convolutional correction is not normalised, but is normalised
 * during use in convolution correction by C(0) to get max of 1
 **********************************************************************/
template<typename FP>
__device__ FP conv_corr(
        const FP support,
        const FP k,
        const FP* const __restrict__ quadrature_kernel,
        const FP* const __restrict__ quadrature_nodes,
        const FP* const __restrict__ quadrature_weights
)
{
    FP correction = 0.0;
    uint32_t p = (uint32_t)(ceil(FP(1.5) * support + FP(2.0)));

    for (uint32_t i = 0; i < p; i++)
        correction += quadrature_kernel[i] *
                cos(PI * k * support * quadrature_nodes[i]) *
                quadrature_weights[i];

    return correction * support;
}


/**********************************************************************
 * Calculates the exponential of semicircle
 * Note the parameter x must be normalised to be in range [-1,1]
 * Source Paper: A parallel non-uniform fast Fourier transform library
 * based on an "exponential of semicircle" kernel
 * Address: https://arxiv.org/abs/1808.06736
 **********************************************************************/
template<typename VFP>
__device__ VFP exp_semicircle(const VFP beta, const VFP x)
{
    const VFP xx = x * x;
    return (xx > VFP(1)) ? VFP(0) : exp(beta * (sqrt(VFP(1) - xx) - VFP(1)));
}


/**********************************************************************
 * Calculates complex phase shift for applying to each
 * w layer (note: w layer = iFFT(w grid))
 * Note: l and m are expected to be in the range  [-0.5, 0.5]
 **********************************************************************/
template<typename FP, typename FP2>
__device__ FP2 phase_shift(const FP w, const FP l, const FP m, const FP signage)
{
    FP2 phasor;
    const FP sos = l * l + m * m;
    const FP nm1 = (-sos) / (sqrt(FP(1.0) - sos) + FP(1.0));
    const FP x = FP(2.0) * FP(PI) * w * nm1;
    const FP xn = FP(1.0) / (nm1 + FP(1.0));
    // signage = -1.0 if gridding, 1.0 if degridding
    sincos(signage * x, &phasor.y, &phasor.x);
    phasor.x *= xn;
    phasor.y *= xn;
    return phasor;
}


/*
 * Runs through all the visibilities and counts how many fall into each tile.
 * Grid updates for each visibility will intersect one or more tiles.
 */
template<typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_tile_count (
        const int support,
        const int num_vis_rows,
        const int num_vis_chan,
        const FP* const __restrict__ freq_hz,
        const FP3* const __restrict__ uvw,
        const int grid_size,
        const FP grid_scale,
        const int tile_size_u,
        const int tile_size_v,
        const int num_tiles_v,
        const int top_left_u,
        const int top_left_v,
        int* num_points_in_tiles
)
{
    const int i_row  = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_chan = blockDim.y * blockIdx.y + threadIdx.y;
    if (i_chan >= num_vis_chan || i_row >= num_vis_rows)
        return;

    const int centre = grid_size / 2; // offset of origin along u or v axes
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP flip = (uvw[i_row].z < 0.0) ? -1.0 : 1.0;
    const FP inv_wavelength = flip * freq_hz[i_chan] / (FP) C_0;
    const FP pos_u = uvw[i_row].x * inv_wavelength * grid_scale;
    const FP pos_v = uvw[i_row].y * inv_wavelength * grid_scale;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    if (grid_u_min > grid_u_max || grid_v_min > grid_v_max)
    {
        // This visibility does not intersect the grid.
        return;
    }
    const int tile_u_min = (grid_u_min - top_left_u + centre) / tile_size_u;
    const int tile_u_max = (grid_u_max - top_left_u + centre) / tile_size_u;
    const int tile_v_min = (grid_v_min - top_left_v + centre) / tile_size_v;
    const int tile_v_max = (grid_v_max - top_left_v + centre) / tile_size_v;
    for (int pu = tile_u_min; pu <= tile_u_max; pu++)
    {
        for (int pv = tile_v_min; pv <= tile_v_max; pv++)
        {
            atomicAdd(&num_points_in_tiles[pv + pu * num_tiles_v], 1);
        }
    }
}

/*
 * Does a bucket sort on the input visibilities. Each tile is a bucket.
 * Note that tile_offsets gives the start of visibility data for each tile,
 * and it will be modified by this kernel.
 */
template<typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_tile_bucket_sort (
        const int support,
        const int num_vis_rows,
        const int num_vis_chan,
        const FP* const __restrict__ freq_hz,
        const FP3* const __restrict__ uvw,
        const FP2* const __restrict__ vis,
        const FP* const __restrict__ weight,
        const int grid_size,
        const FP grid_scale,
        const int tile_size_u,
        const int tile_size_v,
        const int num_tiles_v,
        const int top_left_u,
        const int top_left_v,
        int* tile_offsets,
        FP* sorted_uu,
        FP* sorted_vv,
        FP* sorted_ww,
        FP2* sorted_vis,
        int* sorted_tile
)
{
    const int i_row  = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_chan = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;
    if (i_chan >= num_vis_chan || i_row >= num_vis_rows)
        return;

    const int centre = grid_size / 2; // offset of origin along u or v axes
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP flip = (uvw[i_row].z < 0.0) ? -1.0 : 1.0;
    const FP inv_wavelength = flip * freq_hz[i_chan] / (FP) C_0;
    FP2 vis_i = vis[i_vis];
    vis_i.x *= weight[i_vis];
    vis_i.y *= (weight[i_vis] * flip); // Conjugate for negative w coords
    const FP pos_u = uvw[i_row].x * inv_wavelength * grid_scale;
    const FP pos_v = uvw[i_row].y * inv_wavelength * grid_scale;
    const FP pos_w = uvw[i_row].z * inv_wavelength;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    if (grid_u_min > grid_u_max || grid_v_min > grid_v_max)
    {
        // This visibility does not intersect the grid.
        return;
    }
    const int tile_u_min = (grid_u_min - top_left_u + centre) / tile_size_u;
    const int tile_u_max = (grid_u_max - top_left_u + centre) / tile_size_u;
    const int tile_v_min = (grid_v_min - top_left_v + centre) / tile_size_v;
    const int tile_v_max = (grid_v_max - top_left_v + centre) / tile_size_v;
    for (int pu = tile_u_min; pu <= tile_u_max; pu++)
    {
        for (int pv = tile_v_min; pv <= tile_v_max; pv++)
        {
            int off = atomicAdd(&tile_offsets[pv + pu * num_tiles_v], 1);
            sorted_uu[off] = pos_u;
            sorted_vv[off] = pos_v;
            sorted_ww[off] = pos_w;
            sorted_vis[off] = vis_i; // Store weighted visibility.
            sorted_tile[off] = pv * 32768 + pu;
        }
    }
}

#define NUM_VIS_LOCAL 32

#define WRITE_ACTIVE_TILE_TO_GRID(FP, FP2) {\
    for (int r = 0; r < REGSZ; r++) {\
        const int64_t p = int64_t(my_grid_u_start + r) * int64_t(grid_size) + my_grid_v;\
        my_atomic_add<FP>(&grid[2 * p],     my_grid[r].x);\
        my_atomic_add<FP>(&grid[2 * p + 1], my_grid[r].y);\
    }\
    for (int s = 0; s < SHMSZ; s++) {\
        const int64_t p = int64_t(my_grid_u_start + s + REGSZ) * int64_t(grid_size) + my_grid_v;\
        const FP2 z = smem[threadIdx.x + s * blockDim.x];\
        my_atomic_add<FP>(&grid[2 * p],     z.x);\
        my_atomic_add<FP>(&grid[2 * p + 1], z.y);\
    }\
    }\


// REGSZ = 8, SHMSZ = 8
template<typename FP, typename FP2, int REGSZ, int SHMSZ>
__global__ void sdp_cuda_nifty_grid_tiled_3d (
        const int support, // full support for gridding kernel
        const int num_vis_total,
        const FP* const __restrict__ sorted_uu,
        const FP* const __restrict__ sorted_vv,
        const FP* const __restrict__ sorted_ww,
        const FP2* const __restrict__ sorted_vis, // weighted in bucket sort
        const int* const __restrict__ sorted_tile,
        const int grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int grid_start_w, // Index of w grid
        const int tile_size_u,
        const int tile_size_v,
        const int top_left_u,
        const int top_left_v,
        const FP beta, // beta value used in exponential of semicircle kernel
        const FP w_scale, // factor to convert w coord to signed w grid index
        const FP min_plane_w, // w coordinate of w plane
        int* vis_counter,
        FP* grid,
        int* vis_gridded // If not NULL, maintain count of visibilities gridded
        )
{
    __shared__ FP s_u[NUM_VIS_LOCAL], s_v[NUM_VIS_LOCAL], s_w[NUM_VIS_LOCAL];
    __shared__ FP2 s_vis[NUM_VIS_LOCAL];
    __shared__ int s_tile_coords[NUM_VIS_LOCAL];
    const int centre = grid_size / 2; // offset of origin along u or v axes
    const FP half_support = (FP)support / (FP)2;
    const FP inv_half_support = (FP)1 / (FP)half_support;
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    int i_vis = 0;
    FP2 my_grid[REGSZ + 1];
    extern __shared__ __align__(64) unsigned char my_smem[];
    FP2* smem = reinterpret_cast<FP2*>(my_smem);
    int tile_u = -1, tile_v = -1, my_grid_v = 0, my_grid_u_start = 0;
    while (true)
    {
        if (threadIdx.x == 0)
        {
            i_vis = atomicAdd(&vis_counter[0], NUM_VIS_LOCAL);
        }
        i_vis = __shfl_sync(0xFFFFFFFF, i_vis, 0);
        if (i_vis >= num_vis_total)
        {
            break;
        }
        __syncthreads();
        const int i_vis_load = i_vis + threadIdx.x;
        if (threadIdx.x < NUM_VIS_LOCAL && i_vis_load < num_vis_total)
        {
            s_u[threadIdx.x] = sorted_uu[i_vis_load];
            s_v[threadIdx.x] = sorted_vv[i_vis_load];
            s_w[threadIdx.x] = (sorted_ww[i_vis_load] - min_plane_w) * w_scale;
            s_vis[threadIdx.x] = sorted_vis[i_vis_load];
            s_tile_coords[threadIdx.x] = sorted_tile[i_vis_load];
        }
        __syncthreads();
        int grid_points_hit = 0;
        for (int i_vis_local = 0; i_vis_local < NUM_VIS_LOCAL; i_vis_local++)
        {
            if ((i_vis + i_vis_local) >= num_vis_total)
            {
                continue;
            }
            const int grid_w_min = max((int)ceil(s_w[i_vis_local] - half_support), grid_start_w);
            const int grid_w_max = min((int)floor(s_w[i_vis_local] + half_support), grid_start_w);
            if (grid_w_min > grid_w_max)
            {
                continue;
            }
            const int grid_u_min = max((int)ceil(s_u[i_vis_local] - half_support), grid_min_uv) + centre;
            const int grid_u_max = min((int)floor(s_u[i_vis_local] + half_support), grid_max_uv) + centre;
            const int grid_v_min = max((int)ceil(s_v[i_vis_local] - half_support), grid_min_uv) + centre;
            const int grid_v_max = min((int)floor(s_v[i_vis_local] + half_support), grid_max_uv) + centre;
            const int tile_coords = s_tile_coords[i_vis_local];
            const int new_tile_u = tile_coords & 32767;
            const int new_tile_v = tile_coords >> 15;
            if (new_tile_u != tile_u || new_tile_v != tile_v)
            {
                if (tile_u != -1) WRITE_ACTIVE_TILE_TO_GRID(FP, FP2)
                tile_u = new_tile_u;
                tile_v = new_tile_v;
                my_grid_v = tile_v * tile_size_v + top_left_v + threadIdx.x;
                my_grid_u_start = tile_u * tile_size_u + top_left_u;
                FP2 zero;
                zero.x = (FP)0;
                zero.y = (FP)0;
                #pragma unroll
                for (int r = 0; r < REGSZ; r++)
                {
                    my_grid[r] = zero;
                }
                #pragma unroll
                for (int s = 0; s < SHMSZ; s++)
                {
                    smem[threadIdx.x + s * blockDim.x] = zero;
                }
            }
            if (my_grid_v >= grid_v_min && my_grid_v <= grid_v_max)
            {
                const FP kernel_w = exp_semicircle(beta,
                        (FP)(grid_start_w - s_w[i_vis_local]) * inv_half_support
                );
                const FP kernel_v = exp_semicircle(beta,
                        (FP)(my_grid_v - centre - s_v[i_vis_local]) * inv_half_support
                );
                const FP kernel_vw = kernel_w * kernel_v;
                const FP2 val = s_vis[i_vis_local];
                #pragma unroll
                for (int t = 0; t < (REGSZ + SHMSZ); t++)
                {
                    const int my_grid_u = my_grid_u_start + t;
                    if (my_grid_u >= grid_u_min && my_grid_u <= grid_u_max)
                    {
                        const FP kernel_u = exp_semicircle(beta,
                                (FP)(my_grid_u - centre - s_u[i_vis_local]) * inv_half_support
                        );
                        FP c = kernel_u * kernel_vw;
                        const bool is_odd = ((my_grid_u - centre + my_grid_v - centre) & 1) != 0;
                        c = is_odd ? -c : c;
                        if (t < REGSZ)
                        {
                            my_grid[t].x += (val.x * c);
                            my_grid[t].y += (val.y * c);
                        }
                        else if (SHMSZ > 0)
                        {
                            const int s = t - REGSZ;
                            FP2 z = smem[threadIdx.x + s * blockDim.x];
                            z.x += (val.x * c);
                            z.y += (val.y * c);
                            smem[threadIdx.x + s * blockDim.x] = z;
                        }
                        grid_points_hit++;
                    }
                }
            }
        }
        // if (vis_gridded) atomicAdd(vis_gridded, NUM_VIS_LOCAL);
        if (vis_gridded) atomicAdd(vis_gridded, grid_points_hit);
        __syncthreads();
    }
    if (tile_u != -1) WRITE_ACTIVE_TILE_TO_GRID(FP, FP2)
}


template<typename VFP, typename VFP2, typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_nifty_grid_3d
(
        const int num_vis_rows,
        const int num_vis_chan,
        const VFP2* const __restrict__ visibilities, // Input visibilities
        const VFP* const __restrict__ vis_weights, // Input visibility weights
        const FP3* const __restrict__ uvw_coords, // (u, v, w) coordinates
        const FP* const __restrict__ freq_hz, // Frequency per channel
        VFP2* __restrict__ w_grid_stack, // Output grid
        const int grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int grid_start_w, // Index of w grid
        const uint32_t support, // full support for gridding kernel
        const VFP beta, // beta value used in exponential of semicircle kernel
        const FP uv_scale, // factor to convert uv coords to grid coordinates (grid_size * cell_size)
        const FP w_scale, // factor to convert w coord to signed w grid index
        const FP min_plane_w, // w coordinate of w plane
        int* vis_count // If not NULL, maintain count of visibilities gridded
)
{
    const int i_chan = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_row  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;
    if (i_chan >= num_vis_chan || i_row >= num_vis_rows)
        return;

    // Calculate bounds of where gridding kernel will be applied for this visibility
    const int origin_offset_uv = (grid_size / 2); // offset of origin along u or v axes
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const FP inv_half_support = (VFP)1.0 / (VFP)half_support;
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP flip = (uvw_coords[i_row].z < 0.0) ? -1.0 : 1.0;
    const FP inv_wavelength = flip * freq_hz[i_chan] / (FP)299792458.0;
    const FP pos_u = uvw_coords[i_row].x * inv_wavelength * uv_scale;
    const FP pos_v = uvw_coords[i_row].y * inv_wavelength * uv_scale;
    const FP pos_w = (uvw_coords[i_row].z * inv_wavelength - min_plane_w) *
            w_scale;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    const int grid_w_min = max((int)ceil(pos_w - half_support), grid_start_w);
    const int grid_w_max = min((int)floor(pos_w + half_support), grid_start_w);
    if (grid_w_min > grid_w_max ||
            grid_u_min > grid_u_max ||
            grid_v_min > grid_v_max)
    {
        // This visibility does not intersect the current grid.
        return;
    }

    // Get the weighted visibility.
    VFP2 vis_weighted;
    vis_weighted.x = visibilities[i_vis].x * vis_weights[i_vis];
    vis_weighted.y = visibilities[i_vis].y * vis_weights[i_vis];
    vis_weighted.y *= flip; // complex conjugate for negative w coords

    // There is only one w-grid active at any time.
    const VFP kernel_w = exp_semicircle(beta,
            (VFP)(grid_start_w - pos_w) * inv_half_support
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
            // Update the grid, applying the separable kernel.
            VFP kernel_value = kernel_u * kernel_vw[grid_v - grid_v_min];
            const bool odd_grid_coordinate = ((grid_u + grid_v) & 1) != 0;
            kernel_value = odd_grid_coordinate ? -kernel_value : kernel_value;
            const size_t i_grid =
                    size_t(grid_u + origin_offset_uv) * grid_size +
                    size_t(grid_v + origin_offset_uv);
            my_atomic_add<VFP>(
                    &w_grid_stack[i_grid].x, vis_weighted.x * kernel_value
            );
            my_atomic_add<VFP>(
                    &w_grid_stack[i_grid].y, vis_weighted.y * kernel_value
            );
        }
    }
    if (vis_count) atomicAdd(vis_count, 1);
}


template<typename VFP, typename VFP2, typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_nifty_degrid_3d
(
        const int num_vis_rows,
        const int num_vis_chan,
        VFP2* __restrict__ visibilities, // Output visibilities
        const VFP* const __restrict__ vis_weights, // Input visibility weights
        const FP3* const __restrict__ uvw_coords, // (u, v, w) coordinates
        const FP* const __restrict__ freq_hz, // Frequency per channel
        const VFP2* const __restrict__ w_grid_stack, // Input grid
        const int grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int grid_start_w, // Index of w grid
        const uint32_t support, // full support for gridding kernel
        const VFP beta, // beta value used in exponential of semicircle kernel
        const FP uv_scale, // scaling factor for conversion of uv coords to grid coordinates (grid_size * cell_size)
        const FP w_scale, // scaling factor for converting w coord to signed w grid index
        const FP min_plane_w // w coordinate of smallest w plane
)
{
    const int i_chan = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_row  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;
    if (i_chan >= num_vis_chan || i_row >= num_vis_rows)
        return;

    // Calculate bounds of where gridding kernel will be applied for this visibility
    const int origin_offset_uv = (grid_size / 2); // offset of origin along u or v axes
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const VFP inv_half_support = (VFP)1.0 / (VFP)half_support;
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP flip = (uvw_coords[i_row].z < 0.0) ? -1.0 : 1.0;
    const FP inv_wavelength = flip * freq_hz[i_chan] / (FP)299792458.0;
    const FP pos_u = uvw_coords[i_row].x * inv_wavelength * uv_scale;
    const FP pos_v = uvw_coords[i_row].y * inv_wavelength * uv_scale;
    const FP pos_w = (uvw_coords[i_row].z * inv_wavelength - min_plane_w) *
            w_scale;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    const int grid_w_min = max((int)ceil(pos_w - half_support), grid_start_w);
    const int grid_w_max = min((int)floor(pos_w + half_support), grid_start_w);
    if (grid_w_min > grid_w_max ||
            grid_u_min > grid_u_max ||
            grid_v_min > grid_v_max)
    {
        // This visibility does not intersect the current grid.
        return;
    }

    // Zero the visibility.
    VFP2 vis_tmp;
    vis_tmp.x = vis_tmp.y = (VFP)0;

    // There is only one w-grid active at any time.
    const VFP kernel_w = exp_semicircle(beta,
            (VFP)(grid_start_w - pos_w) * inv_half_support
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
            // Read from the grid, applying the separable kernel.
            VFP kernel_value = kernel_u * kernel_vw[grid_v - grid_v_min];
            const bool odd_grid_coordinate = ((grid_u + grid_v) & 1) != 0;
            kernel_value = odd_grid_coordinate ? -kernel_value : kernel_value;
            const size_t i_grid =
                    size_t(grid_u + origin_offset_uv) * grid_size +
                    size_t(grid_v + origin_offset_uv);
            vis_tmp.x += w_grid_stack[i_grid].x * kernel_value;
            vis_tmp.y += w_grid_stack[i_grid].y * kernel_value;
        }
    }
    visibilities[i_vis].x += vis_tmp.x * vis_weights[i_vis];
    visibilities[i_vis].y += vis_tmp.y * vis_weights[i_vis] * flip;
}


template<typename VFP, typename VFP2, typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_nifty_grid_2d
(
        const int num_vis_rows,
        const int num_vis_chan,
        const VFP2* const __restrict__ visibilities, // Input visibilities
        const VFP* const __restrict__ vis_weights, // Input visibility weights
        const FP3* const __restrict__ uvw_coords, // (u, v, w) coordinates
        const FP* const __restrict__ freq_hz, // Frequency per channel
        VFP2* __restrict__ w_grid_stack, // Output grid
        const int grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int /*grid_start_w*/, // Index of w grid (always 0, for 2D)
        const uint32_t support, // full support for gridding kernel
        const VFP beta, // beta value used in exponential of semicircle kernel
        const FP uv_scale, // scaling factor for conversion of uv coords to grid coordinates (grid_size * cell_size)
        const FP w_scale, // scaling factor for converting w coord to signed w grid index
        const FP min_plane_w, // w coordinate of smallest w plane
        int* vis_count // If not NULL, maintain count of visibilities gridded
)
{
    const int i_chan = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_row  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;
    if (i_chan >= num_vis_chan || i_row >= num_vis_rows)
        return;

    // Calculate bounds of where gridding kernel will be applied for this visibility
    const int origin_offset_uv = (grid_size / 2); // offset of origin along u or v axes
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const VFP inv_half_support = (VFP)1.0 / (VFP)half_support;
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP inv_wavelength = freq_hz[i_chan] / (FP)299792458.0;
    const FP pos_u = uvw_coords[i_row].x * inv_wavelength * uv_scale;
    const FP pos_v = uvw_coords[i_row].y * inv_wavelength * uv_scale;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    if (grid_u_min > grid_u_max || grid_v_min > grid_v_max)
    {
        // This visibility does not intersect the current grid.
        return;
    }

    // Get the weighted visibility.
    VFP2 vis_weighted;
    vis_weighted.x = visibilities[i_vis].x * vis_weights[i_vis];
    vis_weighted.y = visibilities[i_vis].y * vis_weights[i_vis];

    // Cache the kernel in v direction.
    VFP kernel_v[KERNEL_SUPPORT_BOUND];
    for (int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
    {
        kernel_v[grid_v - grid_v_min] = exp_semicircle(beta,
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
            // Update the grid, applying the separable kernel.
            VFP kernel_value = kernel_u * kernel_v[grid_v - grid_v_min];
            bool odd_grid_coordinate = ((grid_u + grid_v) & 1) != 0;
            kernel_value = odd_grid_coordinate ? -kernel_value : kernel_value;
            const size_t i_grid =
                    size_t(grid_u + origin_offset_uv) * grid_size +
                    size_t(grid_v + origin_offset_uv);
            my_atomic_add<VFP>(
                    &w_grid_stack[i_grid].x, vis_weighted.x * kernel_value
            );
            my_atomic_add<VFP>(
                    &w_grid_stack[i_grid].y, vis_weighted.y * kernel_value
            );
        }
    }
    if (vis_count) atomicAdd(vis_count, 1);
}


template<typename VFP, typename VFP2, typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_nifty_degrid_2d
(
        const int num_vis_rows,
        const int num_vis_chan,
        VFP2* __restrict__ visibilities, // Output visibilities
        const VFP* const __restrict__ vis_weights, // Input visibility weights
        const FP3* const __restrict__ uvw_coords, // (u, v, w) coordinates
        const FP* const __restrict__ freq_hz, // Frequency per channel
        const VFP2* const __restrict__ w_grid_stack, // Input grid
        const int grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int /*grid_start_w*/, // Index of w grid (always 0, for 2D)
        const uint32_t support, // full support for gridding kernel
        const VFP beta, // beta value used in exponential of semicircle kernel
        const FP uv_scale, // scaling factor for conversion of uv coords to grid coordinates (grid_size * cell_size)
        const FP w_scale, // scaling factor for converting w coord to signed w grid index
        const FP min_plane_w // w coordinate of smallest w plane
)
{
    const int i_chan = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_row  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;
    if (i_chan >= num_vis_chan || i_row >= num_vis_rows)
        return;

    // Calculate bounds of where gridding kernel will be applied for this visibility
    const int origin_offset_uv = (grid_size / 2); // offset of origin along u or v axes
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const VFP inv_half_support = (VFP)1.0 / (VFP)half_support;
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP inv_wavelength = freq_hz[i_chan] / (FP)299792458.0;
    const FP pos_u = uvw_coords[i_row].x * inv_wavelength * uv_scale;
    const FP pos_v = uvw_coords[i_row].y * inv_wavelength * uv_scale;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    if (grid_u_min > grid_u_max || grid_v_min > grid_v_max)
    {
        // This visibility does not intersect the current grid.
        return;
    }

    // Zero the visibility.
    VFP2 vis_tmp;
    vis_tmp.x = vis_tmp.y = (VFP)0;

    // Cache the kernel in v direction.
    VFP kernel_v[KERNEL_SUPPORT_BOUND];
    for (int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
    {
        kernel_v[grid_v - grid_v_min] = exp_semicircle(beta,
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
            // Update the grid, applying the separable kernel.
            VFP kernel_value = kernel_u * kernel_v[grid_v - grid_v_min];
            bool odd_grid_coordinate = ((grid_u + grid_v) & 1) != 0;
            kernel_value = odd_grid_coordinate ? -kernel_value : kernel_value;
            const size_t i_grid =
                    size_t(grid_u + origin_offset_uv) * grid_size +
                    size_t(grid_v + origin_offset_uv);
            vis_tmp.x += w_grid_stack[i_grid].x * kernel_value;
            vis_tmp.y += w_grid_stack[i_grid].y * kernel_value;
        }
    }
    visibilities[i_vis].x += vis_tmp.x * vis_weights[i_vis];
    visibilities[i_vis].y += vis_tmp.y * vis_weights[i_vis];
}


/**********************************************************************
 * Applies the w screen phase shift and accumulation of each w layer onto dirty image
 * Parallelised so each CUDA thread processes one pixel in each quadrant of the dirty image
 **********************************************************************/
template<typename FP, typename FP2>
__global__ void apply_w_screen_and_sum(
        FP* __restrict__ dirty_image, // INPUT & OUTPUT: real plane for accumulating phase corrected w layers across batches
        const uint32_t image_size, // one dimensional size of image plane (grid_size / sigma), assumed square
        const FP pixel_size, // converts pixel index (x, y) to normalised image coordinate (l, m) where l, m between -0.5 and 0.5
        const FP2* const __restrict__ w_grid_stack, // INPUT: flat array containing 2D computed w layers (w layer = iFFT(w grid))
        const uint32_t grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int grid_start_w, // index of first w grid in current subset stack
        const uint32_t num_w_grids_subset, // number of w grids bound in current subset stack
        const FP inv_w_scale, // inverse of scaling factor for converting w coord to signed w grid index
        const FP min_plane_w, // w coordinate of smallest w plane
        const bool perform_shift_fft, // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
        const bool do_wstacking
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t half_image_size = image_size / 2;

    if (i <= (int)half_image_size && j <= (int)half_image_size)  // allow extra in negative x and y directions, for asymmetric image centre
    {
        // Init pixel sums for the four quadrants
        FP pixel_sum_pos_pos = 0.0;
        FP pixel_sum_pos_neg = 0.0;
        FP pixel_sum_neg_pos = 0.0;
        FP pixel_sum_neg_neg = 0.0;

        const int origin_offset_grid_centre = (int)(grid_size / 2); // offset of origin (in w layer) along l or m axes
        const int grid_index_offset_image_centre = origin_offset_grid_centre *
                ((int)grid_size) + origin_offset_grid_centre;

        for (int grid_coord_w = grid_start_w; grid_coord_w < grid_start_w +
                (int)num_w_grids_subset; grid_coord_w++)
        {
            FP l = pixel_size * (FP)i;
            FP m = pixel_size * (FP)j;

            FP2 shift;
            if (do_wstacking)
            {
                FP w = (FP)grid_coord_w * inv_w_scale + min_plane_w;
                shift = phase_shift<FP, FP2>(w, l, m, FP(-1.0));
            }
            else
            {
                shift.x = 1.0;
                shift.y = 0.0;
            }

            int grid_index_offset_w = (grid_coord_w - grid_start_w) *
                    ((int)(grid_size * grid_size));
            int grid_index_image_centre = grid_index_offset_w +
                    grid_index_offset_image_centre;

            // Calculate the real component of the complex w layer value
            // multiplied by the complex phase shift.
            // Note w_grid_stack presumed to be larger than dirty_image
            // (sigma > 1) so has extra pixels around boundary.
            FP2 w_layer_pos_pos = w_grid_stack[grid_index_image_centre + j *
                            ((int)grid_size) + i];
            pixel_sum_pos_pos += w_layer_pos_pos.x * shift.x -
                    w_layer_pos_pos.y * shift.y;
            FP2 w_layer_pos_neg = w_grid_stack[grid_index_image_centre - j *
                            ((int)grid_size) + i];
            pixel_sum_pos_neg += w_layer_pos_neg.x * shift.x -
                    w_layer_pos_neg.y * shift.y;
            FP2 w_layer_neg_pos = w_grid_stack[grid_index_image_centre + j *
                            ((int)grid_size) - i];
            pixel_sum_neg_pos += w_layer_neg_pos.x * shift.x -
                    w_layer_neg_pos.y * shift.y;
            FP2 w_layer_neg_neg = w_grid_stack[grid_index_image_centre - j *
                            ((int)grid_size) - i];
            pixel_sum_neg_neg += w_layer_neg_neg.x * shift.x -
                    w_layer_neg_neg.y * shift.y;
        }

        // Equivalently rearrange each grid so origin is at lower-left
        // corner for FFT.
        bool odd_grid_coordinate = ((i + j) & 1) != 0;
        if (perform_shift_fft && odd_grid_coordinate)
        {
            pixel_sum_pos_pos = -pixel_sum_pos_pos;
            pixel_sum_pos_neg = -pixel_sum_pos_neg;
            pixel_sum_neg_pos = -pixel_sum_neg_pos;
            pixel_sum_neg_neg = -pixel_sum_neg_neg;
        }

        // Add the four pixel sums to the dirty image,
        // taking care to be within bounds for positive x and y quadrants.
        const int origin_offset_image_centre = (int)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int image_index_offset_image_centre = origin_offset_image_centre *
                ((int)image_size) + origin_offset_image_centre;

        // Special cases along centre or edges of image
        if (i < (int)half_image_size && j < (int)half_image_size)
        {
            dirty_image[image_index_offset_image_centre + j *
                    ((int)image_size) + i] += pixel_sum_pos_pos;
        }
        if (i > 0 && j < (int)half_image_size)
        {
            // Special case along centre of image doesn't update four pixels
            dirty_image[image_index_offset_image_centre + j *
                    ((int)image_size) - i] += pixel_sum_neg_pos;
        }
        if (j > 0 && i < (int)half_image_size)
        {
            // Special case along centre of image doesn't update four pixels
            dirty_image[image_index_offset_image_centre - j *
                    ((int)image_size) + i] += pixel_sum_pos_neg;
        }
        if (i > 0 && j > 0)
        {
            // Special case along centre of image doesn't update four pixels
            dirty_image[image_index_offset_image_centre - j *
                    ((int)image_size) - i] += pixel_sum_neg_neg;
        }
    }
}


/**********************************************************************
 * Reverses w screen phase shift for each w layer from dirty image
 * Parallelised so each CUDA thread processes one pixel in each quadrant of the dirty image
 **********************************************************************/
template<typename FP, typename FP2>
__global__ void reverse_w_screen_to_stack(
        const FP* const __restrict__ dirty_image, // INPUT: real plane for input dirty image
        const uint32_t image_size, // one dimensional size of image plane (grid_size / sigma), assumed square
        const FP pixel_size, // converts pixel index (x, y) to normalised image coordinate (l, m) where l, m between -0.5 and 0.5
        FP2* __restrict__ w_grid_stack, // OUTPUT: flat array containing 2D computed w layers (w layer = iFFT(w grid))
        const uint32_t grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
        const int grid_start_w, // index of first w grid in current subset stack
        const uint32_t num_w_grids_subset, // number of w grids bound in current subset stack
        const FP inv_w_scale, // inverse of scaling factor for converting w coord to signed w grid index
        const FP min_plane_w, // w coordinate of smallest w plane
        const bool perform_shift_fft, // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
        const bool do_wstacking
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t half_image_size = image_size / 2;

    if (i <= (int)half_image_size && j <= (int)half_image_size)  // allow extra in negative x and y directions, for asymmetric image centre
    {
        // Obtain four pixels from the dirty image, taking care to be within bounds for positive x and y quadrants
        const int origin_offset_image_centre = (int)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int image_index_offset_image_centre = origin_offset_image_centre *
                ((int)image_size) + origin_offset_image_centre;

        // Look up dirty pixels in four quadrants
        FP dirty_image_pos_pos = FP(0.0);
        FP dirty_image_neg_pos = FP(0.0);
        FP dirty_image_pos_neg = FP(0.0);
        FP dirty_image_neg_neg = FP(0.0);

        if (j < (int)half_image_size && i < (int)half_image_size)
        {
            dirty_image_pos_pos = dirty_image[image_index_offset_image_centre +
                            j * ((int)image_size) + i];
        }
        if (j < (int)half_image_size)
        {
            dirty_image_neg_pos = dirty_image[image_index_offset_image_centre +
                            j * ((int)image_size) - i];
        }
        if (i < (int)half_image_size)
        {
            dirty_image_pos_neg = dirty_image[image_index_offset_image_centre -
                            j * ((int)image_size) + i];
        }

        dirty_image_neg_neg = dirty_image[image_index_offset_image_centre - j *
                        ((int)image_size) - i];

        // Equivalently rearrange each grid so origin is at lower-left corner for FFT
        bool odd_grid_coordinate = ((i + j) & 1) != 0;
        if (perform_shift_fft && odd_grid_coordinate)
        {
            dirty_image_pos_pos = -dirty_image_pos_pos;
            dirty_image_pos_neg = -dirty_image_pos_neg;
            dirty_image_neg_pos = -dirty_image_neg_pos;
            dirty_image_neg_neg = -dirty_image_neg_neg;
        }

        const int origin_offset_grid_centre = (int)(grid_size / 2); // offset of origin (in w layer) along l or m axes
        const int grid_index_offset_image_centre = origin_offset_grid_centre *
                ((int)grid_size) + origin_offset_grid_centre;

        for (int grid_coord_w = grid_start_w; grid_coord_w < grid_start_w +
                (int)num_w_grids_subset; grid_coord_w++)
        {
            FP l = pixel_size * (FP)i;
            FP m = pixel_size * (FP)j;

            FP2 shift;
            if (do_wstacking)
            {
                FP w = (FP)grid_coord_w * inv_w_scale + min_plane_w;
                shift = phase_shift<FP, FP2>(w, l, m, FP(1.0));
                // shift.y = -shift.y; // inverse of original phase shift (equivalent to division)
            }
            else
            {
                shift.x = 1.0;
                shift.y = 0.0;
            }
            int grid_index_offset_w = (grid_coord_w - grid_start_w) *
                    ((int)(grid_size * grid_size));
            int grid_index_image_centre = grid_index_offset_w +
                    grid_index_offset_image_centre;

            // Calculate the complex product of the (real) dirty image by the complex phase shift
            // Special cases along centre or edges of image
            FP2 out;
            if (i < (int)half_image_size && j < (int)half_image_size)
            {
                out = shift;
                out.x *= dirty_image_pos_pos;
                out.y *= dirty_image_pos_pos;
                w_grid_stack[grid_index_image_centre + j * ((int)grid_size) +
                        i] = out;
            }
            if (j > 0 && i < (int)half_image_size)
            {
                out = shift;
                out.x *= dirty_image_pos_neg;
                out.y *= dirty_image_pos_neg;
                w_grid_stack[grid_index_image_centre - j * ((int)grid_size) +
                        i] = out;
            }
            if (i > 0 && j < (int)half_image_size)
            {
                out = shift;
                out.x *= dirty_image_neg_pos;
                out.y *= dirty_image_neg_pos;
                w_grid_stack[grid_index_image_centre + j * ((int)grid_size) -
                        i] = out;
            }
            if (i > 0 && j > 0)
            {
                out = shift;
                out.x *= dirty_image_neg_neg;
                out.y *= dirty_image_neg_neg;
                w_grid_stack[grid_index_image_centre - j * ((int)grid_size) -
                        i] = out;
            }
        }
    }
}


/**********************************************************************
 * Performs convolution correction and final scaling of dirty image
 * using precalculated and runtime calculated correction values.
 * See conv_corr device function for more details.
 * Note precalculated convolutional correction for (l, m) are normalised
 * to max of 1, but value for n is calculated at runtime, therefore
 * normalised at runtime by C(0).
 **********************************************************************/
template<typename FP>
__global__ void conv_corr_and_scaling(
        FP* dirty_image,
        const uint32_t image_size,
        const FP pixel_size,
        const uint32_t support,
        const FP conv_corr_norm_factor,
        const FP* conv_corr_kernel,
        const FP inv_w_range,
        const FP inv_w_scale,
        const FP* const __restrict__ quadrature_kernel,
        const FP* const __restrict__ quadrature_nodes,
        const FP* const __restrict__ quadrature_weights,
        const bool gridding,
        const bool do_wstacking
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t half_image_size = image_size / 2;

    if (i <= (int)half_image_size && j <= (int)half_image_size)
    {
        FP l = pixel_size * i;
        FP m = pixel_size * j;
        FP n = sqrt(FP(1.0) - l * l - m * m) - FP(1.0);
        FP l_conv = conv_corr_kernel[i];
        FP m_conv = conv_corr_kernel[j];

        FP n_conv = conv_corr((FP)support, n * inv_w_scale,
                quadrature_kernel, quadrature_nodes, quadrature_weights
        );
        n_conv *= (conv_corr_norm_factor * conv_corr_norm_factor);

        // Note: scaling (everything after division) does not appear to be present in reference NIFTY code
        // so it may need to be removed if testing this code against the reference code
        // repo: https://gitlab.mpcdf.mpg.de/ift/nifty_gridder
        FP correction =
                do_wstacking ? (l_conv * m_conv * n_conv) : (l_conv * m_conv *
                    conv_corr_norm_factor * conv_corr_norm_factor);
        // correction /= ((n + FP(1.0)) * inv_w_range); // see above note

        if (gridding)
        {
            // correction = FP(1.0)/(correction*weight_channel_product);
            correction = FP(1.0) / (correction);
        }
        else
        {
            correction = FP(1.0) / (correction);
        }

        // Going to need offsets to stride from pixel to pixel for this thread
        const int origin_offset_image_centre = (int)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int image_index_offset_image_centre = origin_offset_image_centre *
                ((int)image_size) + origin_offset_image_centre;

        if (i < (int)half_image_size && j < (int)half_image_size)
        {
            dirty_image[image_index_offset_image_centre + j *
                    ((int)image_size) + i] *= correction;
        }
        // Special cases along centre of image doesn't update four pixels
        if (i > 0 && j < (int)half_image_size)
        {
            dirty_image[image_index_offset_image_centre + j *
                    ((int)image_size) - i] *= correction;
        }
        if (j > 0 && i < (int)half_image_size)
        {
            dirty_image[image_index_offset_image_centre - j *
                    ((int)image_size) + i] *= correction;
        }
        if (i > 0 && j > 0)
        {
            dirty_image[image_index_offset_image_centre - j *
                    ((int)image_size) - i] *= correction;
        }
    }
}

// *INDENT-OFF*
// register kernels
SDP_CUDA_KERNEL(sdp_cuda_tile_count<float, float2, float3>)
SDP_CUDA_KERNEL(sdp_cuda_tile_count<double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_tile_bucket_sort<float, float2, float3>)
SDP_CUDA_KERNEL(sdp_cuda_tile_bucket_sort<double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_grid_tiled_3d<float, float2, 8, 8>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_grid_tiled_3d<double, double2, 8, 8>)

SDP_CUDA_KERNEL(sdp_cuda_nifty_grid_3d<double, double2, double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_grid_3d<float,  float2,  float,  float2,  float3>)

SDP_CUDA_KERNEL(sdp_cuda_nifty_grid_2d<double, double2, double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_grid_2d<float,  float2,  float,  float2,  float3>)

SDP_CUDA_KERNEL(sdp_cuda_nifty_degrid_3d<double, double2, double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_degrid_3d<float,  float2,  float,  float2,  float3>)

SDP_CUDA_KERNEL(sdp_cuda_nifty_degrid_2d<double, double2, double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_degrid_2d<float,  float2,  float,  float2,  float3>)

SDP_CUDA_KERNEL(apply_w_screen_and_sum<double, double2>)
SDP_CUDA_KERNEL(apply_w_screen_and_sum<float, float2>)

SDP_CUDA_KERNEL(reverse_w_screen_to_stack<double, double2>)
SDP_CUDA_KERNEL(reverse_w_screen_to_stack<float, float2>)

SDP_CUDA_KERNEL(conv_corr_and_scaling<double>)
SDP_CUDA_KERNEL(conv_corr_and_scaling<float>)
// *INDENT-ON*
