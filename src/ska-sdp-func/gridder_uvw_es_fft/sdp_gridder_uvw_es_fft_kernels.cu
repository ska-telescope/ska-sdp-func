/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"

#define KERNEL_SUPPORT_BOUND 16

#ifndef PI
#define PI 3.1415926535897931
#endif

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
        old_ = atomicCAS(laddr, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    }
    while (assumed != old_);
#endif
}

/**********************************************************************
* Evaluates the convolutional correction C(k) in one dimension.
* As the exponential of semicircle gridding kernel psi(u,v,w) is separable its Fourier
* transform Fourier(psi)(l,m,n) is likewise separable into one-dimensional components C(l)C(m)C(n).
* As psi is even and zero outside its support the convolutional correction is given by:
*   C(k) = 2\integral_{0}^{supp/2} psi(u)cos(2\pi ku) du = supp\integral_{0}^{1}psi(supp*x)cos(\pi*k*supp*x) dx by change of variables
* This integral from 0 to 1 can be numerically approximated via a 2p-node Gauss-Legendre quadrature, as recommended 
* in equation 3.10 of ‘A Parallel Non-uniform Fast Fourier Transform Library Based on an “Exponential of Semicircle” Kernel’
* by Barnett, Magland, Klintenberg and only using the p positive nodes):
*   C(k) ~ Sum_{i=1}^{p} weight_i*psi(supp*node_i)*cos(\pi*k*supp*node_i)
* Note this convolutional correction is not normalised, 
* but is normalised during use in convolution correction by C(0) to get max of 1
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
    uint p = (uint)(ceil(FP(1.5)*support + FP(2.0)));

    for (uint i = 0; i < p; i++)
        correction += quadrature_kernel[i] *
                cos(PI * k * support * quadrature_nodes[i]) *
                quadrature_weights[i];

    return correction*support;
}

/**********************************************************************
 * Calculates the exponential of semicircle
 * Note the parameter x must be normalised to be in range [-1,1]
 * Source Paper: A parallel non-uniform fast Fourier transform library based on an "exponential of semicircle" kernel
 * Address: https://arxiv.org/abs/1808.06736
 **********************************************************************/
template<typename VFP>
__device__ VFP exp_semicircle(const VFP beta, const VFP x)
{
    const VFP xx = x*x;
    
    return ((xx > VFP(1.0)) ? VFP(0.0) : exp(beta*(sqrt(VFP(1.0) - xx) - VFP(1.0))));
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
    const FP sos = l*l + m*m;
    const FP nm1 = (-sos) / (sqrt(FP(1.0) - sos) + FP(1.0));
    const FP x = FP(2.0) * FP(PI) * w * nm1;
    const FP xn = FP(1.0) / (nm1 + FP(1.0));
    // signage = -1.0 if gridding, 1.0 if degridding
    sincos(signage * x, &phasor.y, &phasor.x);
    phasor.x *= xn;
    phasor.y *= xn;
    return phasor;
} 

template<typename VFP, typename VFP2, typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_nifty_gridder_gridding_3d
(
    const int num_vis_rows,
    const int num_vis_chan,
    
          VFP2*       __restrict__ visibilities, // INPUT(gridding) OR OUTPUT(degridding): complex visibilities
    const VFP*  const __restrict__ vis_weights,  // INPUT: weight for each visibility
    const FP3*  const __restrict__ uvw_coords,   // INPUT: (u, v, w) coordinates for each visibility
    const FP*   const __restrict__ freq_hz,      // INPUT: array of frequencies per channel
          VFP2*       __restrict__ w_grid_stack, // OUTPUT: flat array containing 2D computed w grids, presumed initially clear
          
    const int  grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
    const int  grid_start_w, // signed index of first w grid in current subset stack
    const uint num_w_grids_subset, // number of w grids bound in current subset stack

    const uint support, // full support for gridding kernel
    const VFP  beta, // beta constant used in exponential of semicircle kernel
    const FP   uv_scale, // scaling factor for conversion of uv coords to grid coordinates (grid_size * cell_size)
    const FP   w_scale, // scaling factor for converting w coord to signed w grid index
    const FP   min_plane_w, // w coordinate of smallest w plane
    const bool solving // flag to enable degridding operations instead of gridding
)
{
    const int i_chan = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_row  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;

    if (i_chan >= num_vis_chan || i_row >= num_vis_rows) 
        return;

    // Determine whether to flip visibility coordinates, so w is usually positive
    const FP flip = (uvw_coords[i_row].z < 0.0) ? -1.0 : 1.0;
    const FP inv_wavelength = flip * freq_hz[i_chan] / (FP)299792458.0;

    // Get the weighted visibility.
    VFP2 vis_weighted;
    if (solving)
    {
        vis_weighted.x = visibilities[i_vis].x * vis_weights[i_vis];
        vis_weighted.y = visibilities[i_vis].y * vis_weights[i_vis];
        vis_weighted.y *= flip; // complex conjugate for negative w coords
    }
    else
    {
        vis_weighted.x = vis_weighted.y = (VFP)0.0;
    }

    // Calculate bounds of where gridding kernel will be applied for this visibility
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP pos_u = uvw_coords[i_row].x * inv_wavelength * uv_scale;
    const FP pos_v = uvw_coords[i_row].y * inv_wavelength * uv_scale;
    const FP pos_w = (uvw_coords[i_row].z * inv_wavelength - min_plane_w) * w_scale;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    const int grid_w_min = max((int)ceil(pos_w - half_support), grid_start_w);
    const int grid_w_max = min((int)floor(pos_w + half_support), grid_start_w + num_w_grids_subset - 1);
    if (grid_w_min > grid_w_max ||
        grid_u_min > grid_u_max ||
        grid_v_min > grid_v_max)
    {
        // this visibility has no overlap with the current subset stack
        return;
    }

    // Calculate kernel values along u and v directions for this uvw
    VFP inv_half_support = (VFP)1.0 / (VFP)half_support;
    // bound above the maximum possible support when precalculating kernel values
    VFP kernel_u[KERNEL_SUPPORT_BOUND], kernel_v[KERNEL_SUPPORT_BOUND];
    for (int grid_u = grid_u_min; grid_u <= grid_u_max; grid_u++)
    {
        kernel_u[grid_u - grid_u_min] = exp_semicircle(beta,
                (VFP)(grid_u - pos_u) * inv_half_support);
    }
    for (int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
    {
        kernel_v[grid_v - grid_v_min] = exp_semicircle(beta,
                (VFP)(grid_v - pos_v) * inv_half_support);
    }

    // Iterate through each w-grid
    const int origin_offset_uv = (grid_size / 2); // offset of origin along u or v axes
    for (int grid_w = grid_w_min; grid_w <= grid_w_max; grid_w++)
    {
        const VFP kernel_w = exp_semicircle(beta,
                (VFP)(grid_w - pos_w) * inv_half_support);
        const size_t grid_offset_w = (grid_w - grid_start_w) *
                size_t(grid_size * grid_size);

        // Swapped u and v for consistency with original nifty gridder.
        for (int grid_u = grid_u_min; grid_u <= grid_u_max; grid_u++)
        {
            for (int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
            {
                // Apply the separable kernel to the weighted visibility.
                VFP kernel_value = kernel_u[grid_u - grid_u_min] *
                        kernel_v[grid_v - grid_v_min] * kernel_w;
                bool odd_grid_coordinate = ((grid_u + grid_v) & 1) != 0;
                kernel_value = odd_grid_coordinate ? -kernel_value : kernel_value;

                // Update or access the grid.
                const size_t grid_offset_uvw = grid_offset_w +
                        size_t(grid_u + origin_offset_uv) * grid_size +
                        size_t(grid_v + origin_offset_uv);
                        
                    if(solving) // accumulation of visibility onto w-grid plane
                {
                    // accumulation of visibility onto w-grid plane
                    my_atomic_add<VFP>(
                            &w_grid_stack[grid_offset_uvw].x,
                            vis_weighted.x * kernel_value);
                    my_atomic_add<VFP>(
                            &w_grid_stack[grid_offset_uvw].y,
                            vis_weighted.y * kernel_value);
                }
                    else // extraction of visibility from w-grid plane
                {
                        vis_weighted.x += w_grid_stack[grid_offset_uvw].x * kernel_value;
                        vis_weighted.y += w_grid_stack[grid_offset_uvw].y * kernel_value;
                }
            }
        }
    }

    if(!solving) // degridding
    {
        visibilities[i_vis].x += vis_weighted.x;
        visibilities[i_vis].y += vis_weighted.y * flip;
    }
}

/**********************************************************************
 * Performs the gridding (or degridding) of visibilities across a subset of w planes
 * Parallelised so each CUDA thread processes a single visibility
 **********************************************************************/
template<typename VFP, typename VFP2, typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_nifty_gridder_gridding_2d
(
    const int num_vis_rows,
    const int num_vis_chan,
    
          VFP2*       __restrict__ visibilities, // INPUT(gridding) OR OUTPUT(degridding): complex visibilities
    const VFP*  const __restrict__ vis_weights,  // INPUT: weight for each visibility
    const FP3*  const __restrict__ uvw_coords,   // INPUT: (u, v, w) coordinates for each visibility
    const FP*   const __restrict__ freq_hz,      // INPUT: array of frequencies per channel
          VFP2*       __restrict__ w_grid_stack, // OUTPUT: flat array containing 2D computed w grids, presumed initially clear

    const int  grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
    const int  grid_start_w, // signed index of first w grid in current subset stack
    const uint num_w_grids_subset, // number of w grids bound in current subset stack

    const uint support, // full support for gridding kernel
    const VFP beta, // beta constant used in exponential of semicircle kernel
    const FP uv_scale, // scaling factor for conversion of uv coords to grid coordinates (grid_size * cell_size)
    const FP w_scale, // scaling factor for converting w coord to signed w grid index
    const FP min_plane_w, // w coordinate of smallest w plane
    const bool solving // flag to enable degridding operations instead of gridding
)
{
    const int i_chan = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_row  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;

    if (i_chan >= num_vis_chan || i_row >= num_vis_rows) 
        return;

    // Determine whether to flip visibility coordinates, so w is usually positive
    const FP flip = 1.0;  // (uvw_coords[i_row].z < 0.0) ? -1.0 : 1.0;  // ignoring w
    const FP inv_wavelength = flip * freq_hz[i_chan] / (FP)299792458.0;

    // Get the weighted visibility.
    VFP2 vis_weighted;
    if (solving)
    {
        vis_weighted.x = visibilities[i_vis].x * vis_weights[i_vis];
        vis_weighted.y = visibilities[i_vis].y * vis_weights[i_vis];
        vis_weighted.y *= flip; // complex conjugate for negative w coords
    }
    else
    {
        vis_weighted.x = vis_weighted.y = (VFP)0.0;
    }

    // Calculate bounds of where gridding kernel will be applied for this visibility
    const FP half_support = FP(support) / FP(2.0); // NOTE confirm everyone's understanding of what support means eg when even/odd
    const int grid_min_uv = -grid_size / 2; // minimum coordinate on grid along u or v axes
    const int grid_max_uv = (grid_size - 1) / 2; // maximum coordinate on grid along u or v axes
    const FP pos_u = uvw_coords[i_row].x * inv_wavelength * uv_scale;
    const FP pos_v = uvw_coords[i_row].y * inv_wavelength * uv_scale;
    const FP pos_w = 0.0; // ignoring w // (uvw_coords[i_row].z * inv_wavelength - min_plane_w) * w_scale;
    const int grid_u_min = max((int)ceil(pos_u - half_support), grid_min_uv);
    const int grid_u_max = min((int)floor(pos_u + half_support), grid_max_uv);
    const int grid_v_min = max((int)ceil(pos_v - half_support), grid_min_uv);
    const int grid_v_max = min((int)floor(pos_v + half_support), grid_max_uv);
    const int grid_w_min = max((int)ceil(pos_w - half_support), grid_start_w);
    const int grid_w_max = min((int)floor(pos_w + half_support), grid_start_w + num_w_grids_subset - 1);
    
    if (i_vis == -1)
    {
        printf("inv_wavelength is %.12e\n", inv_wavelength);
        printf("uv_scale is %.12e\n", uv_scale);
        printf("pos_u, pos_v is [%.12e, %.12e]\n", pos_u, pos_v);
    }
    if (i_vis == -1)
    {
        printf("grid_w_min   is %i\n", grid_w_min);
        printf("grid_w_max   is %i\n", grid_w_max);
        printf("grid_start_w is %i\n", grid_start_w);
    }

    if (grid_w_min > grid_w_max ||
            grid_u_min > grid_u_max ||
            grid_v_min > grid_v_max)
    {
        // this visibility has no overlap with the current subset stack
        return;
    }

    // Calculate kernel values along u and v directions for this uvw
    VFP inv_half_support = (VFP)1.0 / (VFP)half_support;

    // bound above the maximum possible support when precalculating kernel values
    VFP kernel_u[KERNEL_SUPPORT_BOUND], kernel_v[KERNEL_SUPPORT_BOUND];
    for (int grid_u = grid_u_min; grid_u <= grid_u_max; grid_u++)
    {
        kernel_u[grid_u - grid_u_min] = exp_semicircle(beta,
                (VFP)(grid_u - pos_u) * inv_half_support);

        if (i_vis == -1)
        {
            printf("grid_u, pos_u is [%i, %.12e]\n", grid_u, pos_u);
            printf("(VFP)(grid_u - pos_u) is %.12e\n", (VFP)(grid_u - pos_u));
            printf("(VFP)(grid_u - pos_u) * inv_half_support is %.12e\n", (VFP)(grid_u - pos_u) * inv_half_support);
            printf("kernel_u[%i] is %.12e\n", grid_u, kernel_u[grid_u - grid_u_min]);
        }
    }
    for (int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
    {
        kernel_v[grid_v - grid_v_min] = exp_semicircle(beta,
                (VFP)(grid_v - pos_v) * inv_half_support);

        if (i_vis == -1)
        {
            printf("kernel_v[%i] is %.12e\n", grid_v, kernel_v[grid_v - grid_v_min]);
        }
    }

    // Iterate through each w-grid
    const int origin_offset_uv = (grid_size / 2); // offset of origin along u or v axes
    for (int grid_w = grid_w_min; grid_w <= grid_w_max; grid_w++)
    {
        const VFP kernel_w = exp_semicircle(beta,
                (VFP)(grid_w - pos_w) * inv_half_support);
        const size_t grid_offset_w = (grid_w - grid_start_w) *
                size_t(grid_size * grid_size);

        if (i_vis == -1)
        {
            printf("grid_offset_w   is %li\n", grid_offset_w);
            printf("kernel_w   is %f\n", kernel_w);
        }

        // Swapped u and v for consistency with original nifty gridder.
        for (int grid_u = grid_u_min; grid_u <= grid_u_max; grid_u++)
        {
            for (int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
            {
                // Apply the separable kernel to the weighted visibility.
                VFP kernel_value = kernel_u[grid_u - grid_u_min] *
                        kernel_v[grid_v - grid_v_min] * kernel_w;
                bool odd_grid_coordinate = ((grid_u + grid_v) & 1) != 0;
                kernel_value = odd_grid_coordinate ? -kernel_value : kernel_value;

                // Update or access the grid.
                const size_t grid_offset_uvw = grid_offset_w +
                        size_t(grid_u + origin_offset_uv) * grid_size +
                        size_t(grid_v + origin_offset_uv);
                        
                if(solving) // accumulation of visibility onto w-grid plane
                {
                    // accumulation of visibility onto w-grid plane
                    my_atomic_add<VFP>(
                            &w_grid_stack[grid_offset_uvw].x,
                            vis_weighted.x * kernel_value);
                    my_atomic_add<VFP>(
                            &w_grid_stack[grid_offset_uvw].y,
                            vis_weighted.y * kernel_value);
                }
                else // extraction of visibility from w-grid plane
                {
                        vis_weighted.x += w_grid_stack[grid_offset_uvw].x * kernel_value;
                        vis_weighted.y += w_grid_stack[grid_offset_uvw].y * kernel_value;

                        //printf("vis_weighted is [%.12e, %.12e]\n\n", vis_weighted.x, vis_weighted.y);
                }
            }
        }
    }

    if(!solving) // degridding
    {
        visibilities[i_vis].x += vis_weighted.x;
        visibilities[i_vis].y += vis_weighted.y * flip;
    }
}

/**********************************************************************
 * Applies the w screen phase shift and accumulation of each w layer onto dirty image
 * Parallelised so each CUDA thread processes one pixel in each quadrant of the dirty image
 **********************************************************************/
template<typename FP, typename FP2>
__global__ void apply_w_screen_and_sum(
    FP* __restrict__ dirty_image, // INPUT & OUTPUT: real plane for accumulating phase corrected w layers across batches
    const uint image_size, // one dimensional size of image plane (grid_size / sigma), assumed square
    const FP pixel_size, // converts pixel index (x, y) to normalised image coordinate (l, m) where l, m between -0.5 and 0.5
    const FP2* const __restrict__ w_grid_stack, // INPUT: flat array containing 2D computed w layers (w layer = iFFT(w grid))
    const uint grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
    const int grid_start_w, // index of first w grid in current subset stack
    const uint num_w_grids_subset, // number of w grids bound in current subset stack
    const FP inv_w_scale, // inverse of scaling factor for converting w coord to signed w grid index
    const FP min_plane_w, // w coordinate of smallest w plane
    const bool perform_shift_fft,  // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
    const bool do_wstacking
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint half_image_size = image_size / 2;
    
    if(i <= (int)half_image_size && j <= (int)half_image_size)  // allow extra in negative x and y directions, for asymmetric image centre
    {
        // Init pixel sums for the four quadrants
        FP pixel_sum_pos_pos = 0.0;
        FP pixel_sum_pos_neg = 0.0;
        FP pixel_sum_neg_pos = 0.0;
        FP pixel_sum_neg_neg = 0.0;

        const int origin_offset_grid_centre = (int)(grid_size/2); // offset of origin (in w layer) along l or m axes
        const int grid_index_offset_image_centre = origin_offset_grid_centre*((int)grid_size) + origin_offset_grid_centre;

        for (int grid_coord_w=grid_start_w; grid_coord_w < grid_start_w + (int)num_w_grids_subset; grid_coord_w++)
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
            
            int grid_index_offset_w = (grid_coord_w-grid_start_w)*((int)(grid_size*grid_size));
            int grid_index_image_centre = grid_index_offset_w + grid_index_offset_image_centre;
            
            // Calculate the real component of the complex w layer value multiplied by the complex phase shift
            // Note w_grid_stack presumed to be larger than dirty_image (sigma > 1) so has extra pixels around boundary
            FP2 w_layer_pos_pos = w_grid_stack[grid_index_image_centre + j*((int)grid_size) + i];
            pixel_sum_pos_pos += w_layer_pos_pos.x*shift.x - w_layer_pos_pos.y*shift.y;
            FP2 w_layer_pos_neg = w_grid_stack[grid_index_image_centre - j*((int)grid_size) + i];
            pixel_sum_pos_neg += w_layer_pos_neg.x*shift.x - w_layer_pos_neg.y*shift.y;
            FP2 w_layer_neg_pos = w_grid_stack[grid_index_image_centre + j*((int)grid_size) - i];
            pixel_sum_neg_pos += w_layer_neg_pos.x*shift.x - w_layer_neg_pos.y*shift.y;
            FP2 w_layer_neg_neg = w_grid_stack[grid_index_image_centre - j*((int)grid_size) - i];
            pixel_sum_neg_neg += w_layer_neg_neg.x*shift.x - w_layer_neg_neg.y*shift.y;
        }

        // Equivalently rearrange each grid so origin is at lower-left corner for FFT
        bool odd_grid_coordinate = ((i+j) & 1) != 0;
        if(perform_shift_fft && odd_grid_coordinate)
        {
            pixel_sum_pos_pos = -pixel_sum_pos_pos;
            pixel_sum_pos_neg = -pixel_sum_pos_neg;
            pixel_sum_neg_pos = -pixel_sum_neg_pos;
            pixel_sum_neg_neg = -pixel_sum_neg_neg;
        }

        // Add the four pixel sums to the dirty image taking care to be within bounds for positive x and y quadrants
        const int origin_offset_image_centre = (int)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int image_index_offset_image_centre = origin_offset_image_centre*((int)image_size) + origin_offset_image_centre;

        // Special cases along centre or edges of image
        if(i < (int)half_image_size && j < (int)half_image_size)
        {
            dirty_image[image_index_offset_image_centre + j*((int)image_size) + i] += pixel_sum_pos_pos;
        }
        if(i > 0 && j < (int)half_image_size)
        {
            // Special case along centre of image doesn't update four pixels
            dirty_image[image_index_offset_image_centre + j*((int)image_size) - i] += pixel_sum_neg_pos;
        }
        if(j > 0 && i < (int)half_image_size)
        {
            // Special case along centre of image doesn't update four pixels
            dirty_image[image_index_offset_image_centre - j*((int)image_size) + i] += pixel_sum_pos_neg;
        }
        if(i > 0 && j > 0)
        {
            // Special case along centre of image doesn't update four pixels
            dirty_image[image_index_offset_image_centre - j*((int)image_size) - i] += pixel_sum_neg_neg;
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
    const uint image_size, // one dimensional size of image plane (grid_size / sigma), assumed square
    const FP pixel_size, // converts pixel index (x, y) to normalised image coordinate (l, m) where l, m between -0.5 and 0.5
    FP2* __restrict__ w_grid_stack, // OUTPUT: flat array containing 2D computed w layers (w layer = iFFT(w grid))
    const uint grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
    const int grid_start_w, // index of first w grid in current subset stack
    const uint num_w_grids_subset, // number of w grids bound in current subset stack
    const FP inv_w_scale, // inverse of scaling factor for converting w coord to signed w grid index
    const FP min_plane_w, // w coordinate of smallest w plane
    const bool perform_shift_fft,  // flag to (equivalently) rearrange each grid so origin is at lower-left corner for FFT
    const bool do_wstacking
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint half_image_size = image_size / 2;

    if (i <= (int)half_image_size && j <= (int)half_image_size)  // allow extra in negative x and y directions, for asymmetric image centre
    {
        // Obtain four pixels from the dirty image, taking care to be within bounds for positive x and y quadrants
        const int origin_offset_image_centre = (int)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int image_index_offset_image_centre = origin_offset_image_centre*((int)image_size) + origin_offset_image_centre;

        // Look up dirty pixels in four quadrants
        FP dirty_image_pos_pos = FP(0.0);
        FP dirty_image_neg_pos = FP(0.0);
        FP dirty_image_pos_neg = FP(0.0);
        FP dirty_image_neg_neg = FP(0.0);

        if (j < (int)half_image_size && i < (int)half_image_size)
            dirty_image_pos_pos = dirty_image[image_index_offset_image_centre + j*((int)image_size) + i];
        if (j < (int)half_image_size)
            dirty_image_neg_pos = dirty_image[image_index_offset_image_centre + j*((int)image_size) - i];
        if (i < (int)half_image_size)
            dirty_image_pos_neg = dirty_image[image_index_offset_image_centre - j*((int)image_size) + i];
        
        dirty_image_neg_neg = dirty_image[image_index_offset_image_centre - j*((int)image_size) - i];

        // Equivalently rearrange each grid so origin is at lower-left corner for FFT
        bool odd_grid_coordinate = ((i+j) & 1) != 0;
        if(perform_shift_fft && odd_grid_coordinate)
        {
            dirty_image_pos_pos = -dirty_image_pos_pos;
            dirty_image_pos_neg = -dirty_image_pos_neg;
            dirty_image_neg_pos = -dirty_image_neg_pos;
            dirty_image_neg_neg = -dirty_image_neg_neg;
        }

        const int origin_offset_grid_centre = (int)(grid_size/2); // offset of origin (in w layer) along l or m axes
        const int grid_index_offset_image_centre = origin_offset_grid_centre*((int)grid_size) + origin_offset_grid_centre;

        for (int grid_coord_w=grid_start_w; grid_coord_w < grid_start_w + (int)num_w_grids_subset; grid_coord_w++)
        {
            FP l = pixel_size * (FP)i;
            FP m = pixel_size * (FP)j;
            
            FP2 shift;
            if (do_wstacking)
            {
                FP w = (FP)grid_coord_w * inv_w_scale + min_plane_w; 
                shift = phase_shift<FP, FP2>(w, l, m, FP(1.0));
                //shift.y = -shift.y; // inverse of original phase shift (equivalent to division)
            }
            else
            {
                shift.x = 1.0;
                shift.y = 0.0;
            }
            int grid_index_offset_w = (grid_coord_w-grid_start_w)*((int)(grid_size*grid_size));
            int grid_index_image_centre = grid_index_offset_w + grid_index_offset_image_centre;
            
            // Calculate the complex product of the (real) dirty image by the complex phase shift
            // Special cases along centre or edges of image
            FP2 out;
            if(i < (int)half_image_size && j < (int)half_image_size)
            {   
                out = shift;
                out.x *= dirty_image_pos_pos;
                out.y *= dirty_image_pos_pos;
                w_grid_stack[grid_index_image_centre + j*((int)grid_size) + i] = out;
            }
            if(j > 0 && i < (int)half_image_size)
            {   
                out = shift;
                out.x *= dirty_image_pos_neg;
                out.y *= dirty_image_pos_neg;
                w_grid_stack[grid_index_image_centre - j*((int)grid_size) + i] = out;
            }
            if(i > 0 && j < (int)half_image_size)
            {   
                out = shift;
                out.x *= dirty_image_neg_pos;
                out.y *= dirty_image_neg_pos;
                w_grid_stack[grid_index_image_centre + j*((int)grid_size) - i] = out;
            }
            if(i > 0 && j > 0)
            {   
                out = shift;
                out.x *= dirty_image_neg_neg;
                out.y *= dirty_image_neg_neg;
                w_grid_stack[grid_index_image_centre - j*((int)grid_size) - i] = out;
            }
        }
    }
}

/**********************************************************************
 * Performs convolution correction and final scaling of dirty image
 * using precalculated and runtime calculated correction values.
 * See conv_corr device function for more details
 * Note precalculated convolutional correction for (l, m) are normalised to max of 1,
 * but value for n is calculated at runtime, therefore normalised at runtime by C(0)
 **********************************************************************/
template<typename FP>
__global__ void conv_corr_and_scaling(
    FP *dirty_image,
    const uint image_size,
    const FP pixel_size,
    const uint support,
    const FP conv_corr_norm_factor,
    const FP *conv_corr_kernel,
    const FP inv_w_range,
    const FP inv_w_scale,
    const FP* const __restrict__ quadrature_kernel,
    const FP* const __restrict__ quadrature_nodes,
    const FP* const __restrict__ quadrature_weights,
    const bool solving,
    const bool do_wstacking
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint half_image_size = image_size / 2;

    if(i <= (int)half_image_size && j <= (int)half_image_size)
    {
        FP l = pixel_size * i;
        FP m = pixel_size * j;
        FP n = sqrt(FP(1.0)-l*l-m*m) - FP(1.0);
        FP l_conv = conv_corr_kernel[i];
        FP m_conv = conv_corr_kernel[j];

        FP n_conv = conv_corr((FP)support, n * inv_w_scale,
                quadrature_kernel, quadrature_nodes, quadrature_weights);
        n_conv *= (conv_corr_norm_factor * conv_corr_norm_factor);

        // Note: scaling (everything after division) does not appear to be present in reference NIFTY code
        // so it may need to be removed if testing this code against the reference code
        // repo: https://gitlab.mpcdf.mpg.de/ift/nifty_gridder
        FP correction = do_wstacking ? (l_conv * m_conv * n_conv) : (l_conv * m_conv * conv_corr_norm_factor * conv_corr_norm_factor); 
        // correction /= ((n + FP(1.0)) * inv_w_range); // see above note

        if(solving)
            //correction = FP(1.0)/(correction*weight_channel_product);
            correction = FP(1.0)/(correction);
        else
            correction = FP(1.0)/(correction);

        // Going to need offsets to stride from pixel to pixel for this thread
        const int origin_offset_image_centre = (int)half_image_size; // offset of origin (in dirty image) along l or m axes
        const int image_index_offset_image_centre = origin_offset_image_centre*((int)image_size) + origin_offset_image_centre;

        if(i < (int)half_image_size && j < (int)half_image_size)
        {
            dirty_image[image_index_offset_image_centre + j*((int)image_size) + i] *= correction; 
        }
        // Special cases along centre of image doesn't update four pixels
        if(i > 0 && j < (int)half_image_size)
        {
            dirty_image[image_index_offset_image_centre + j*((int)image_size) - i] *= correction; 
        }
        if(j > 0 && i < (int)half_image_size)
        {
            dirty_image[image_index_offset_image_centre - j*((int)image_size) + i] *= correction; 
        }
        if(i > 0 && j > 0)
        {
            dirty_image[image_index_offset_image_centre - j*((int)image_size) - i] *= correction; 
        }
    }
}

template<typename T>
void get_w_range(const int num_rows, const T* uvw, const int num_chan, const T* freq_hz,
        double& min_abs_w, double& max_abs_w)
{
    for (int i = 0; i < num_rows; ++i)
    {
        const double abs_w = std::fabs(uvw[3 * i + 2]);
        if (abs_w < min_abs_w) min_abs_w = abs_w;
        if (abs_w > max_abs_w) max_abs_w = abs_w;
    }

    double fscaleMin = freq_hz[0         ]/299792458.0;
    double fscaleMax = freq_hz[num_chan-1]/299792458.0;
    
    min_abs_w *= fscaleMin;
    max_abs_w *= fscaleMax; 
}

// register kernels
SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_3d<double, double2, double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_3d<float,  float2,  double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_3d<float,  float2,  float,  float2,  float3>)

SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_2d<double, double2, double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_2d<float,  float2,  double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_2d<float,  float2,  float,  float2,  float3>)

SDP_CUDA_KERNEL(apply_w_screen_and_sum<double, double2>)
SDP_CUDA_KERNEL(apply_w_screen_and_sum<float, float2>)

SDP_CUDA_KERNEL(reverse_w_screen_to_stack<double, double2>)
SDP_CUDA_KERNEL(reverse_w_screen_to_stack<float, float2>)

SDP_CUDA_KERNEL(conv_corr_and_scaling<double>)
SDP_CUDA_KERNEL(conv_corr_and_scaling<float>)

SDP_CUDA_KERNEL(get_w_range<double>)
SDP_CUDA_KERNEL(get_w_range<float>)
