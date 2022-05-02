/* See the LICENSE file at the top-level directory of this distribution. */

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
        old_ = atomicCAS(laddr, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    }
    while (assumed != old_);
#endif
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
	
	if(0)
	{
		printf("xx is %.16f\n", xx);
		printf("1 - xx is %.16f\n", VFP(1.0) - xx);
		printf("sqrt(1 - xx) is %.16f\n", sqrt(VFP(1.0) - xx));
		printf("sqrt(1 - xx) - 1 is %.16f\n", sqrt(VFP(1.0) - xx) - VFP(1.0));
	}
	
	return ((xx > VFP(1.0)) ? VFP(0.0) : exp(beta*(sqrt(VFP(1.0) - xx) - VFP(1.0))));
}



template<typename VFP, typename VFP2, typename FP, typename FP2, typename FP3>
__global__ void sdp_cuda_nifty_gridder_gridding_2d(
    const int                                num_vis_rows,
    const int                                num_vis_chan,
    
    VFP2*             __restrict__ visibilities, // INPUT(gridding) OR OUTPUT(degridding): complex visibilities
    const VFP*  const __restrict__ vis_weights, // INPUT: weight for each visibility
    const FP3*     const __restrict__ uvw_coords, // INPUT: (u, v, w) coordinates for each visibility

    const FP*      const __restrict__ freq_hz, // INPUT: array of frequencies per channel
    VFP2*             __restrict__ w_grid_stack, // OUTPUT: flat array containing 2D computed w grids, presumed initially clear
    const int                                grid_size, // one dimensional size of w_plane (image_size * upsampling), assumed square
    const int                                grid_start_w, // signed index of first w grid in current subset stack
    const uint                                num_w_grids_subset, // number of w grids bound in current subset stack

    const uint                                support, // full support for gridding kernel
    const VFP                      beta, // beta constant used in exponential of semicircle kernel
    const FP                          uv_scale, // scaling factor for conversion of uv coords to grid coordinates (grid_size * cell_size)
    const FP                          w_scale, // scaling factor for converting w coord to signed w grid index
    const FP                          min_plane_w, // w coordinate of smallest w plane
    const bool                                solving // flag to enable degridding operations instead of gridding
)
{
    const int i_chan = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_row  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_vis = i_chan + num_vis_chan * i_row;

    if (i_chan >= num_vis_chan || i_row >= num_vis_rows) 
    	return;

    //if (i_chan > 2 || i_row > 5) // AG Debug
    //if (i_row > 0) // AG Debug
    //	return;

	//printf("\n[i_row, i_chan, i_vis] = [%8i, %1i, %8i]\n", i_row, i_chan, i_vis);

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

		if (i_vis == -1)
		{
			printf("grid_offset_w   is %i\n", grid_offset_w);
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
                        
				if (0) //(i_chan == 0 && i_row == 0)
				{
					printf("kernel[u,v,w] is [%e, %e, %e]\n", kernel_u[grid_u - grid_u_min], kernel_v[grid_v - grid_v_min], kernel_w);
					printf("vis_weighted is [%e, %e]\n", vis_weighted.x, vis_weighted.y);
					printf("kernel_value is %e\n", kernel_value);
					printf("w_grid_stack[%i] is [%e, %e]\n", grid_offset_uvw, w_grid_stack[grid_offset_uvw].x, w_grid_stack[grid_offset_uvw].y);
				}		
				
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

				if (0) //(i_chan == 0 && i_row == 0)
				{
					printf("w_grid_stack[%i] is [%e, %e]\n\n", grid_offset_uvw, w_grid_stack[grid_offset_uvw].x, w_grid_stack[grid_offset_uvw].y);
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

SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_2d<double, double2, double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_2d<float,  float2,  double, double2, double3>)
SDP_CUDA_KERNEL(sdp_cuda_nifty_gridder_gridding_2d<float,  float2,  float,  float2,  float3>)
