/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_DEGRIDDING_H_
#define SKA_SDP_PROC_DEGRIDDING_H_

/**
 * @file sdp_degridding.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Degrid visibilities. 
 * 
 * @param grid input grid data 
 * @param x_size Number of samples on X axis
 * @param y_size Number of samples on y axis
 * @param z_size Number of samples on z axis
 * @param u0 Index of first coordinate on U axis
 * @param v0 Index of first coordinate on V axis
 * @param w0 Index of first coordinate on W axis
 * @param theta Conversion parameter from uv coordinates to xy coordinates x=u*theta
 * @param wstep, Conversion parameter from w coordinates to z coordinates z=w*wstep 
 * @param uv_kernel U,V plane kernel
 * @param uv_kernel_stride U,V plane kernel padding
 * @param uv_kernel_oversampling U,V plane kernel oversampling
 * @param w_kernel W plane Kernel
 * @param w_kernel_stride W plane kernel padding
 * @param w_kernel_oversampling W plane kernel oversampling
 * @param conjugate  Whether to generate conjugated visibilities
 * @param vis_out Output Visabilities
 * @param status Error status.
 */
void sdp_degridding(
        const sdp_Mem* grid,
        const int64_t x_size,
        const int64_t y_size,
        const int64_t z_size,
        const int64_t u0,
        const int64_t v0, 
        const int64_t w0,
        const int64_t theta,
        const int64_t wstep, 
        const sdp_Mem* uv_kernel,
        const int64_t uv_kernal_stride,
        const int64_t uv_kernal_oversampling,
        const sdp_Mem* w_kernel,
        const int64_t w_kernal_stride,
        const int64_t w_kernal_oversampling,
        const bool conjugate, 
        sdp_Mem* vis_out,
        sdp_Error* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */



// /**
// * Interface from current tensor based IO Test code.
// *
// * @param u0  Index of first coordinate on u axis
// * @param v0  Index of first coordinate on v axis
// * @param w0  Index of first coordinate on w axis
// * @param ch0  Index of first channel
// * @param conjugate  Whether to generate conjugated visibilities
// * @param uvw_ld  `double[step][ld=6:1]` Uvw coordinates per step, in `wavelengths + delta form`
// * @param channels  `int[step][chs=2:1]` Channel ranges, given as pairs `(start, end)` per step.
// * @param grid  `complex<double>[w][v][u]` Input grid data
// * @param vis_out  `complex<double>[step][ch]` Output visibilities
// */
// void degrid(const int64_t u0, const int64_t v0, const int64_t w0, const int64_t ch0, const bool conjugate,
//            const tensor_ptr<double> &uvw_ld,
//            const tensor_ptr<int64_t> &channels,
//            const tensor_ptr<double> &grid, int64_t grid_w_roll,
//            const tensor_ptr<double> &vis_out);
           



// /**
//  * interface from previous non-tensor based IO Test code
//  *
//  * @param u0  Index of first coordinate on u axis
//  * @param u_size  Number of samples along u axis
//  * @param v0  Index of first coordinate on v axis
//  * @param v_size  Number of samples along v axis
//  * @param w0  Index of first coordinate on w axis
//  * @param w_size  Number of samples along w axis
//  * @param step0  Index of first step
//  * @param step_count  Number of steps
//  * @param ch0  Index of first channel
//  * @param ch_count  Number of channels
//  * @param plan  Planning data
//  * @param conjugate  Whether to generate conjugated visibilities
//  * @param uvw_ld  `double[step][6]` Uvw coordinates per step, in `wavelengths + delta form`
//  * @param channels  `int64[step][2]` Channel ranges, given as pairs `(start, end)` per step.
//  *   See @verbatim embed:rst:inline :cpp:func:`ska_sdp_func::clamp_channels()`. @endverbatim
//  * @param grid  `double complex[w][v][u]` Grid data
//  * @param grid_u_stride  Stride of `grid` for u axis
//  * @param grid_v_stride  Stride of `grid` for v axis
//  * @param grid_w_stride  Stride of `grid` for w axis
//  * @param grid_w_roll  Roll of `grid` along w axis. I.e. use `(w - w0 + w_roll) % w_size`
//  *   to calculate positions in array.
//  * @param vis_out `double complex[step][ch]` Buffer for visibility outputs
//  * @param vis_ch_stride  Stride of `vis` for channel axis
//  * @param vis_step_stride  Stride of `vis` for step axis
//  */
// void uvw_grid_degrid(const int64_t u0, const int64_t u_size,
//                      const int64_t v0, const int64_t v_size,
//                      const int64_t w0, const int64_t w_size,
//                      const int64_t _step0, const int64_t step_count,
//                      const int64_t ch0, const int64_t ch_count,
//                      // Arrays, w/ strides
//                      uvw_gridder_plan const*const plan, const bool conjugate,
//                      double const*const uvw_ld, int64_t const*const channels,
//                      complex<double> const*const grid,
//                      const int64_t grid_u_stride, const int64_t grid_v_stride, const int64_t grid_w_stride,
//                      const int64_t grid_w_roll,
//                      complex<double> *const vis_out, const int64_t vis_ch_stride, const int64_t vis_step_stride
// );



// function call from Karels code.

// void custom_degrid(
// 		struct double2 *vis_values,
// 		struct double2 *grid,
// 		int grid_z,
// 		int grid_y,
// 		int grid_x,
// 		double *kernel_data,
// 		int c_kernel_stride,
// 		int c_kernel_oversampling,
// 		double *wkernel_data,
// 		int c_wkernel_stride,
// 		int c_wkernel_oversampling,
// 		double *u_vis_coordinates,
// 		double *v_vis_coordinates,
// 		double *w_vis_coordinates,
// 		int c_vis_count,
// 		double theta,
// 		double wstep,
// 		bool conjugate
// 	) {