/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRIDDER_UTILS_H_
#define SDP_GRIDDER_UTILS_H_

/**
 * @file sdp_gridder_utils.h
 */

#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup gridder_util_func
 * @{
 */

/**
 * @brief Multiply every element in an array by those in another array.
 *
 * Elements in the second array are raised to the given exponent,
 * and the results are accumulated into the output, so that:
 * out += in1 * in2 ** exponent
 *
 * Exponent values of 0 and 1 are handled separately, so pow() is not called
 * unnecessarily.
 * If the exponent is zero, in2 is ignored, and the output is given by:
 * out += in1
 *
 * @param out Output array.
 * @param in1 First input array.
 * @param in2 Second input array. May be NULL if not required.
 * @param exponent Exponent to use for second input array.
 * @param status Error status.
 */
void sdp_gridder_accumulate_scaled_arrays(
        sdp_Mem* out,
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        int exponent,
        sdp_Error* status
);

/**
 * @brief Count non-zero pixels in an image.
 *
 * @param image Image to analyse.
 * @param status Error status.
 */
int64_t sdp_gridder_count_nonzero_pixels(
        const sdp_Mem* image,
        sdp_Error* status
);

/**
 * @brief Determine a value for the w_step parameter.
 *
 * @param theta Size of padded field of view, in direction cosines.
 * @param fov Size of imaged field of view, in direction cosines.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param x0 If not zero, scaling factor for fov_n; if zero, this
 *           will be calculated as fov / theta.
 */
double sdp_gridder_determine_w_step(
        double theta,
        double fov,
        double shear_u,
        double shear_v,
        double x0
);

/**
 * @brief Perform prediction of visibilities via direct Fourier transform.
 *
 * Used primarily to generate reference data for testing.
 *
 * If @p start_chs and @p end_chs are NULL, all visibilities will be generated.
 *
 * @param uvws ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
 * @param start_chs ``int[uvw_count]`` First channel to degrid for every uvw.
 * @param end_chs ``int[uvw_count]`` Channel at which to stop degridding for every uvw.
 * @param flux ``float[source_count]`` Flux value of each source component.
 * @param lmn ``float[source_count, 3]`` Direction cosine of each source component.
 * @param subgrid_offset_u Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_v Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_w Offset of subgrid centre relative to grid centre.
 * @param theta Total image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param vis ``complex[uvw_count, ch_count]`` Output visibilities.
 * @param status Error status.
 */
void sdp_gridder_dft(
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* flux,
        const sdp_Mem* lmn,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double theta,
        double w_step,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* vis,
        sdp_Error* status
);

/**
 * @brief Perform imaging of visibilities via direct Fourier transform.
 *
 * Used primarily to generate reference data for testing.
 *
 * If @p start_chs and @p end_chs are NULL, all visibilities will be used.
 *
 * @param uvws ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
 * @param vis ``complex[uvw_count, ch_count]`` Input visibilities.
 * @param start_chs ``int[uvw_count]`` First channel to degrid for every uvw.
 * @param end_chs ``int[uvw_count]`` Channel at which to stop degridding for every uvw.
 * @param lmn ``float[source_count, 3]`` Direction cosine of each source component.
 * @param image_taper_1d Optional separable taper to apply across
 *                       both dimensions of the image. May be NULL.
 * @param subgrid_offset_u Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_v Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_w Offset of subgrid centre relative to grid centre.
 * @param theta Total image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param image Output image.
 * @param status Error status.
 */
void sdp_gridder_idft(
        const sdp_Mem* uvws,
        const sdp_Mem* vis,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* lmn,
        const sdp_Mem* image_taper_1d,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double theta,
        double w_step,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* image,
        sdp_Error* status
);

/**
 * @brief Convert image pixels to coordinates and optionally, fluxes.
 *
 * If the @p flux parameter is not NULL, then only the non-zero pixel values
 * and corresponding coordinates will be returned.
 * Both the output arrays must be created with the correct size before this
 * function is called, which can be done by first
 * calling ::sdp_gridder_count_nonzero_pixels().
 *
 * If the @p flux parameter is NULL, then the output pixel coordinates will be
 * for the whole image, and the @p lmn array must be sized appropriately.
 *
 * @param image Image to convert.
 * @param theta Field of view in direction cosines.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param image_taper_1d Optional separable taper to apply across
 *                       both dimensions of the image. May be NULL.
 * @param flux Optional output pixel brightness values.
 *             If not NULL, must be pre-sized.
 * @param lmn Output pixel (l, m, n)-coordinates. Must be pre-sized.
 * @param status Error status.
 */
void sdp_gridder_image_to_flmn(
        const sdp_Mem* image,
        double theta,
        double shear_u,
        double shear_v,
        const sdp_Mem* image_taper_1d,
        sdp_Mem* flux,
        sdp_Mem* lmn,
        sdp_Error* status
);

/**
 * @brief Convert image-space window function to oversampled kernel.
 *
 * This uses a DFT to do the transformation to Fourier space.
 * The supplied input window function is 1-dimensional, with a shape
 * of (support), and the output kernel is 2-dimensional, with a shape
 * of (oversample + 1, support), and it must be sized appropriately on entry.
 *
 * @param window Image-space window function in 1D.
 * @param kernel Oversampled output kernel in Fourier space.
 * @param status Error status.
 */
void sdp_gridder_make_kernel(
        const sdp_Mem* window,
        sdp_Mem* kernel,
        sdp_Error* status
);

/**
 * @brief Generate an oversampled kernel using PSWF window function.
 *
 * This uses a DFT to do the transformation to Fourier space.
 * The output kernel is 2-dimensional, with a shape
 * of (oversample + 1, vr_size), and it must be sized appropriately on entry.
 *
 * @param support Support size for kernel, usually the same as vr_size.
 * @param kernel Oversampled output kernel in Fourier space.
 * @param status Error status.
 */
void sdp_gridder_make_pswf_kernel(
        int support,
        sdp_Mem* kernel,
        sdp_Error* status
);

/**
 * @brief Generate w-pattern.
 *
 * This is the iDFT of a single visibility at (0, 0, w).
 *
 * @param subgrid_size Subgrid size.
 * @param theta Total image size in direction cosines.
 * @param shear_u Shear parameter in u.
 * @param shear_v Shear parameter in v.
 * @param w_step Distance between w-planes.
 * @param w_pattern Complex w-pattern, dimensions (subgrid_size, subgrid_size).
 * @param status Error status.
 */
void sdp_gridder_make_w_pattern(
        int subgrid_size,
        double theta,
        double shear_u,
        double shear_v,
        double w_step,
        sdp_Mem* w_pattern,
        sdp_Error* status
);

/**
 * @brief Calculates the residual difference between two 2D arrays: (a - b).
 *
 * The arrays must be 2D and have the same shape.
 *
 * @param a The first input array.
 * @param b The second input array.
 * @param out The output array.
 * @param status Error status.
 */
void sdp_gridder_residual(
        const sdp_Mem* a,
        const sdp_Mem* b,
        sdp_Mem* out,
        sdp_Error* status
);

/**
 * @brief Returns the RMS of the difference between two 2D arrays: rms(a - b).
 *
 * The two arrays must be 2D and have the same shape.
 *
 * @param a The first input array.
 * @param b The second input array.
 * @param status Error status.
 */
double sdp_gridder_rms_diff(
        const sdp_Mem* a,
        const sdp_Mem* b,
        sdp_Error* status
);

/**
 * @brief Divides every element in an array by those in another array.
 *
 * Elements in the second array are raised to the given exponent,
 * so the result is out = in1 / in2 ** exponent
 *
 * An exponent of 1 is handled separately, so pow() is not called unnecessarily.
 *
 * @param out Output array.
 * @param in1 First input array (numerator).
 * @param in2 Second input array (denominator).
 * @param exponent Exponent to use for second input array.
 * @param status Error status.
 */
void sdp_gridder_scale_inv_array(
        sdp_Mem* out,
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        int exponent,
        sdp_Error* status
);

/**
 * @brief Shift the contents of the sub-grid stack up by one plane.
 *
 * @param subgrids Sub-grid stack.
 * @param status Error status.
 */
void sdp_gridder_shift_subgrids(sdp_Mem* subgrids, sdp_Error* status);

/**
 * @brief Add the supplied sub-grid to the grid.
 *
 * @param grid Output grid.
 * @param offset_u Offset in u.
 * @param offset_v Offset in v.
 * @param subgrid Input sub-grid.
 * @param factor Factor by which to multiply elements of sub-grid before adding.
 * @param status Error status.
 */
void sdp_gridder_subgrid_add(
        sdp_Mem* grid,
        int offset_u,
        int offset_v,
        const sdp_Mem* subgrid,
        double factor,
        sdp_Error* status
);

/**
 * @brief Cut out a sub-grid from the supplied grid.
 *
 * @param grid Input grid.
 * @param offset_u Offset in u.
 * @param offset_v Offset in v.
 * @param subgrid Output sub-grid.
 * @param status Error status.
 */
void sdp_gridder_subgrid_cut_out(
        const sdp_Mem* grid,
        int offset_u,
        int offset_v,
        sdp_Mem* subgrid,
        sdp_Error* status
);

/**
 * @brief Determine sum of element-wise difference: result = sum(a - b).
 *
 * @param a First input array.
 * @param b Second input array.
 * @param result Sum of a - b.
 * @param status Error status.
 */
void sdp_gridder_sum_diff(
        const sdp_Mem* a,
        const sdp_Mem* b,
        int64_t* result,
        sdp_Error* status
);

/**
 * @brief Determine (scaled) min and max values in uvw coordinates.
 *
 * @param uvws ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param start_chs ``int[uvw_count]`` First channel to degrid for every uvw.
 * @param end_chs ``int[uvw_count]`` Channel at which to stop degridding for every uvw.
 * @param uvw_min Output 3-element array containing minimum (u,v,w) values.
 * @param uvw_max Output 3-element array containing maximum (u,v,w) values.
 * @param status Error status.
 */
void sdp_gridder_uvw_bounds_all(
        const sdp_Mem* uvws,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double uvw_min[3],
        double uvw_max[3],
        sdp_Error* status
);

/** @} */ /* End group gridder_util_func. */

#ifdef __cplusplus
}


// Convert (l, m) to (n) directions, allowing for shear.
template<typename T>
SDP_INLINE
T lm_to_n(const T& l, const T& m, const T& h_u, const T& h_v)
{
    // Easy case.
    if (h_u == 0 and h_v == 0)
        return sqrt(1 - l * l - m * m) - 1;

    // Sheared case.
    const T hul_hvm_1 = h_u * l + h_v * m - 1; // = -1 with h_u = h_v = 0
    const T hu2_hv2_1 = h_u * h_u + h_v * h_v + 1; // = 1 with h_u = h_v = 0
    return (
        sqrt(hul_hvm_1 * hul_hvm_1 - hu2_hv2_1 * (l * l + m * m)) +
        hul_hvm_1
    ) / hu2_hv2_1;
}

#endif

#endif /* include guard */
