/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRIDDER_UTILS_H_
#define SDP_GRIDDER_UTILS_H_

/**
 * @file sdp_gridder_utils.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert all image pixel positions to coordinates.
 *
 * @param image_size Side length of image, in pixels.
 * @param theta Total image size in direction cosines.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param l Output pixel l directions. May be NULL if not needed.
 * @param m Output pixel m directions. May be NULL if not needed.
 * @param n Output pixel n directions. May be NULL if not needed.
 * @param status Error status.
 */
void sdp_gridder_image_to_lmn(
        int image_size,
        double theta,
        double shear_u,
        double shear_v,
        sdp_Mem* l,
        sdp_Mem* m,
        sdp_Mem* n,
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
        double exponent,
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

#ifdef __cplusplus
}
#endif

#endif /* include guard */
