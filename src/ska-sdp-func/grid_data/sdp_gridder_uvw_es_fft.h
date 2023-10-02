/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRID_UVW_ES_FFT_H_
#define SDP_GRID_UVW_ES_FFT_H_

/**
 * @file sdp_grid_uvw_es_fft.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare structure handles for private implementation. */
struct sdp_GridderUvwEsFft;
typedef struct sdp_GridderUvwEsFft sdp_GridderUvwEsFft;

/**
 * @defgroup GridderUvwEsFft_func
 * @{
 */

/**
 * @brief Creates Gridder's plan based on the given buffers and parameters.
 *
 * @param uvw [in] The (u,v,w) coordinates.  Must be complex with shape [num_rows, 3], and either float or double.
 * @param freq_hz [in] The channel frequiencies in Hz. Must be real with shape [num_chan], and either float or double.
 * @param vis [in, out] The visibility data.  Must be complex with shape [num_rows, num_chan], and either float or double.  Its data type determines the precision used for the (de)gridding.
 * @param weight [in] Its values are used to multiply the input, must be real and same shape and precision as \b vis.
 * @param dirty_image [in, out] The input/output dirty image, *must be square*.  Must be real, and either float or double.
 * @param pixel_size_x_rad [in] Angular \a x pixel size (in radians) of the dirty image (must be the same as \b pixel_size_y_rad).
 * @param pixel_size_y_rad [in] Angular \a y pixel size (in radians) of the dirty image (must be the same as \b pixel_size_x_rad).
 * @param epsilon [in] Accuracy at which the computation should be done. Must be larger than 2e-13. If \b vis is type float, it must be larger than 1e-5.
 * @param min_abs_w [in] The minimum absolute value of the w-coords in \b uvw.
 * @param max_abs_w [in] The maximum absolute value of the w-coords in \b uvw.
 * @param do_w_stacking [in] Set true for 3D (de)gridding, false for 2D (de)gridding (treats all w-coords as zero, a faster, less-accurate option).
 * @param status [in] Error status.
 * @return sdp_GridderUvwEsFft*  Handle to Gridder's plan.
 */
sdp_GridderUvwEsFft* sdp_gridder_uvw_es_fft_create_plan(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,  // in Hz
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        const sdp_Mem* dirty_image,
        const double pixel_size_x_rad,
        const double pixel_size_y_rad,
        const double epsilon,
        const double min_abs_w,
        const double max_abs_w,
        const int do_w_stacking,
        sdp_Error* status
);

/**
 * @brief Generate a dirty image from visibility data.
 *
 * See \b sdp_gridder_uvw_es_fft_create_plan() for more details
 * on the parameters.
 *
 * @param plan  Handle to Gridder's plan.
 * @param uvw [in]
 * @param freq_hz [in]
 * @param vis [in]
 * @param weight [in]
 * @param dirty_image [out]
 * @param status Error status.
 */
void sdp_grid_uvw_es_fft(
        sdp_GridderUvwEsFft* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        sdp_Mem* dirty_image,
        sdp_Error* status
);

/**
 * @brief Generate visibility data from a dirty image.
 *
 * See \b sdp_gridder_uvw_es_fft_create_plan() for more details
 * on the parameters.
 *
 * @param plan  Handle to Gridder's plan.
 * @param uvw [in]
 * @param freq_hz [in]
 * @param vis [out]
 * @param weight [in]
 * @param dirty_image [in, out]  \b NB: Even though this is an input, it is modified in place so can't be const.
 * @param status Error status.
 */
void sdp_ifft_degrid_uvw_es(
        sdp_GridderUvwEsFft* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        sdp_Mem* vis,
        const sdp_Mem* weight,
        sdp_Mem* dirty_image,         // even though this is an input, it is modified in place so can't be constant
        sdp_Error* status
);

/**
 * @brief Frees memory allocated to Gridder's plan.
 *
 * @param plan  Handle to Gridder's plan.
 */
void sdp_gridder_uvw_es_fft_free_plan(sdp_GridderUvwEsFft* plan);

/** @} */ /* End group GridderUvwEsFft_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
