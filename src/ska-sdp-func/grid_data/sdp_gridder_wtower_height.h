/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRIDDER_WTOWER_HEIGHT_H_
#define SDP_GRIDDER_WTOWER_HEIGHT_H_

/**
 * @file sdp_gridder_wtower_height.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Find the maximum w-tower height for a given configuration.
 *
 * This function is ported from the implementation provided
 * in Peter Wortmann's subgrid_imaging Jupyter notebook.
 *
 * Note that the supplied image size should not usually be the same as the
 * final image - a smaller representative image should be sufficient
 * (twice the sub-grid size should be acceptable).
 *
 * @param image_size Total image size in pixels.
 * @param subgrid_size Sub-grid size in pixels.
 * @param theta Total (padded) image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param support Kernel support size in (u, v).
 * @param oversampling Oversampling factor for uv-kernel.
 * @param w_support Support size in w.
 * @param w_oversampling Oversampling factor for w-kernel.
 * @param fov Total image size in direction cosines.
 * @param subgrid_frac Fraction of subgrid size that should be used.
 * @param num_samples Number of samples in u and v (if 0, defaults to 3).
 * @param target_err Optional target error (if 0, will be evaluated internally).
 * @param status Error status.
 */
double sdp_gridder_find_max_w_tower_height(
        int image_size,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int oversampling,
        int w_support,
        int w_oversampling,
        double fov,
        double subgrid_frac,
        int num_samples,
        double target_err,
        sdp_Error* status
);

/**
 * @brief Generate a worst-case image to evaluate gridder accuracy.
 *
 * The supplied image must be square, and of complex type.
 *
 * @param theta Total (padded) image size in direction cosines.
 * @param fov Total image size in direction cosines.
 * @param image Image to fill.
 * @param status Error status.
 */
void sdp_gridder_worst_case_image(
        double theta,
        double fov,
        sdp_Mem* image,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
