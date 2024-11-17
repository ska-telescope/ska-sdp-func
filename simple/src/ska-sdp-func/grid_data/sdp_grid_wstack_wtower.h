/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRID_WSTACK_WTOWER_H_
#define SDP_GRID_WSTACK_WTOWER_H_

/**
 * @file sdp_grid_wstack_wtower.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup grid_wstack_wtower_func
 * @{
 */

/**
 * @brief Degrid visibilities using w-stacking with w-towers.
 *
 * @param image Image to degrid from.
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param uvw ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
 * @param subgrid_size Sub-grid size in pixels.
 * @param theta Total image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param support Kernel support size in (u, v).
 * @param oversampling Oversampling factor for uv-kernel.
 * @param w_support Support size in w.
 * @param w_oversampling Oversampling factor for w-kernel.
 * @param subgrid_frac Fraction of subgrid size that should be used.
 * @param w_tower_height Height of w-tower to use.
 * @param verbosity Verbosity level.
 * @param vis ``complex[uvw_count, ch_count]`` Output degridded visibilities.
 * @param num_threads The number of CPU threads to use. Automatic if <= 0.
 * @param status Error status.
 */
void sdp_grid_wstack_wtower_degrid_all(
        const sdp_Mem* image,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvw,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int oversampling,
        int w_support,
        int w_oversampling,
        double subgrid_frac,
        double w_tower_height,
        int verbosity,
        sdp_Mem* vis,
        int num_threads,
        sdp_Error* status
);

/**
 * @brief Grid visibilities using w-stacking with w-towers.
 *
 * @param vis ``complex[uvw_count, ch_count]`` Input visibilities.
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param uvw ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
 * @param subgrid_size Sub-grid size in pixels.
 * @param theta Total image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param support Kernel support size in (u, v).
 * @param oversampling Oversampling factor for uv-kernel.
 * @param w_support Support size in w.
 * @param w_oversampling Oversampling factor for w-kernel.
 * @param subgrid_frac Fraction of subgrid size that should be used.
 * @param w_tower_height Height of w-tower to use.
 * @param verbosity Verbosity level.
 * @param image Output image.
 * @param num_threads The number of CPU threads to use. Automatic if <= 0.
 * @param status Error status.
 */
void sdp_grid_wstack_wtower_grid_all(
        const sdp_Mem* vis,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvw,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int oversampling,
        int w_support,
        int w_oversampling,
        double subgrid_frac,
        double w_tower_height,
        int verbosity,
        sdp_Mem* image,
        int num_threads,
        sdp_Error* status
);

/** @} */ /* End group grid_wstack_wtower_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
