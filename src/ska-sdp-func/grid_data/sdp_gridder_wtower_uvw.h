/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRIDDER_WTOWER_UVW_H_
#define SDP_GRIDDER_WTOWER_UVW_H_

/**
 * @file sdp_gridder_wtower_uvw.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup GridderWtowerUVW_struct
 * @{
 */

/**
 * @struct sdp_GridderWtowerUVW
 *
 * @brief
 * Uses 3D w-towers implementation and PSWF for subgrid (de)gridding.
 *
 * These functions are ported from the implementation provided
 * in Peter Wortmann's subgrid_imaging Jupyter notebook.
 */
struct sdp_GridderWtowerUVW;

/** @} */ /* End group GridderWtowerUVW_struct. */

/**
 * @defgroup GridderWtowerUVW_enum
 * @{
 */

enum sdp_GridderWtowerUVWTimer
{
    SDP_WTOWER_TMR_GRID_CORRECT,
    SDP_WTOWER_TMR_FFT,
    SDP_WTOWER_TMR_KERNEL,
    SDP_WTOWER_TMR_TOTAL
};

/** @} */ /* End group GridderWtowerUVW_enum. */

/* Typedefs. */
typedef struct sdp_GridderWtowerUVW sdp_GridderWtowerUVW;
typedef enum sdp_GridderWtowerUVWTimer sdp_GridderWtowerUVWTimer;

/**
 * @defgroup GridderWtowerUVW_func
 * @{
 */

/**
 * @brief Create plan for w-towers (de)gridder.
 *
 * @param image_size Total image size in pixels.
 * @param subgrid_size Sub-grid size in pixels.
 * @param theta Total image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param support Kernel support size in (u, v).
 * @param oversampling Oversampling factor for uv-kernel.
 * @param w_support Support size in w.
 * @param w_oversampling Oversampling factor for w-kernel.
 * @param status Error status.
 */
sdp_GridderWtowerUVW* sdp_gridder_wtower_uvw_create(
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
        sdp_Error* status
);

/**
 * @brief Degrid visibilities using w-stacking/towers.
 *
 * The caller must ensure the output visibility array is sized correctly.
 *
 * @param plan Handle to gridder plan.
 * @param subgrid_image Fourier transformed subgrid to degrid from.
 *     Note that the subgrid could especially span the entire grid,
 *     in which case this could simply be the entire (corrected) image.
 * @param subgrid_offset_u Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_v Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_w Offset of subgrid centre relative to grid centre.
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param uvws ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
 * @param start_chs ``int[uvw_count]`` First channel to degrid for every uvw.
 * @param end_chs ``int[uvw_count]`` Channel at which to stop degridding for every uvw.
 * @param vis ``complex[uvw_count, ch_count]`` Output degridded visibilities.
 * @param status Error status.
 */
void sdp_gridder_wtower_uvw_degrid(
        sdp_GridderWtowerUVW* plan,
        const sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        sdp_Mem* vis,
        sdp_Error* status
);

/**
 * @brief Do degrid correction to enable degridding from the FT of the image.
 *
 * @param plan Handle to gridder plan.
 * @param facet ``complex[facet_size, facet_size]`` Facet.
 * @param facet_offset_l Offset of facet centre in l, relative to image centre.
 * @param facet_offset_m Offset of facet centre in m, relative to image centre.
 * @param w_offset Offset in w, to allow for w-stacking.
 * @param status Error status.
 */
void sdp_gridder_wtower_uvw_degrid_correct(
        sdp_GridderWtowerUVW* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        sdp_Error* status
);

/**
 * @brief Grid visibilities using w-stacking/towers.
 *
 * The caller must ensure the output subgrid image is sized correctly.
 *
 * @param plan Handle to gridder plan.
 * @param vis ``complex[uvw_count, ch_count]`` Input visibilities.
 * @param uvws ``float[uvw_count, 3]`` UVW coordinates of visibilities (in m).
 * @param start_chs ``int[uvw_count]`` First channel to degrid for every uvw.
 * @param end_chs ``int[uvw_count]`` Channel at which to stop degridding for every uvw.
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param subgrid_image Fourier transformed subgrid to be gridded to.
 *     Note that the subgrid could especially span the entire grid,
 *     in which case this could simply be the entire (corrected) image.
 * @param subgrid_offset_u Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_v Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_w Offset of subgrid centre relative to grid centre.
 * @param status Error status.
 */
void sdp_gridder_wtower_uvw_grid(
        sdp_GridderWtowerUVW* plan,
        const sdp_Mem* vis,
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        sdp_Error* status
);

/**
 * @brief Do grid correction after gridding.
 *
 * @param plan Handle to gridder plan.
 * @param facet ``complex[facet_size, facet_size]`` Facet.
 * @param facet_offset_l Offset of facet centre in l, relative to image centre.
 * @param facet_offset_m Offset of facet centre in m, relative to image centre.
 * @param w_offset Offset in w, to allow for w-stacking.
 * @param status Error status.
 */
void sdp_gridder_wtower_uvw_grid_correct(
        sdp_GridderWtowerUVW* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        sdp_Error* status
);

/**
 * @brief Destroy plan for w-towers (de)gridder.
 *
 * @param plan Handle to gridder plan.
 */
void sdp_gridder_wtower_uvw_free(sdp_GridderWtowerUVW* plan);

/**
 * @brief Report elapsed time taken in the specified part of the gridder.
 *
 * @param plan Handle to gridder plan.
 * @param timer Timer enumeration to return.
 * @param grid 0 for degridding time, 1 for gridding time.
 */
double sdp_gridder_wtower_uvw_elapsed_time(
        const sdp_GridderWtowerUVW* plan,
        sdp_GridderWtowerUVWTimer timer,
        int grid
);

/** @} */ /* End group GridderWtowerUVW_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
