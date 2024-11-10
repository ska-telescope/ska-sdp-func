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

/* Typedefs. */
typedef struct sdp_GridderWtowerUVW sdp_GridderWtowerUVW;

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
 *
 * @return Handle to gridder plan.
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
 * @param start_row Row (uvw index) at which to start processing data.
 * @param end_row Row (uvw index) at which to stop processing data (exclusive).
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
        int64_t start_row,
        int64_t end_row,
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
 * @param start_row Row (uvw index) at which to start processing data.
 * @param end_row Row (uvw index) at which to stop processing data (exclusive).
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
        int64_t start_row,
        int64_t end_row,
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
 * @brief Report total number of w-planes processed in the sub-grid stack.
 *
 * @param plan Handle to gridder plan.
 * @param gridding 0 for w-planes in degridding, 1 for w-planes in gridding.
 */
int sdp_gridder_wtower_uvw_num_w_planes(
        const sdp_GridderWtowerUVW* plan,
        int gridding
);

/**
 * @brief Accessor function to return the image size.
 *
 * @param plan Handle to gridder plan.
 * @return The configured image size.
 */
int sdp_gridder_wtower_uvw_image_size(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the (u, v)-oversampling factor.
 *
 * @param plan Handle to gridder plan.
 * @return The configured (u,v)-oversampling factor.
 */
int sdp_gridder_wtower_uvw_oversampling(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the shear factor in the u-dimension.
 *
 * @param plan Handle to gridder plan.
 * @return The configured shear factor in the u-dimension.
 */
double sdp_gridder_wtower_uvw_shear_u(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the shear factor in the v-dimension.
 *
 * @param plan Handle to gridder plan.
 * @return The configured shear factor in the v-dimension.
 */
double sdp_gridder_wtower_uvw_shear_v(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the sub-grid size.
 *
 * @param plan Handle to gridder plan.
 * @return The configured sub-grid size.
 */
int sdp_gridder_wtower_uvw_subgrid_size(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the kernel support size in (u, v).
 *
 * @param plan Handle to gridder plan.
 * @return The configured kernel support size in (u, v).
 */
int sdp_gridder_wtower_uvw_support(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the padded field of view.
 *
 * @param plan Handle to gridder plan.
 * @return The configured padded field of view, in direction cosines.
 */
double sdp_gridder_wtower_uvw_theta(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the w-oversampling factor.
 *
 * @param plan Handle to gridder plan.
 * @return The configured w-oversampling factor.
 */
int sdp_gridder_wtower_uvw_w_oversampling(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the distance between w-layers.
 *
 * @param plan Handle to gridder plan.
 * @return The configured distance between w-layers.
 */
double sdp_gridder_wtower_uvw_w_step(const sdp_GridderWtowerUVW* plan);

/**
 * @brief Accessor function to return the kernel support size in w.
 *
 * @param plan Handle to gridder plan.
 * @return The configured kernel support size in w.
 */
int sdp_gridder_wtower_uvw_w_support(const sdp_GridderWtowerUVW* plan);

/** @} */ /* End group GridderWtowerUVW_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
