/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRIDDER_DIRECT_H_
#define SDP_GRIDDER_DIRECT_H_

/**
 * @file sdp_gridder_direct.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup GridderDirect_struct
 * @{
 */

/**
 * @struct sdp_GridderDirect
 *
 * @brief
 * Uses discrete Fourier transformation and PSWF for subgrid (de)gridding.
 *
 * Very inefficient, but as accurate as one can be (by defintion).
 *
 * These functions are ported from the implementation provided
 * in Peter Wortmann's subgrid_imaging Jupyter notebook.
 */
struct sdp_GridderDirect;

/** @} */ /* End group GridderDirect_struct. */

typedef struct sdp_GridderDirect sdp_GridderDirect;

/**
 * @defgroup GridderDirect_func
 * @{
 */

/**
 * @brief Create plan for DFT (de)gridder.
 *
 * @param image_size Total image size in pixels.
 * @param subgrid_size Sub-grid size in pixels.
 * @param theta Total image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param support Kernel support size in (u, v).
 * @param status Error status.
 *
 * @return Handle to gridder plan.
 */
sdp_GridderDirect* sdp_gridder_direct_create(
        int image_size,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        sdp_Error* status
);

/**
 * @brief Degrid visibilities using direct Fourier transformation.
 *
 * This is painfully slow, but as good as we can make it by definition.
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
 * @param uvws ``float[uvw_count, 3]`` UVW coordinates of vibilities (in m).
 * @param start_chs ``int[uvw_count]`` First channel to degrid for every uvw.
 * @param end_chs ``int[uvw_count]`` Channel at which to stop degridding for every uvw.
 * @param vis ``complex[uvw_count, ch_count]`` Output degridded visibilities.
 * @param status Error status.
 */
void sdp_gridder_direct_degrid(
        sdp_GridderDirect* plan,
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
void sdp_gridder_direct_degrid_correct(
        sdp_GridderDirect* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        sdp_Error* status
);

/**
 * @brief Grid visibilities using direct Fourier transformation.
 *
 * This is painfully slow, but as good as we can make it by definition.
 *
 * The caller must ensure the output subgrid image is sized correctly.
 *
 * @param plan Handle to gridder plan.
 * @param vis ``complex[uvw_count, ch_count]`` Input visibilities.
 * @param uvws ``float[uvw_count, 3]`` UVW coordinates of vibilities (in m).
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
void sdp_gridder_direct_grid(
        sdp_GridderDirect* plan,
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
void sdp_gridder_direct_grid_correct(
        sdp_GridderDirect* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        sdp_Error* status
);

/**
 * @brief Destroy plan for DFT (de)gridder.
 *
 * @param plan Handle to gridder plan.
 */
void sdp_gridder_direct_free(sdp_GridderDirect* plan);

/**
 * @brief Accessor function to return the image size.
 *
 * @param plan Handle to gridder plan.
 * @return The configured image size.
 */
int sdp_gridder_direct_image_size(const sdp_GridderDirect* plan);

/**
 * @brief Accessor function to return the shear factor in the u-dimension.
 *
 * @param plan Handle to gridder plan.
 * @return The configured shear factor in the u-dimension.
 */
double sdp_gridder_direct_shear_u(const sdp_GridderDirect* plan);

/**
 * @brief Accessor function to return the shear factor in the v-dimension.
 *
 * @param plan Handle to gridder plan.
 * @return The configured shear factor in the v-dimension.
 */
double sdp_gridder_direct_shear_v(const sdp_GridderDirect* plan);

/**
 * @brief Accessor function to return the sub-grid size.
 *
 * @param plan Handle to gridder plan.
 * @return The configured sub-grid size.
 */
int sdp_gridder_direct_subgrid_size(const sdp_GridderDirect* plan);

/**
 * @brief Accessor function to return the kernel support size in (u, v).
 *
 * @param plan Handle to gridder plan.
 * @return The configured kernel support size in (u, v).
 */
int sdp_gridder_direct_support(const sdp_GridderDirect* plan);

/**
 * @brief Accessor function to return the padded field of view.
 *
 * @param plan Handle to gridder plan.
 * @return The configured padded field of view, in direction cosines.
 */
double sdp_gridder_direct_theta(const sdp_GridderDirect* plan);

/**
 * @brief Accessor function to return the distance between w-layers.
 *
 * @param plan Handle to gridder plan.
 * @return The configured distance between w-layers.
 */
double sdp_gridder_direct_w_step(const sdp_GridderDirect* plan);

/** @} */ /* End group GridderDirect_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
