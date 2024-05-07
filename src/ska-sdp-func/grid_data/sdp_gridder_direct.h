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
 */
sdp_GridderDirect* sdp_gridder_direct_create(
        int image_size,
        int subgrid_size,
        double theta,
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
 * @param subgrid_image Fourier transformed subgrid to degrid from.
 *     Note that the subgrid could especially span the entire grid,
 *     in which case this could simply be the entire (corrected) image.
 * @param subgrid_offset_u Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_v Offset of subgrid centre relative to grid centre.
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param uvw ``float[uvw_count, 3]`` UVW coordinates of vibilities (in m).
 * @param start_chs ``int[uvw_count]`` First channel to degrid for every uvw.
 * @param end_chs ``int[uvw_count]`` Channel at which to stop degridding for every uvw.
 * @param vis ``complex[uvw_count, ch_count]`` Output degridded visibilities.
 */
void sdp_gridder_direct_degrid(
        sdp_GridderDirect* plan,
        const sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvw,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        sdp_Mem* vis,
        sdp_Error* status
);

/**
 * @brief Do degrid correction to enable degridding from the FT of the image.
 *
 * @param facet ``complex[facet_size, facet_size]`` Facet.
 * @param facet_offset_l Offset of facet centre in l, relative to image centre.
 * @param facet_offset_m Offset of facet centre in m, relative to image centre.
 * @param status Error status.
 */
void sdp_gridder_direct_degrid_correct(
        sdp_GridderDirect* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
);

/**
 * @brief Grid visibilities using direct Fourier transformation.
 *
 * This is painfully slow, but as good as we can make it by definition.
 *
 * The caller must ensure the output subgrid image is sized correctly.
 *
 * @param vis ``complex[uvw_count, ch_count]`` Input visibilities.
 * @param uvw ``float[uvw_count, 3]`` UVW coordinates of vibilities (in m).
 * @param start_chs ``int[uvw_count]`` First channel to degrid for every uvw.
 * @param end_chs ``int[uvw_count]`` Channel at which to stop degridding for every uvw.
 * @param freq0_hz Frequency of first channel (Hz).
 * @param dfreq_hz Channel separation (Hz).
 * @param subgrid_image Fourier transformed subgrid to be gridded to.
 *     Note that the subgrid could especially span the entire grid,
 *     in which case this could simply be the entire (corrected) image.
 * @param subgrid_offset_u Offset of subgrid centre relative to grid centre.
 * @param subgrid_offset_v Offset of subgrid centre relative to grid centre.
 */
void sdp_gridder_direct_grid(
        sdp_GridderDirect* plan,
        const sdp_Mem* vis,
        const sdp_Mem* uvw,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        sdp_Error* status
);

/**
 * @brief Do grid correction after gridding.
 *
 * @param facet ``complex[facet_size, facet_size]`` Facet.
 * @param facet_offset_l Offset of facet centre in l, relative to image centre.
 * @param facet_offset_m Offset of facet centre in m, relative to image centre.
 * @param status Error status.
 */
void sdp_gridder_direct_grid_correct(
        sdp_GridderDirect* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
);

/**
 * @brief Destroy plan for DFT (de)gridder.
 */
void sdp_gridder_direct_free(sdp_GridderDirect* plan);

/** @} */ /* End group GridderDirect_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
