/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRIDDER_GRID_CORRECT_H_
#define SDP_GRIDDER_GRID_CORRECT_H_

/**
 * @file sdp_gridder_grid_correct.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply grid correction for the PSWF.
 *
 * @param image_size Total image size in pixels.
 * @param theta Total image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param support Kernel support size in (u, v).
 * @param w_support Support size in w.
 * @param facet Image facet (may be complex).
 * @param facet_offset_l Offset of facet centre in l, relative to image centre.
 * @param facet_offset_m Offset of facet centre in m, relative to image centre.
 * @param status Error status.
 */
void sdp_gridder_grid_correct_pswf(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int w_support,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
);

/**
 * @brief Apply w-correction to allow for w-stacking.
 *
 * @param image_size Total image size in pixels.
 * @param theta Total image size in direction cosines.
 * @param w_step Spacing between w-planes.
 * @param shear_u Shear parameter in u (use zero for no shear).
 * @param shear_v Shear parameter in v (use zero for no shear).
 * @param facet Complex image facet.
 * @param facet_offset_l Offset of facet centre in l, relative to image centre.
 * @param facet_offset_m Offset of facet centre in m, relative to image centre.
 * @param w_offset Offset in w, to allow for w-stacking.
 * @param inverse If true, apply inverse w-correction.
 * @param status Error status.
 */
void sdp_gridder_grid_correct_w_stack(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        int inverse,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
