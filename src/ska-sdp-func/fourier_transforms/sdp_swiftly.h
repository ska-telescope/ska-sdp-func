/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_SWIFTLY_H_
#define SKA_SDP_PROC_FUNC_SWIFTLY_H_

/**
 * @file sdp_swiftly.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sdp_SwiFTly;
typedef struct sdp_SwiFTly sdp_SwiFTly;

/**
 * @defgroup swiftly_func
 * @{
 */

/**
 * @brief Creates a plan for the SwiFTly algorithm (streaming
 * widefield Fourier transform for large-scale interferometry)
 *
 * Image, facet and subgrid sizes must be compatible. Facets and
 * subgrids must be appropriately padded for algorithm to work at any
 * level of precision, this is the responsibility of the user.
 *
 * @param image_size Size of entire (virtual) image in pixels
 * @param xM_size Internal padded subgrid size
 * @param yN_size Internal padded facet size
 * @param W Parameter for PSWF window function
 * @param status Error status.
 *
 * @return sdp_SwiFTly* Handle to SwiFTly plan.
 */
sdp_SwiFTly* sdp_swiftly_create(
    int64_t image_size,
    int64_t yN_size,
    int64_t xM_size,
    double W,
    sdp_Error* status);

/**
 * @brief Destroys the SwiFTly plan.
 *
 * @param swiftly Handle to SwiFTly plan.
 */
void sdp_swiftly_free(sdp_SwiFTly* swiftly);

/**
 * @brief Performs SwiFTly facet preparation
 *
 * This multiplies in Fb and does Fourier transformation. Common work
 * to be done for each facet before calling
 * `sdp_swiftly_extract_from_facet`.
 *
 * @param swiftly Handle to SwiFTly plan.
 * @param facet          Facet data. Size must be smaller than yN_size.
 * @param facet_data     Offset of facet mid-point relative to image mid-point
 * @param prep_facet_out Prepared facet output. Must have size yN_size.
 */
void sdp_swiftly_prepare_facet(
    sdp_SwiFTly* swiftly,
    sdp_Mem *facet,
    sdp_Mem *prep_facet_out,
    int64_t facet_offset,
    sdp_Error* status);

/** @} */ /* End group swiftly_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
