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
 * @brief Queries image size associated with SwiFTly plan
 *
 * @param swiftly        SwiFTly plan.
 */
int64_t sdp_swiftly_get_image_size(sdp_SwiFTly* swiftly);

/**
 * @brief Queries facet size (yN) associated with SwiFTly plan
 *
 * @param swiftly        SwiFTly plan.
 */
int64_t sdp_swiftly_get_facet_size(sdp_SwiFTly* swiftly);

/**
 * @brief Queries subgrid size (xM) associated with SwiFTly plan
 *
 * @param swiftly        SwiFTly plan.
 */
int64_t sdp_swiftly_get_subgrid_size(sdp_SwiFTly* swiftly);

/**
 * @brief Destroys the SwiFTly plan.
 *
 * @param swiftly Handle to SwiFTly plan.
 */
void sdp_swiftly_free(sdp_SwiFTly* swiftly);

/**
 * @brief Performs SwiFTly facet preparation
 *
 * This multiplies by Fb and does Fourier transformation. Common work
 * to be done for each facet (and axis) before calling
 * sdp_swiftly_extract_from_facet()
 *
 * @param swiftly        SwiFTly plan.
 * @param facet          `[*,<yN_size]` Facet data
 * @param prep_facet_out `[*,yN_size]` Prepared facet output
 * @param facet_offset   Offset of facet mid-point relative to image mid-point
 * @param status         Error status
 */
void sdp_swiftly_prepare_facet(
    sdp_SwiFTly* swiftly,
    sdp_Mem *facet,
    sdp_Mem *prep_facet_out,
    int64_t facet_offset,
    sdp_Error* status);

/**
 * @brief Extract facet contribution to a subgrid
 *
 * Copies out all data from prepared facet that relates to a
 * subgrid at a particular offset. The returned representation is
 * optimised for representing this data in a compact way, and should
 * be used for distribution. Use sdp_swiftly_add_to_subgrid() or
 * sdp_swiftly_add_to_subgrid_2d() in order to accumulate such
 * contributions from multiple facets.
 *
 * @param swiftly        SwiFTly plan.
 * @param prep_facet     `[*,yN_size]` Prepared facet output
 * @param contribution_out `[*,xM_yN_size]` Facet contribution to subgrid
 * @param subgrid_offset Offset of subgrid mid-point relative to grid mid-point
 * @param status         Error status
 */
void sdp_swiftly_extract_from_facet(
    sdp_SwiFTly* swiftly,
    sdp_Mem *prep_facet,
    sdp_Mem *contribution_out,
    int64_t subgrid_offset,
    sdp_Error* status);

/**
 * @brief Add facet contribution to a subgrid image
 *
 * Accumulates a facet contribution in subgrid image passed. Subgrid
 * image should be filled with zeros when passed to function
 * initially. Use sdp_swiftly_finish_subgrid_inplace() or
 * sdp_swiftly_finish_subgrid_inplace_2d() to obtain subgrid
 * data.
 *
 * @param swiftly        SwiFTly plan.
 * @param contribution   `[*,xM_yN_size]` Facet contribution to subgrid
 * @param subgrid_image_inout `[*,xM_size]` Subgrid image for accumulation
 * @param facet_offset   Offset of facet mid-point relative to image mid-point
 * @param status         Error status
 */
void sdp_swiftly_add_to_subgrid(
    sdp_SwiFTly* swiftly,
    sdp_Mem *contribution,
    sdp_Mem *subgrid_image_inout,
    int64_t facet_offset,
    sdp_Error* status);


/**
 * @brief Add facet contribution to a subgrid image (both axes)
 *
 * Accumulates a facet contribution in subgrid image passed. Subgrid
 * image should be filled with zeros when passed to function
 * initially. Use sdp_swiftly_finish_subgrid_inplace() or
 * sdp_swiftly_finish_subgrid_inplace_2d() to obtain subgrid
 * data.
 *
 * This is equivalent to applying sdp_swiftly_add_to_subgrid() to a
 * contribution and subarray image, then adding their transposition.
 *
 * @param swiftly        SwiFTly plan.
 * @param contribution   `[xM_yN_size,xM_yN_size]` Facet contribution to subgrid
 * @param subgrid_image_inout `[xM_size,xM_size]` Subgrid image for accumulation
 * @param facet_offset0  Facet mid-point offset relative to image
 *                        mid-point along first axis
 * @param facet_offset1  Facet mid-point offset relative to image
 *                        mid-point along second axis
 * @param status         Error status
 */
void sdp_swiftly_add_to_subgrid_2d(
    sdp_SwiFTly* swiftly,
    sdp_Mem *contribution,
    sdp_Mem *subgrid_image_inout,
    int64_t facet_offset0,
    int64_t facet_offset1,
    sdp_Error* status);

/**
 * @brief Finish subgrid after contribution accumulation
 *
 * Performs the final Fourier Transformation to obtain the subgrid
 * from the subgrid image sum.
 *
 * @param swiftly        SwiFTly plan.
 * @param subgrid_inout  `[*,xM_size]` Subgrid / subgrid image for accumulation.
 * @param subgrid_offset Offset of subgrid mid-point relative to grid mid-point
 * @param status         Error status
 */
void sdp_swiftly_finish_subgrid_inplace(
    sdp_SwiFTly* swiftly,
    sdp_Mem *subgrid_inout,
    int64_t subgrid_offset,
    sdp_Error* status);

/**
 * @brief Finish subgrid after contribution accumulation (both axes)
 *
 * Performs the final Fourier Transformation to obtain the subgrid
 * from the subgrid image sum. This performs the transformation on
 * both axes - equivalent to applying
 * sdp_swiftly_finish_subgrid_inplace to a subarray image, then its
 * transposition.
 *
 * @param swiftly        SwiFTly plan.
 * @param subgrid_inout  `[xM_size,xM_size]` Subgrid / subgrid image for accumulation
 * @param subgrid_offset0 Subgrid mid-point offset relative to grid
 *                         mid-point along first axis
 * @param subgrid_offset1 Subgrid mid-point offset relative to grid
 *                         mid-point along second axis
 * @param status         Error status
 */
void sdp_swiftly_finish_subgrid_inplace_2d(
    sdp_SwiFTly* swiftly,
    sdp_Mem *subgrid_inout,
    int64_t subgrid_offset0,
    int64_t subgrid_offset1,
    sdp_Error* status);

void sdp_swiftly_prepare_subgrid_inplace(
    sdp_SwiFTly* swiftly,
    sdp_Mem *subgrid_inout,
    int64_t subgrid_offset,
    sdp_Error* status);

void sdp_swiftly_prepare_subgrid_inplace_2d(
    sdp_SwiFTly* swiftly,
    sdp_Mem *subgrid_inout,
    int64_t subgrid_offset0,
    int64_t subgrid_offset1,
    sdp_Error* status);

void sdp_swiftly_extract_from_subgrid(
    sdp_SwiFTly* swiftly,
    sdp_Mem *subgrid_image,
    sdp_Mem *contribution_out,
    int64_t facet_offset,
    sdp_Error* status);

void sdp_swiftly_extract_from_subgrid_2d(
    sdp_SwiFTly* swiftly,
    sdp_Mem *subgrid_image,
    sdp_Mem *contribution_out,
    int64_t facet_offset0,
    int64_t facet_offset1,
    sdp_Error* status);

void sdp_swiftly_add_to_facet(
    sdp_SwiFTly* swiftly,
    sdp_Mem *contribution,
    sdp_Mem *prep_facet_inout,
    int64_t subgrid_offset,
    sdp_Error* status);

void sdp_swiftly_finish_facet(
    sdp_SwiFTly* swiftly,
    sdp_Mem *prep_facet_inout,
    sdp_Mem *facet_out,
    int64_t facet_offset,
    sdp_Error* status);

/** @} */ /* End group swiftly_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
