/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_STATION_H_
#define SKA_SDP_PROC_FUNC_STATION_H_

/**
 * @file sdp_station.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare structure handle for private implementation. */
struct sdp_Station;
typedef struct sdp_Station sdp_Station;

/**
 * @defgroup station_func
 * @{
 */

/**
 * @brief Create a handle to a station.
 *
 * (Currently unused)
 *
 * @param type Enumerated element data type of memory to allocate.
 * @param location Enumerated memory location.
 * @param status Error status.
 * @return ::sdp_Station* Handle to station data.
 */
sdp_Station* sdp_station_create(
        sdp_MemType type,
        sdp_MemLocation location,
        int num_elements,
        sdp_Error* status
);

/**
 * @brief
 * Frees memory held by the data structure.
 *
 * (Currently unused)
 *
 * This function frees memory held by the data structure.
 *
 * @param model Pointer to data structure to free.
 */
void sdp_station_free(sdp_Station* model);

/** @} */ /* End group station_func. */

/**
 * @defgroup station_aa_func
 * @{
 */

/**
 * @brief
 * Evaluates a station beam from an aperture array.
 *
 * @details
 * This function evaluates an aperture-array-based station beam for a
 * given set of antenna/element coordinates, at a set of source positions.

 * Element beam data can be supplied if required via the @p element_beam
 * parameter, and may be either complex scalar or fully polarised.
 * If provided, this must be a complex matrix containing the element response
 * for each element type and each direction on the sky.
 * The matrix dimemsions of @p element_beam are therefore the number of
 * element responses and the number of point directions, with the point
 * directions the fastest-varying dimension.
 * If there are fewer element types than there are elements, then
 * the @p element_beam_index array should be used to specify which vector
 * of the element response matrix is used for each element.
 * If polarised element data is supplied, the output station beam is also
 * polarised, and the fastest-varying dimension or dimensions of both must be
 * either of size 4 or (2, 2).
 *
 * Example dimensions allowed for @p element_beam:
 *
 * - (`num_elements`, @p num_points ) for complex scalar responses.
 * - (`num_elements`, @p num_points , 1) for complex scalar responses.
 * - (`num_elements`, @p num_points , 4) for complex polarised responses.
 * - (`num_elements`, @p num_points , 2, 2) for complex polarised responses.
 *
 * The @p station_beam array must be consistent in shape, but without
 * the element dimension.
 *
 * Example dimensions allowed for @p station_beam:
 *
 * - ( @p num_points ) for complex scalar responses.
 * - ( @p num_points , 1) for complex scalar responses.
 * - ( @p num_points , 4) for complex polarised responses.
 * - ( @p num_points , 2, 2) for complex polarised responses.
 *
 * @param wavenumber Wavenumber for the current frequency channel.
 * @param element_weights Complex array of element beamforming weights.
 * @param element_x Element x coordinates, in metres.
 * @param element_y Element y coordinates, in metres.
 * @param element_z Element z coordinates, in metres.
 * @param index_offset_points Start offset in source arrays.
 * @param num_points Number of source coordinates to use.
 * @param point_x Source x direction cosines.
 * @param point_y Source y direction cosines.
 * @param point_z Source z direction cosines.
 * @param element_beam_index Pointer to element beam indices.
 *     May be empty array.
 * @param element_beam Pointer to element beam matrix. May be empty array.
 * @param index_offset_station_beam Start offset in output array.
 * @param station_beam Output complex station beam array.
 * @param normalise If true, normalise output by dividing by the number
 *                  of elements.
 * @param eval_x If true, evaluate polarised beam using X antennas.
 * @param eval_y If true, evaluate polarised beam using Y antennas.
 * @param status Error status.
*/
void sdp_station_beam_aperture_array(
        const double wavenumber,
        const sdp_Mem* element_weights,
        const sdp_Mem* element_x,
        const sdp_Mem* element_y,
        const sdp_Mem* element_z,
        const int index_offset_points,
        const int num_points,
        const sdp_Mem* point_x,
        const sdp_Mem* point_y,
        const sdp_Mem* point_z,
        const sdp_Mem* element_beam_index,
        const sdp_Mem* element_beam,
        const int index_offset_station_beam,
        sdp_Mem* station_beam,
        const int normalise,
        const int eval_x,
        const int eval_y,
        sdp_Error* status
);

/** @} */ /* End group station_aa_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
