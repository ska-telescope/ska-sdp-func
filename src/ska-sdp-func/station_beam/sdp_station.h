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

/**
 * @brief
 * Evaluates a basic array factor.
 *
 * @details
 * This function evaluates a basic array factor for a given set of
 * antenna coordinates, at a set of source positions.
 *
 * Antenna data can be supplied if required via the @p data parameter.
 *
 * @param wavenumber Wavenumber for the current frequency channel.
 * @param element_weights Complex array of element beamforming weights.
 * @param element_x Element x coordinates, in metres.
 * @param element_y Element y coordinates, in metres.
 * @param element_z Element z coordinates, in metres.
 * @param idx_offset_points Start offset in source arrays.
 * @param num_points Number of source coordinates to use.
 * @param point_x Source x direction cosines.
 * @param point_y Source y direction cosines.
 * @param point_z Source z direction cosines.
 * @param data_index Pointer to indirection indices. May be empty array.
 * @param data Pointer to element response matrix. May be empty array.
 * @param idx_offset_beam Start offset in output array.
 * @param beam Output complex beam array.
 * @param normalise If true, normalise output by dividing by the number
 *                  of elements.
 * @param status Error status.
*/
void sdp_station_beam_array_factor(
        const double wavenumber,
        const sdp_Mem* element_weights,
        const sdp_Mem* element_x,
        const sdp_Mem* element_y,
        const sdp_Mem* element_z,
        int idx_offset_points,
        int num_points,
        const sdp_Mem* point_x,
        const sdp_Mem* point_y,
        const sdp_Mem* point_z,
        const sdp_Mem* data_index,
        const sdp_Mem* data,
        int idx_offset_beam,
        sdp_Mem* beam,
        int normalise,
        sdp_Error* status
);

/** @} */ /* End group station_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
