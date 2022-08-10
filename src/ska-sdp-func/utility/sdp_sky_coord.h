
/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_SKY_COORD_H_
#define SKA_SDP_PROC_FUNC_SKY_COORD_H_

/**
 * @file sdp_sky_coord.h
 */

#include <stdint.h>
#include "ska-sdp-func/utility/sdp_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup SkyCoord_struct
 * @{
 */

struct sdp_SkyCoord;

/** @} */ /* End group SkyCoord_struct. */

/* Typedefs. */
typedef struct sdp_SkyCoord sdp_SkyCoord;

/**
 * @defgroup SkyCoord_func
 * @{
 */

/**
 * @brief Creates a data structure to encapsulate sky coordinates.
 *
 * Sky coordinates are fully described by a coordinate type string, an epoch,
 * and up to three coordinate values (one for each spatial dimension).
 *
 * Context for the coordinate values is given by the coordinate type
 * (values for which are still to be defined).
 *
 * The default epoch value is 2000.0, but can be set using
 * ::sdp_sky_coord_set_epoch().
 *
 * @param type String describing coordinate type.
 * @param coord0 Value of the first coordinate.
 * @param coord1 Value of the second coordinate.
 * @param coord2 Value of the third coordinate.
 * @return ::sdp_SkyCoord* Handle to sky coordinate structure.
 */
sdp_SkyCoord* sdp_sky_coord_create(
    const char* type,
    double coord0,
    double coord1,
    double coord2
);

/**
 * @brief Releases memory held by the sdp_SkyCoord handle.
 *
 * @param sky_coord Handle to sky coordinate.
 */
void sdp_sky_coord_free(sdp_SkyCoord *sky_coord);

/**
 * @brief Returns the value of the coordinate epoch.
 *
 * @param sky_coord Handle to sky coordinate.
 * @return Value of the coordinate epoch.
 */
double sdp_sky_coord_epoch(const sdp_SkyCoord *sky_coord);

/**
 * @brief Sets the coordinate epoch value.
 *
 * @param sky_coord Handle to sky coordinate.
 * @param epoch Value of coordinate epoch.
 */
void sdp_sky_coord_set_epoch(sdp_SkyCoord *sky_coord, double epoch);

/**
 * @brief Returns the coordinate type string.
 *
 * @param sky_coord Handle to sky coordinate.
 * @return Pointer to string describing coordinate type.
 */
const char* sdp_sky_coord_type(const sdp_SkyCoord *sky_coord);

/**
 * @brief Returns the value of the specified coordinate.
 *
 * @param sky_coord Handle to sky coordinate.
 * @param dim Coordinate dimension index (starting 0; max 2).
 * @return Value of specified coordinate.
 */
double sdp_sky_coord_value(const sdp_SkyCoord *sky_coord, int32_t dim);

/** @} */ /* End group SkyCoord_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
