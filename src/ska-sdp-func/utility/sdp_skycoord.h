
/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_SKYCOORD_H_
#define SKA_SDP_PROC_FUNC_SKYCOORD_H_

/**
 * @file sdp_mem.h
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
 * @brief Creates a data structure that describes sky coordinates.
 *
 * Coordinates are described by coordinate type as string, epoch and three 
 * coordinate values. Context for the coordinate values is given by the 
 * coordinate type.
 *
 * @param type string declering coordinate type.
 * @param epoch epoch.
 * @param C0 value of the first coordinate.
 * @param C1 value of the second coordinate.
 * @param C2 value of the third coordinate.
 * @param status Error status.
 * @return ::sdp_SkyCoord* Handle to sky coordinate structure.
 */
sdp_SkyCoord* sdp_sky_coord_create(
    const char* type,
    double epoch,
    double C0,
    double C1,
    double C2,
    sdp_Error* status
);


/**
 * @brief Releases memory allocated to the sdp_SkyCoord handle.
 *
 * Releases memory held by sdp_SkyCoord handle.
 *
 * @param sky_coordinates Handle to memory block.
 */
void sdp_sky_coord_free(sdp_SkyCoord *sky_coordinates);


/**
 * @brief Returns coordinates type.
 *
 * @param sky_coordinates Handle to memory block.
 * @return pointer to char with coordinate type.
 */
const char* sdp_sky_coord_type(const sdp_SkyCoord *sky_coordinates);

/**
 * @brief Returns epoch value.
 *
 * @param sky_coordinates Handle to memory block.
 * @return value of epoch.
 */
double sdp_sky_coord_epoch(const sdp_SkyCoord *sky_coordinates);


/**
 * @brief Returns value of the selected coordinate.
 *
 * @param sky_coordinates Handle to memory block.
 * @param coordinate coordinate index (starting 0; max 2).
 * @return value of chosen coordinate.
 */
double sdp_sky_coord_coordinate(const sdp_SkyCoord *sky_coordinates, int coordinate);


/**
 * @brief Returns value of C0 coordinate.
 *
 * @param sky_coordinates Handle to memory block.
 * @return value of chosen coordinate.
 */
double sdp_sky_coord_c0(const sdp_SkyCoord *sky_coordinates);


/**
 * @brief Returns value of C1 coordinate.
 *
 * @param sky_coordinates Handle to memory block.
 * @return value of chosen coordinate.
 */
double sdp_sky_coord_c1(const sdp_SkyCoord *sky_coordinates);


/**
 * @brief Returns value of C2 coordinate.
 *
 * @param sky_coordinates Handle to memory block.
 * @return value of chosen coordinate.
 */
double sdp_sky_coord_c2(const sdp_SkyCoord *sky_coordinates);


/** @} */ /* End group SkyCoord_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
