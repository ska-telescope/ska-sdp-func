/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_SKY_COORD_TEST_H_
#define SKA_SDP_PROC_FUNC_SKY_COORD_TEST_H_

/**
 * @file sdp_vector_add.h
 */

#include "ska-sdp-func/utility/sdp_skycoord.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Simple example to test sdp_skycoord.
 *
 * @param sky_coordinates sky coordinates.
 * @param status Error status.
 */
void sdp_sky_coordinate_test(
    const sdp_SkyCoord* sky_coordinates,
    sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */


