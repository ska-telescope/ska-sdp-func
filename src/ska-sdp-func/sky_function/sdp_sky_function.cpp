/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/sky_function/sdp_sky_function.h"
#include "ska-sdp-func/utility/sdp_logging.h"

void sdp_sky_coordinate_test(
    const sdp_SkyCoord* sky_coordinates,
    sdp_Error* status
) {
    if(*status!=SDP_SUCCESS) return;
    const char *type = sdp_sky_coord_type(sky_coordinates);
    double epoch, coord_0, coord_1, coord_2;
    
    epoch = sdp_sky_coord_epoch(sky_coordinates);
    coord_0 = sdp_sky_coord_c0(sky_coordinates);
    coord_1 = sdp_sky_coord_c1(sky_coordinates);
    coord_2 = sdp_sky_coord_c2(sky_coordinates);
    
    printf("Coordinate type: %s;\n", type);
    printf("Epoch: %f\n", epoch);
    printf("Coordinate 0: %f\n", coord_0);
    printf("Coordinate 1: %f\n", coord_1);
    printf("Coordinate 2: %f\n", coord_2);
    
    coord_0 = sdp_sky_coord_coordinate(sky_coordinates, 0);
    coord_1 = sdp_sky_coord_coordinate(sky_coordinates, 1);
    coord_2 = sdp_sky_coord_coordinate(sky_coordinates, 2);
    
    printf("Coordinate 0: %f\n", coord_0);
    printf("Coordinate 1: %f\n", coord_1);
    printf("Coordinate 2: %f\n", coord_2);
}

