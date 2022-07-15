/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ska-sdp-func/sky_function/sdp_sky_function.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_skycoord.h"


int main() {
    sdp_SkyCoord *sky_coordinates;
    sdp_Error status = SDP_SUCCESS;
    sky_coordinates = sdp_sky_coord_create("Type01", 1.0, 0.0, 1.0, 2.0, &status);
    sdp_sky_coordinate_test(sky_coordinates, &status);
    sdp_sky_coord_free(sky_coordinates);
    
    sdp_SkyCoord *sky_coordinates_failed;
    status = SDP_SUCCESS;
    sky_coordinates_failed = sdp_sky_coord_create("BuhahBuhahBuhahBuhahBuhahBuhahBuhahBuhahBuhahBuhahBuhah", 1.0, 0.0, 1.0, 2.0, &status);
    sdp_sky_coord_free(sky_coordinates_failed);
}