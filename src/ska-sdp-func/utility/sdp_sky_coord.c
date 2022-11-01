/* See the LICENSE file at the top-level directory of this distribution. */

#include <stdlib.h>
#include <string.h>

#include "ska-sdp-func/utility/sdp_sky_coord.h"

// Private implementation.
struct sdp_SkyCoord
{
    char* type; // A string to describe the coordinate type.
    double epoch;
    double coords[3];
};


sdp_SkyCoord* sdp_sky_coord_create(
        const char* type,
        double coord0,
        double coord1,
        double coord2)
{
    sdp_SkyCoord* sky_coord = (sdp_SkyCoord*) calloc(1, sizeof(sdp_SkyCoord));
    const size_t type_len = 1 + strlen(type);
    sky_coord->type = (char*) calloc(type_len, sizeof(char));
    memcpy(sky_coord->type, type, type_len);
    sky_coord->epoch = 2000.0; // Set default (this is only needed sometimes).
    sky_coord->coords[0] = coord0;
    sky_coord->coords[1] = coord1;
    sky_coord->coords[2] = coord2;
    return sky_coord;
}


void sdp_sky_coord_free(sdp_SkyCoord* sky_coord)
{
    if (!sky_coord) return;
    free(sky_coord->type);
    free(sky_coord);
}


double sdp_sky_coord_epoch(const sdp_SkyCoord* sky_coord)
{
    return sky_coord ? sky_coord->epoch : 0.0;
}


void sdp_sky_coord_set_epoch(sdp_SkyCoord* sky_coord, double epoch)
{
    if (!sky_coord) return;
    sky_coord->epoch = epoch;
}


const char* sdp_sky_coord_type(const sdp_SkyCoord* sky_coord)
{
    return sky_coord ? sky_coord->type : 0;
}


double sdp_sky_coord_value(const sdp_SkyCoord* sky_coord, int32_t dim)
{
    return (sky_coord && dim >= 0 && dim < 3) ? sky_coord->coords[dim] : 0.0;
}
