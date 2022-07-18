/* See the LICENSE file at the top-level directory of this distribution. */

#include <stdlib.h>
#include <string.h>

#include "ska-sdp-func/utility/sdp_sky_coord.h"
#include "ska-sdp-func/utility/sdp_logging.h"

// Private implementation.
struct sdp_SkyCoord
{
    char *type; // string indicating coordinate type.
    double epoch; // epoch
    double c0; // Coordinate 1
    double c1; // Coordinate 2
    double c2; // Coordinate 3
};

sdp_SkyCoord* sdp_sky_coord_create(
        const char *type,
        double epoch,
        double c0,
        double c1,
        double c2,
        sdp_Error* status
) {
    if(*status!=0) return(NULL);
    sdp_SkyCoord *sky_coordinates;
    sky_coordinates = (sdp_SkyCoord*) calloc(1, sizeof(sdp_SkyCoord));
    // I think we need to copy because this should not be a wrapper
    if(strlen(type)<50) {
        sky_coordinates->type = (char *) malloc(50);
        sprintf(sky_coordinates->type,"%s",type);
    }
    else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_CRITICAL("Coordinate type is too long (maximum 50 characters)");
        return(NULL);
    }
    sky_coordinates->epoch = epoch;
    sky_coordinates->c0 = c0;
    sky_coordinates->c1 = c1;
    sky_coordinates->c2 = c2;
    
    return(sky_coordinates);
}

void sdp_sky_coord_free(sdp_SkyCoord *sky_coordinates) {
    if (!sky_coordinates) return;
    free(sky_coordinates->type);
    free(sky_coordinates);
}

// All these are awfully clumsy
const char* sdp_sky_coord_type(const sdp_SkyCoord *sky_coordinates) {
    if(sky_coordinates!=NULL) {
        return(sky_coordinates->type);
    }
    else return(NULL);
}

double sdp_sky_coord_epoch(const sdp_SkyCoord *sky_coordinates) {
    if(sky_coordinates!=NULL) {
        return(sky_coordinates->epoch);
    }
    return(0);
}

double sdp_sky_coord_coordinate(const sdp_SkyCoord *sky_coordinates, int coordinate) {
    if(sky_coordinates==NULL || coordinate<0 || coordinate>2) return(0);
    if(coordinate==0) return(sky_coordinates->c0);
    else if(coordinate==1) return(sky_coordinates->c1);
    else if(coordinate==2) return(sky_coordinates->c2);
    else return(0);
}

double sdp_sky_coord_c0(const sdp_SkyCoord *sky_coordinates) {
    if(sky_coordinates!=NULL) return(sky_coordinates->c0);
    else return(0);
}

double sdp_sky_coord_c1(const sdp_SkyCoord *sky_coordinates) {
    if(sky_coordinates!=NULL) return(sky_coordinates->c1);
    else return(0);
}

double sdp_sky_coord_c2(const sdp_SkyCoord *sky_coordinates) {
    if(sky_coordinates!=NULL) return(sky_coordinates->c2);
    else return(0);
}
