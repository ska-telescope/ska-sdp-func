/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_ST_BASED_PREDICT_H_
#define SKA_SDP_PROC_FUNC_ST_BASED_PREDICT_H_

/**
 * @file sdp_station_based_predict.h
 */

/**
 * @defgroup station_based_predict_func
 * @{
 */

/**
 * @brief Calculate the predicted visiblities per station and per source using the Jones formalism
 *
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 *
 * - @p station_coordinates is 2D and real-valued, with shape:
 *   - [ x, y, z ]
 *
 * - @p source_directions is 2D and real-valued, with shape:
 *   - [ num_sources, 3 ]
 *
 * - @p source_stoke_parameters is 2D and real-valued, with shape:
 *   - [ num_sources, 4 ]
 *
 * - @p visibilities is 1D and complex-valued, with shape:
 *   - [ num_visibilites]
 *
 * @param num_stations Number of antennas in the array.
 * @param station_coordinates XYZ coordinates in metres for each station. Dimensions as above.
 * @param source_directions Source direction cosines. Dimensions as above.
 * @param source_stokes_parameters IQUV stokes paramaters for each source. Dimensions as above.
 * @param wavenumber 2 * pi * λ, where λ is the channel frequency in metres. To scale the station coordinates from metres into wavelengths.
 * @param visbilities List of visibilities predicted by the function. 1D array.
 * @param status Error status.
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void sdp_station_based_predict(
        int64_t num_stations,
        sdp_Mem* station_coordinates,
        sdp_Mem* source_directions,
        sdp_Mem* source_stoke_parameters,
        int wavenumber,
        sdp_Mem* visibilites,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif
