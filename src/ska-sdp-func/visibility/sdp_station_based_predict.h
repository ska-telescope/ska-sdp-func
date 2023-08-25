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
 *   - [ num_sources, 3 ]
 *
 * - @p source_directions is 2D and real-valued, with shape:
 *   - [ num_sources, 3 ]
 *
 * - @p source_stoke_parameters is 2D and real-valued, with shape:
 *   - [ num_sources, 4 ]
 *
 * - @p jones_matrices is 2D and complex-valued, with shape:
 *   - [ scalar_jones_values_x, scalar_jones_values_y ]
 *
 * - @p jones_matrics_workspace is an empty 2D array, with shape:
 *   - [ scalar_jones_values_x, scalar_jones_values_y ]
 *
 * - @p brightness_matrix_predict is 1D and complex valued, with shape:
 *   - [ brightness_I ]
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
        const int64_t num_stations,
        const sdp_Mem* station_coordinates,
        const sdp_Mem* source_directions,
        const sdp_Mem* source_stoke_parameters,
        int wavenumber,
        sdp_Mem* visibilites,
        const sdp_Mem* jones_matrices,
        sdp_Mem* brightness_matrix_predict,
        sdp_Mem* jones_matrices_workspace,
        sdp_Error* status
);

#ifdef __cplusplus
}
#endif

#endif
