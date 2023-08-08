#include <cmath>
#include <complex>
#include <iostream>

#include "ska-sdp-func/utility/sdp_data_model_checks.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/visibility/sdp_station_based_predict.h"

using std::complex;
using namespace std::complex_literals;

#define INDEX_2D(N2, N1, I2, I1)(N1 * I2 + I1)

struct brightness_matrix
{
    complex<double> i1;
    complex<double> i2;
    complex<double> i3;
    complex<double> i4;
};


template<typename STOKES_TYPE>
static brightness_matrix circular_pol_brightness_values(
        const int64_t i_source,
        const STOKES_TYPE stokes_parameters
)
{
    brightness_matrix bm;
    double I = stokes_parameters[i_source];
    double Q = stokes_parameters[i_source + 1];
    double U = stokes_parameters[i_source + 2];
    double V = stokes_parameters[i_source + 3];
    complex<double> iU = U * 1i;
    bm.i1 = I - V;
    bm.i2 = Q - iU;
    bm.i3 = Q + iU;
    bm.i4 = I + V;
    return bm;
}


template<typename STOKES_TYPE>
static brightness_matrix linear_pol_brightness_values(
        const int64_t i_source,
        const STOKES_TYPE stokes_parameters
)
{
    brightness_matrix bm;
    double I = stokes_parameters[i_source];
    double Q = stokes_parameters[i_source + 1];
    double U = stokes_parameters[i_source + 2];
    double V = stokes_parameters[i_source + 3];
    complex<double> iV = V * 1i;
    bm.i1 = I + Q;
    bm.i2 = U + iV;
    bm.i3 = U - iV;
    bm.i4 = I - Q;
    return bm;
}


template<typename BRIGHTNESS_TYPE, typename STOKES_TYPE>
static void create_brightness_values_for_predict(
        BRIGHTNESS_TYPE* brightness_matrix_predict,
        const STOKES_TYPE* stokes_parameters,
        const int64_t num_sources
)
{
    for (int source = 0; source < num_sources; source++)
    {
        brightness_matrix_predict[source] = linear_pol_brightness_values(source,
                stokes_parameters
        ).i1;
    }
}


template<typename DIR_TYPE, typename STATION_COORDINATES,
        typename SCALAR_JONES_TYPE>
static void scalar_create_phase_difference_Jones_values(
        int wavenumber,
        const int64_t num_sources,
        const int64_t num_stations,
        const STATION_COORDINATES* station_coordinates,
        const DIR_TYPE* source_directions,
        const SCALAR_JONES_TYPE* jones,
        SCALAR_JONES_TYPE* jones_workspace
)
{
    for (int station = 0; station < num_stations; station++)
    {
        for (int source = 0; source < num_sources; source++)
        {
            const unsigned int i_coordinate = 3 * station;
            const unsigned int i_dir = 3 * source;
            const DIR_TYPE l = source_directions[i_dir];
            const DIR_TYPE m = source_directions[i_dir + 1];
            const DIR_TYPE n = source_directions[i_dir + 2];
            const double phase = wavenumber *
                    (station_coordinates[i_coordinate] * l) +
                    (station_coordinates[i_coordinate + 1] * m) +
                    (station_coordinates[i_coordinate + 2] * (n - 1));
            const std::complex<double> phasor(cos(phase), sin(phase));
            jones_workspace[station * num_sources +
                    source] = jones[station * num_sources + source] * phasor;
            // copy the scalar twice if it's a matrix...
        }
    }
}


// template<typename DIR_TYPE, typename UVW_COORDINATES, typename VECTOR_JONES_TYPE>


// static void vector_create_phase_difference_Jones_values(


//     int wavenumber,


//     const int64_t num_sources,


//     const int64_t num_stations,


//     const UVW_COORDINATES* station_coordinates,


//     const DIR_TYPE* source_directions,


//     VECTOR_JONES_TYPE* Jones


// )


// {


//     for(int station=0; station < num_stations; station++)


//     {


//         for (int source = 0; source < num_sources; source++)


//         {


//             const unsigned int i_coordinate = 3 * station;


//             const unsigned int i_dir = 3 * source;


//             const DIR_TYPE l = source_directions[i_dir];


//             const DIR_TYPE m = source_directions[i_dir + 1];


//             const DIR_TYPE n = source_directions[i_dir + 2];


//             const double phase = wavenumber*(station_coordinates[i_coordinate]*l)+(station_coordinates[i_coordinate+1]*m)+(station_coordinates[i_coordinate+2]*(n-1));


//             const VECTOR_JONES_TYPE phasor_x(cos(phase), sin(phase));


//             const VECTOR_JONES_TYPE phasor_y = phasor_x;


//             Jones[station * num_sources + source] = phasor_x, phasor_y; //check if you can even do this...


//         }


//     }


// }


template<typename VIS_TYPE, typename BRIGHTNESS_TYPE,
        typename SCALAR_JONES_TYPE>
static void scalar_predict_visibilites(
        const int64_t num_stations,
        const int64_t num_sources,
        const SCALAR_JONES_TYPE* jones_matrix,
        const BRIGHTNESS_TYPE* brightness_matrix_predict,
        VIS_TYPE* visibility
)
{
    int ivis = 0;
    for (int p_station = 0; p_station < num_stations; p_station++)
    {
        for (int q_station = p_station + 1;
                q_station < num_stations;
                q_station++, ivis++)
        {
            complex<double> sum = 0;
            for (int source = 0; source < num_sources; source++)
            {
                sum += *brightness_matrix_predict *
                        jones_matrix[p_station * num_sources + source] *
                        std::conj(jones_matrix[q_station * num_sources +
                        source]
                        );
                // you can only do this in the scalar version of this function...otherwise the conjugate is differently calculated
            }
            visibility[ivis] = sum; // only in the scalar version would the visibilities be scalar...
        }
    }
}


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
)
{
    if (*status) return;
    sdp_MemType source_type = sdp_mem_type(source_directions);
    sdp_MemType station_type = sdp_mem_type(station_coordinates);
    sdp_MemLocation visibility_location = sdp_mem_location(visibilites);

    int64_t num_sources = sdp_mem_shape_dim(source_directions, 0);
    // complex<double> Jones = 0.0f;

    // Check memory locations for sources and visibilties
    sdp_mem_check_location(source_directions, visibility_location, status);
    sdp_mem_check_location(station_coordinates, visibility_location, status);

    sdp_mem_check_writeable(visibilites, status);

    if (visibility_location == SDP_MEM_CPU)
    {
        if (source_type == SDP_MEM_DOUBLE &&
                station_type == SDP_MEM_DOUBLE)
        {
            scalar_create_phase_difference_Jones_values
            (
                    wavenumber,
                    num_sources,
                    num_stations,
                    (const double*) sdp_mem_data_const(station_coordinates),
                    (const double*) sdp_mem_data_const(source_directions),
                    (const std::complex<double>*) sdp_mem_data_const(
                    jones_matrices
                    ),
                    (std::complex<double>*) sdp_mem_data(
                    jones_matrices_workspace
                    )
            );

            create_brightness_values_for_predict
            (
                    (std::complex<double>*) sdp_mem_data(
                    brightness_matrix_predict
                    ),
                    (const double*) sdp_mem_data_const(source_stoke_parameters),
                    num_sources
            );

            scalar_predict_visibilites
            (
                    num_stations,
                    num_sources,
                    (const std::complex<double>*) sdp_mem_data_const(
                    jones_matrices_workspace
                    ),
                    (const complex<double>*) sdp_mem_data_const(
                    brightness_matrix_predict
                    ),
                    (std::complex<double>*) sdp_mem_data(visibilites)
            );
        }
        else if (source_type == SDP_MEM_FLOAT &&
                station_type == SDP_MEM_DOUBLE)
        {
            scalar_create_phase_difference_Jones_values
            (
                    wavenumber,
                    num_sources,
                    num_stations,
                    (const double*) sdp_mem_data_const(station_coordinates),
                    (const float*) sdp_mem_data_const(source_directions),
                    (const std::complex<double>*) sdp_mem_data_const(
                    jones_matrices
                    ),
                    (std::complex<double>*) sdp_mem_data(
                    jones_matrices_workspace
                    )
            );

            create_brightness_values_for_predict
            (
                    (std::complex<double>*) sdp_mem_data(
                    brightness_matrix_predict
                    ),
                    (const float*) sdp_mem_data_const(source_stoke_parameters),
                    num_sources
            );

            scalar_predict_visibilites
            (
                    num_stations,
                    num_sources,
                    (const std::complex<double>*) sdp_mem_data_const(
                    jones_matrices_workspace
                    ),
                    (const complex<double>*) sdp_mem_data_const(
                    brightness_matrix_predict
                    ),
                    (std::complex<double>*) sdp_mem_data(visibilites)
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }

    if (visibility_location == SDP_MEM_GPU)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "GPU version of the station based direct predict is currently not implemented"
        );
    }
}
