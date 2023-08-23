/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ska-sdp-func/visibility/sdp_station_based_predict.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#define C_0 299792458.0

static void run_and_check(
    const char* test_name,
    bool read_only_output,
    sdp_MemType vis_type,
    sdp_MemType stokes_type,
    sdp_MemType brightness_type,
    sdp_MemType directions_type,
    sdp_MemType coordinates_type,
    sdp_MemType jones_type,
    sdp_MemLocation output_location,
    sdp_Error* status
)
{
    //Generate some test data
    
    const int64_t num_sources = 5;
    const int64_t num_stations = 15;
    const int64_t num_baselines = (num_stations * (num_stations -1))/2;
    int wavelength = C_0 / 1.1e9;
    const int wavenumber = 2 * M_PI / wavelength;
    
    int64_t vis_shape[] = {1,num_baselines,3,4};
    int64_t source_directions_shape[] = {num_sources,3};
    int64_t stokes_parameters_shape[] = {num_sources,4};
    int64_t brightness_matrix_shape[] = {num_sources};
    int64_t jones_matrix_shape[] ={num_sources, num_sources};
    int64_t coordinates_shape[] = {num_stations, num_stations, num_stations};

    sdp_Mem* visibilites = sdp_mem_create(vis_type,output_location, 4, vis_shape, status);
    
    sdp_Mem* source_directions = sdp_mem_create(directions_type, SDP_MEM_CPU,2,source_directions_shape, status);
    
    sdp_Mem* stokes_parameters = sdp_mem_create(stokes_type, SDP_MEM_CPU, 2, stokes_parameters_shape, status);
    
    sdp_Mem* brightness_matrix = sdp_mem_create(brightness_type, output_location, 1, brightness_matrix_shape, status);

    sdp_Mem* jones_matrix = sdp_mem_create(jones_type, SDP_MEM_CPU, 2, jones_matrix_shape, status);

    sdp_Mem* jones_matrix_workspace = sdp_mem_create(jones_type, output_location, 2, jones_matrix_shape,status);

    sdp_Mem* coordinates = sdp_mem_create(coordinates_type, SDP_MEM_CPU,3,coordinates_shape,status);

    sdp_mem_clear_contents(visibilites, status);
    sdp_mem_set_read_only(visibilites, read_only_output);

    sdp_mem_clear_contents(jones_matrix_workspace, status);
    sdp_mem_set_read_only(jones_matrix_workspace, read_only_output);
    
    sdp_mem_clear_contents(brightness_matrix,status);
    sdp_mem_set_read_only(brightness_matrix, read_only_output);

    sdp_mem_random_fill(source_directions, status);
    sdp_mem_random_fill(coordinates,status);
    sdp_mem_random_fill(stokes_parameters,status);
    sdp_mem_random_fill(jones_matrix,status); 

    // Copy inputs into the specified location

    // sdp_Mem* source_directions_in = sdp_mem_create_copy(source_directions, SDP_MEM_CPU, status);
    // //sdp_mem_ref_dec(source_directions);
    // sdp_Mem* coordinates_in = sdp_mem_create_copy(coordinates, SDP_MEM_CPU , status);
    // //sdp_mem_ref_dec(coordinates);
    // sdp_Mem* stokes_parameters_in = sdp_mem_create_copy(stokes_parameters, SDP_MEM_CPU, status);
    // //sdp_mem_ref_dec(stokes_parameters);
    // sdp_Mem* jones_matrix_in = sdp_mem_create_copy(jones_matrix,SDP_MEM_CPU, status);
    // //sdp_mem_ref_dec(jones_matrix);

    // Call the function to test

    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_station_based_predict(
        num_stations,
        coordinates,
        source_directions,
        stokes_parameters,
        wavenumber,
        visibilites,
        jones_matrix,
        brightness_matrix,
        jones_matrix_workspace,
        status
    );

    sdp_mem_ref_dec(visibilites);
    sdp_mem_ref_dec(brightness_matrix);
    sdp_mem_ref_dec(jones_matrix_workspace);
}


int main()
{
    // Check for happy paths.

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Double Precision", false, SDP_MEM_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
        SDP_MEM_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,SDP_MEM_CPU,&status);
        assert(status == SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Single Precision", false, SDP_MEM_DOUBLE, SDP_MEM_FLOAT, SDP_MEM_FLOAT,
        SDP_MEM_FLOAT, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,SDP_MEM_CPU, &status);
        assert(status == SDP_SUCCESS);
    }

// #ifdef SDP_HAVE_CUDA
//     {
//         sdp_Error status = SDP_SUCCESS;
//         run_and_check("GPU version(not implemented)", false, SDP_MEM_DOUBLE, SDP_MEM_FLOAT, SDP_MEM_FLOAT,
//         SDP_MEM_FLOAT, SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_GPU, SDP_MEM_GPU, &status);
//         assert(status == SDP_ERR_MEM_LOCATION);
//     }
// #endif
//     return 0;
}
