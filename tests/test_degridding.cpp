/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "ska-sdp-func/degridding/sdp_degridding.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"


// TODO results check to be added when degridding implementation is checked and confirmed good 
// static void check_results(
//         const char* test_name,
//         const sdp_Mem* grid,
//         const sdp_Mem* vis_coordinates,
//         const sdp_Mem* uv_kernel,
//         const sdp_Mem* w_kernel,
//         const int64_t uv_kernel_oversampling,
//         const int64_t w_kernel_oversampling,
//         const double theta,
//         const double wstep, 
//         const bool conjugate, 
//         const sdp_Mem* vis,
//         const sdp_Error* status)

// {

//      if (*status)
//     {
//         SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
//         return;
//     } 

// }

static void run_and_check(
    const char* test_name,
    bool read_only_output,
    sdp_MemType input_type,
    sdp_MemType input_type_complex,
    sdp_MemType output_type,
    sdp_MemLocation input_location,
    sdp_MemLocation output_location,
    int64_t uv_kernel_oversampling,
    int64_t w_kernel_oversampling,
    double theta,
    double wstep, 
    bool conjugate, 
    sdp_Error* status)

{
    
    // Generate some test data.

    sdp_MemType grid_type = input_type_complex;
    int64_t grid_shape[] = {1,512*512*sdp_mem_type_size(grid_type)};
    
    int64_t vis_coordinates_shape[] = {3, 56};
    sdp_MemType vis_coordinates_type = input_type;   
    
    sdp_MemType uv_kernal_type = input_type;    
    int64_t uv_kernel_shape[] = {uv_kernel_oversampling * sdp_mem_type_size(uv_kernal_type)};
    
    sdp_MemType w_kernel_type = input_type;
    int64_t w_kernel_shape[] = {w_kernel_oversampling * sdp_mem_type_size(w_kernel_type)};
    
    int64_t vis_shape[] = {3, 56};
    sdp_MemType vis_type = output_type;  

    sdp_Mem* grid = sdp_mem_create(
        grid_type, input_location, 2, grid_shape, status);
    sdp_Mem* vis_coordinates = sdp_mem_create(
        vis_coordinates_type, input_location, 2, vis_coordinates_shape, status);
    sdp_Mem* uv_kernel = sdp_mem_create(
        uv_kernal_type, input_location, 1, uv_kernel_shape, status);
    sdp_Mem* w_kernel = sdp_mem_create(
        w_kernel_type, input_location, 1, w_kernel_shape, status);

    sdp_Mem* vis = sdp_mem_create(
        vis_type, output_location, 2, vis_shape, status);


    sdp_mem_random_fill(grid, status);
    sdp_mem_random_fill(vis_coordinates, status);
    sdp_mem_random_fill(uv_kernel, status);
    sdp_mem_random_fill(w_kernel, status);

    sdp_mem_clear_contents(vis, status);
    sdp_mem_set_read_only(vis, read_only_output);

    // Copy inputs to specified location.

    sdp_Mem* grid_in = sdp_mem_create_copy(grid, input_location, status);
    sdp_Mem* vis_coordinates_in = sdp_mem_create_copy(vis_coordinates, input_location, status);
    sdp_Mem* uv_kernel_in = sdp_mem_create_copy(uv_kernel, input_location, status);
    sdp_Mem* w_kernel_in = sdp_mem_create_copy(w_kernel, input_location, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_degridding(grid_in, vis_coordinates_in, uv_kernel_in, w_kernel_in, 
            uv_kernel_oversampling, w_kernel_oversampling, 
            theta, wstep, conjugate, vis, status);

    sdp_mem_ref_dec(grid);
    sdp_mem_ref_dec(vis_coordinates);
    sdp_mem_ref_dec(uv_kernel);
    sdp_mem_ref_dec(w_kernel);


    // Copy the output for checking. 

    sdp_Mem* vis_out = sdp_mem_create_copy(vis, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(vis);

    // Check output

    sdp_mem_ref_dec(vis_out);
    
}

int main()
{



// Happy paths.

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("CPU run", false, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, 
            16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_SUCCESS);
    }


// Unappy paths.
    {
    sdp_Error status = SDP_ERR_RUNTIME;
    run_and_check("Error status set on function entry", false, SDP_MEM_DOUBLE, 
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_RUNTIME);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Read only output", true, SDP_MEM_DOUBLE,
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU,
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status != SDP_SUCCESS);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Unsuported data type", false, SDP_MEM_CHAR,
            SDP_MEM_CHAR, SDP_MEM_CHAR, SDP_MEM_CPU,
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_DATA_TYPE);
    }

#ifdef SDP_HAVE_CUDA
    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Memory location mismatch", false, SDP_MEM_DOUBLE,
        SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_GPU,
        16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_MEM_LOCATION);
    }

#endif


}