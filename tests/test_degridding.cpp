/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <complex>
#include <math.h>

#include "ska-sdp-func/degridding/sdp_degridding.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

void calculate_coordinates(
    int64_t grid_size, //dimension of the image's subgrid grid_size x grid_size x 4?
    int x_stride, // padding in x dimension
    int y_stride, // padding in y dimension
    int kernel_size, // gcf kernel support
    int kernel_stride, // padding of the gcf kernel
    int oversample, // oversampling of the uv kernel
    int wkernel_stride, // padding of the gcf w kernel
    int oversample_w, // oversampling of the w kernel
    double theta, //conversion parameter from uv coordinates to xy coordinates x=u*theta
    double wstep, //conversion parameter from w coordinates to z coordinates z=w*wstep 
    double u, // 
    double v, // coordinates of the visibility 
    double w, //
    int *grid_offset, // offset in the image subgrid
    int *sub_offset_x, //
    int *sub_offset_y, // fractional coordinates
    int *sub_offset_z) //
{
    // x coordinate
    double x = theta*u;
    double ox = x*oversample;
    //int iox = lrint(ox);
    int iox = round(ox); // round to nearest
    iox += (grid_size / 2 + 1) * oversample - 1;
    int home_x = iox / oversample;
    int frac_x = oversample - 1 - (iox % oversample);

    // y coordinate
    double y = theta*v;
    double oy = y*oversample;
    //int iox = lrint(ox);
    int ioy = round(oy);
    ioy += (grid_size / 2 + 1) * oversample - 1;
    int home_y = ioy / oversample;
    int frac_y = oversample - 1 - (ioy % oversample);

    // w coordinate
    double z = 1.0 + w/wstep;
    double oz = z*oversample_w;
    //int iox = lrint(ox);
    int ioz = round(oz);
    ioz += oversample_w - 1;
    //int home_z = ioz / oversample_w;
    int frac_z = oversample_w - 1 - (ioz % oversample_w);

    *grid_offset = (home_y-kernel_size/2)*y_stride + (home_x-kernel_size/2)*x_stride;
    *sub_offset_x = kernel_stride * frac_x;
    *sub_offset_y = kernel_stride * frac_y;
    *sub_offset_z = wkernel_stride * frac_z;
}

// Check results of degridding 
static void check_results(
    const char* test_name,
    int64_t uv_kernel_stride,
    int64_t w_kernel_stride,
    int64_t x_size,
    int64_t y_size,
    int64_t vis_count,
    const std::complex<double>* grid,
    const double* vis_coordinates,
    const double* uv_kernel,
    const float* w_kernel,
    int64_t uv_kernel_oversampling,
    int64_t w_kernel_oversampling,
    double theta,
    double wstep, 
    const bool conjugate, 
    std::complex<double>* vis,
    const sdp_Error* status)

{

     if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    } 

    int grid_offset, sub_offset_x, sub_offset_y, sub_offset_z;

     for(int v = 0; v < vis_count; v++)
    {

        int idx = 3*v;

        double u_vis_coordinate = vis_coordinates[idx];
        double v_vis_coordinate = vis_coordinates[idx+1];
        double w_vis_coordinate = vis_coordinates[idx+2];

        calculate_coordinates(
            x_size, 1, y_size,
            uv_kernel_stride, uv_kernel_stride, uv_kernel_oversampling,
            w_kernel_stride, w_kernel_oversampling,
            theta, wstep, 
            u_vis_coordinate, 
            v_vis_coordinate, 
            w_vis_coordinate,
            &grid_offset, 
            &sub_offset_x, &sub_offset_y, &sub_offset_z);

        double vis_r = 0.0, vis_i = 0.0;
        for (int z = 0; z < w_kernel_stride; z++) 
        {
            double visz_r = 0, visz_i = 0;
            for (int y = 0; y < uv_kernel_stride; y++) 
            {
                double visy_r = 0, visy_i = 0;
                for (int x = 0; x < uv_kernel_stride; x++) 
                {
                    double grid_r = 0; //
                    double grid_i = 0; //
                    std::complex<double> temp = grid[z*x_size*y_size + grid_offset + y*y_size + x];
                    grid_r = temp.real();
                    grid_i = temp.imag();
                    visy_r += uv_kernel[sub_offset_x + x] * grid_r;
                    visy_i += uv_kernel[sub_offset_x + x] * grid_i;
                }
                visz_r += uv_kernel[sub_offset_y + y] * visy_r;
                visz_i += uv_kernel[sub_offset_y + y] * visy_i;
            }
            vis_r += w_kernel[sub_offset_z + z] * visz_r;
            vis_i += w_kernel[sub_offset_z + z] * visz_i;

        }

        std::complex<double> temp_result;
        temp_result.real(vis_r);

        if(conjugate) temp_result.imag(-vis_i);
        else temp_result.imag(vis_i);

        std::complex<double> diff = vis[v] - temp_result;

        assert(fabs(real(diff)) < 1e-5);
        assert(fabs(imag(diff)) < 1e-5);        

    }

    SDP_LOG_INFO("%s: Test passed", test_name);

}


static void run_and_check(
    const char* test_name,
    bool expect_pass,
    bool read_only_output,
    sdp_MemType input_type,
    sdp_MemType input_type_complex,
    sdp_MemType input_type_w_kernel,
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
    int64_t grid_shape[] = {512,512,4};
    
    int64_t vis_coordinates_shape[] = {3, 56};
    sdp_MemType vis_coordinates_type = input_type;   
    
    sdp_MemType uv_kernal_type = input_type;    
    int64_t uv_kernel_shape[] = {uv_kernel_oversampling * sdp_MemType(input_type)};
    
    sdp_MemType w_kernel_type = input_type_w_kernel;
    int64_t w_kernel_shape[] = {w_kernel_oversampling * sdp_MemType(w_kernel_type)};
    
    int64_t vis_shape[] = {3, 56};
    sdp_MemType vis_type = output_type;  

    sdp_Mem* grid = sdp_mem_create(
        grid_type, input_location, 3, grid_shape, status);
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

    //sdp_mem_ref_dec(grid_in);
    sdp_mem_ref_dec(vis_coordinates_in);
    sdp_mem_ref_dec(uv_kernel_in);
    sdp_mem_ref_dec(w_kernel_in);

    // Copy the output for checking. 
    sdp_Mem* vis_out = sdp_mem_create_copy(vis, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(vis);

    // Check output
    if (expect_pass)
    {
        int64_t uv_kernel_stride = sdp_mem_stride_dim(uv_kernel, 0);
        int64_t w_kernel_stride = sdp_mem_stride_dim(w_kernel, 0);

        int64_t vis_count = sdp_mem_shape_dim(vis_coordinates, 1);

        int64_t x_size = sdp_mem_shape_dim(grid,0);
        int64_t y_size = sdp_mem_shape_dim(grid,1);
     
        check_results(test_name,
            uv_kernel_stride,
            w_kernel_stride,
            x_size,
            y_size,
            vis_count,
            (const std::complex<double>*)sdp_mem_data_const(grid),
            (const double*)sdp_mem_data_const(vis_coordinates),
            (const double*)sdp_mem_data_const(uv_kernel),
            (const float*)sdp_mem_data_const(w_kernel),
            uv_kernel_oversampling,
            w_kernel_oversampling,
            theta,
            wstep, 
            conjugate, 
            (std::complex<double>*)sdp_mem_data(vis_out),
            status);
    }

    sdp_mem_ref_dec(grid);
    sdp_mem_ref_dec(vis_coordinates);
    sdp_mem_ref_dec(uv_kernel);
    sdp_mem_ref_dec(w_kernel);
    sdp_mem_ref_dec(vis_out);
    
}

int main()
{

// Happy paths.
    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("CPU run", true, false, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, 
            16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_SUCCESS);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("CPU run - complex conjugate", true, false, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, 
            16000, 16000, 0.1, 250, true, &status);
    assert(status == SDP_SUCCESS);
    }


// Unhappy paths.
    {
    sdp_Error status = SDP_ERR_RUNTIME;
    run_and_check("Error status set on function entry", false, false, SDP_MEM_DOUBLE, 
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_RUNTIME);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Read only output", false, true, SDP_MEM_DOUBLE,
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU,
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status != SDP_SUCCESS);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Unsuported data type", false, false, SDP_MEM_CHAR,
            SDP_MEM_CHAR, SDP_MEM_CHAR, SDP_MEM_CHAR, SDP_MEM_CPU,
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_DATA_TYPE);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Unsuported W Kernel data type", false, false, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, 
            16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_DATA_TYPE);
    }

#ifdef SDP_HAVE_CUDA
    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Memory location mismatch", false, false, SDP_MEM_DOUBLE,
        SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_FLOAT, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_GPU,
        16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_MEM_LOCATION);
    }

#endif

}