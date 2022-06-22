/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <complex>
#include <math.h>

#include "ska-sdp-func/degridding/sdp_degrid_uvw_custom.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

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
    int64_t num_times,
    int64_t num_baselines,
    int64_t num_channels,
    int64_t num_pols,
    const std::complex<double>* grid,
    const double* uvw,
    const double* uv_kernel,
    const double* w_kernel,
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

    for (int i_time = 0; i_time < num_times; ++i_time)
    {
        for (int i_channel = 0; i_channel < num_channels; ++i_channel)
        {
            for (int i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
            {
                // Load uvw-coordinates.
                const unsigned int i_uvw = INDEX_4D(
                        num_times, num_baselines, num_channels, 3,
                        i_time, i_baseline, i_channel, 0);
                double u_vis_coordinate = uvw[i_uvw];
                double v_vis_coordinate = uvw[i_uvw + 1];
                double w_vis_coordinate = uvw[i_uvw + 2];


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

                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const unsigned int i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol);
                    
                    std::complex<double> diff = vis[i_out] - temp_result;

                    assert(fabs(real(diff)) < 1e-5);
                    assert(fabs(imag(diff)) < 1e-5); 
                }
            }
        }
     

    }

    SDP_LOG_INFO("%s: Test passed", test_name);

}


static void run_and_check(
    const char* test_name,
    bool expect_pass,
    bool read_only_output,
    int num_channels,
    int num_pols,
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
    const int num_baselines = 14;
    const int num_times = 4;
    const int uv_kernel_stride = 8;
    const int w_kernel_stride = 4;

    sdp_MemType grid_type = input_type_complex;
    int64_t grid_shape[] = {num_channels, 4, 512, 512, num_pols};

    int64_t uvw_shape[] = {num_times, num_baselines, num_channels, 3};
    sdp_MemType uvw_type = input_type;
     
    sdp_MemType uv_kernel_type = input_type;    
    int64_t uv_kernel_shape[] = {uv_kernel_oversampling * uv_kernel_stride};
    
    sdp_MemType w_kernel_type = input_type;
    int64_t w_kernel_shape[] = {w_kernel_oversampling * w_kernel_stride};
    
    int64_t vis_shape[] = {num_times, num_baselines, num_channels, num_pols};
    sdp_MemType vis_type = output_type;  

    sdp_Mem* grid = sdp_mem_create(
        grid_type, input_location, 5, grid_shape, status);
    sdp_Mem* uvw = sdp_mem_create(
        uvw_type, input_location, 4, uvw_shape, status);
    sdp_Mem* uv_kernel = sdp_mem_create(
        uv_kernel_type, input_location, 1, uv_kernel_shape, status);
    sdp_Mem* w_kernel = sdp_mem_create(
        w_kernel_type, input_location, 1, w_kernel_shape, status);

    sdp_Mem* vis = sdp_mem_create(
        vis_type, output_location, 4, vis_shape, status);


    sdp_mem_random_fill(grid, status);
    sdp_mem_random_fill(uvw, status);
    sdp_mem_random_fill(uv_kernel, status);
    sdp_mem_random_fill(w_kernel, status);

    sdp_mem_clear_contents(vis, status);
    sdp_mem_set_read_only(vis, read_only_output);

    // Copy inputs to specified location.

    sdp_Mem* grid_in = sdp_mem_create_copy(grid, input_location, status);
    sdp_Mem* uvw_in = sdp_mem_create_copy(uvw, input_location, status);
    sdp_Mem* uv_kernel_in = sdp_mem_create_copy(uv_kernel, input_location, status);
    sdp_Mem* w_kernel_in = sdp_mem_create_copy(w_kernel, input_location, status);

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_degrid_uvw_custom(grid_in, uvw_in, uv_kernel_in, w_kernel_in, 
            uv_kernel_oversampling, w_kernel_oversampling, 
            theta, wstep, conjugate, vis, status);

    //sdp_mem_ref_dec(grid_in);
    sdp_mem_ref_dec(uvw_in);
    sdp_mem_ref_dec(uv_kernel_in);
    sdp_mem_ref_dec(w_kernel_in);

    // Copy the output for checking. 
    sdp_Mem* vis_out = sdp_mem_create_copy(vis, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(vis);

    // Check output
    if (expect_pass)
    {
        // int64_t uv_kernel_stride = sdp_mem_stride_dim(uv_kernel, 0);
        // int64_t w_kernel_stride = sdp_mem_stride_dim(w_kernel, 0);

        int64_t x_size = sdp_mem_shape_dim(grid,2);
        int64_t y_size = sdp_mem_shape_dim(grid,3);
     
        check_results(test_name,
            uv_kernel_stride,
            w_kernel_stride,
            x_size,
            y_size,
            num_times,
            num_baselines,
            num_channels,
            num_pols,
            (const std::complex<double>*)sdp_mem_data_const(grid),
            (const double*)sdp_mem_data_const(uvw),
            (const double*)sdp_mem_data_const(uv_kernel),
            (const double*)sdp_mem_data_const(w_kernel),
            uv_kernel_oversampling,
            w_kernel_oversampling,
            theta,
            wstep, 
            conjugate, 
            (std::complex<double>*)sdp_mem_data(vis_out),
            status);
    }

    sdp_mem_ref_dec(grid);
    sdp_mem_ref_dec(uvw);
    sdp_mem_ref_dec(uv_kernel);
    sdp_mem_ref_dec(w_kernel);
    sdp_mem_ref_dec(vis_out);
    
}

int main()
{

// Happy paths.
    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("CPU run, 1 polarisation, 1 channel", true, false, 1, 1, SDP_MEM_DOUBLE, 
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, 
            16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_SUCCESS);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("CPU run - complex conjugate, 1 polarisation, 1 channel", true, false, 1, 1,
            SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, 
            16000, 16000, 0.1, 250, true, &status);
    assert(status == SDP_SUCCESS);
    }


// Unhappy paths.
    {
    sdp_Error status = SDP_ERR_RUNTIME;
    run_and_check("Error status set on function entry", false, false, 1, 1, SDP_MEM_DOUBLE, 
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_RUNTIME);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Read only output", false, true, 1, 1, SDP_MEM_DOUBLE,
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU,
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_RUNTIME);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Unsuported data type", false, false, 1, 1, SDP_MEM_DOUBLE,
            SDP_MEM_CHAR, SDP_MEM_DOUBLE, SDP_MEM_CPU,
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_DATA_TYPE);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Wrong grid data type", false, false, 1, 1, SDP_MEM_DOUBLE,
            SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU,
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_DATA_TYPE);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Wrong visabilities data type", false, false, 1, 1, SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_DOUBLE, SDP_MEM_DOUBLE, SDP_MEM_CPU,
            SDP_MEM_CPU, 16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_DATA_TYPE);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("CPU run, 1 polarisation, 2 channels", false, false, 2, 1, SDP_MEM_DOUBLE, 
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, 
            16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_RUNTIME);
    }

    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("CPU run, 2 polarisation, 1 channel", false, false, 1, 2, SDP_MEM_DOUBLE, 
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_CPU, 
            16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_RUNTIME);
    }

#ifdef SDP_HAVE_CUDA
    {
    sdp_Error status = SDP_SUCCESS;
    run_and_check("Memory location mismatch", false, false, 1, 1, SDP_MEM_DOUBLE,
        SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, SDP_MEM_GPU,
        16000, 16000, 0.1, 250, false, &status);
    assert(status == SDP_ERR_MEM_LOCATION);
    }

#endif

}