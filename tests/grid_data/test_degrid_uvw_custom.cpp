/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <complex>
#include <math.h>

#include "ska-sdp-func/grid_data/sdp_degrid_uvw_custom.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#define C_0 299792458.0
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)
#define INDEX_5D(N5, N4, N3, N2, N1, I5, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * (N4 * I5 + I4) + I3) + I2) + I1)

using std::complex;


static void calculate_coordinates(
        int64_t grid_size, // dimension of the image's subgrid grid_size x grid_size x 4?
        int x_stride, // padding in x dimension
        int y_stride, // padding in y dimension
        int kernel_size, // gcf kernel support
        int kernel_stride, // padding of the gcf kernel
        int oversample, // oversampling of the uv kernel
        int wkernel_stride, // padding of the gcf w kernel
        int oversample_w, // oversampling of the w kernel
        double theta, // conversion parameter from uv coordinates to xy coordinates x=u*theta
        double wstep, // conversion parameter from w coordinates to z coordinates z=w*wstep
        double u, //
        double v, // coordinates of the visibility
        double w, //
        int* grid_offset, // offset in the image subgrid
        int* sub_offset_x, //
        int* sub_offset_y, // fractional coordinates
        int* sub_offset_z, //
        int* grid_coord_x, // grid coordinates
        int* grid_coord_y  //
)
{
    // u or x coordinate
    const double ox = theta * u * oversample;
    const int iox = round(ox) + (grid_size / 2 + 1) * oversample - 1;
    const int home_x = iox / oversample;
    const int frac_x = oversample - 1 - (iox % oversample);

    // v or y coordinate
    const double oy = theta * v * oversample;
    const int ioy = round(oy) + (grid_size / 2 + 1) * oversample - 1;
    const int home_y = ioy / oversample;
    const int frac_y = oversample - 1 - (ioy % oversample);

    // w or z coordinate
    const double oz = (1.0 + w / wstep) * oversample_w;
    const int ioz = round(oz) + oversample_w - 1;
    const int frac_z = oversample_w - 1 - (ioz % oversample_w);

    // FIXME Why is this multiplied by x_stride? (c.f. GPU version).
    *grid_offset = (home_y - kernel_size / 2) * y_stride + (
        home_x - kernel_size / 2) * x_stride;
    *sub_offset_x = kernel_stride * frac_x;
    *sub_offset_y = kernel_stride * frac_y;
    *sub_offset_z = wkernel_stride * frac_z;
    *grid_coord_x = home_x;
    *grid_coord_y = home_y;
}


// Check results of degridding
template<typename VIS_TYPE>
static void check_results(
        const char* test_name,
        int64_t uv_kernel_size,
        int64_t w_kernel_size,
        int64_t x_size,
        int64_t y_size,
        int64_t z_size,
        int64_t num_times,
        int64_t num_baselines,
        int64_t num_channels,
        int64_t num_pols,
        const complex<VIS_TYPE>* grid,
        const double* uvw,
        const double* uv_kernel,
        const double* w_kernel,
        int64_t uv_kernel_oversampling,
        int64_t w_kernel_oversampling,
        double theta,
        double wstep,
        double channel_start_hz,
        double channel_step_hz,
        const bool conjugate,
        complex<VIS_TYPE>* vis,
        const sdp_Error* status
)
{
    const int64_t half_uv_kernel_size = uv_kernel_size / 2;

    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }

    for (int i_time = 0; i_time < num_times; ++i_time)
    {
        for (int i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uvw = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );
            const double baseline_u_metres = uvw[i_uvw];
            const double baseline_v_metres = uvw[i_uvw + 1];
            const double baseline_w_metres = uvw[i_uvw + 2];
            for (int i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                // Get uvw-coordinate scaling.
                const double inv_wavelength =
                        (channel_start_hz + i_channel * channel_step_hz) / C_0;

                int grid_offset = 0;
                int sub_offset_x = 0, sub_offset_y = 0, sub_offset_z = 0;
                int grid_coord_x = 0, grid_coord_y = 0;
                calculate_coordinates(
                        x_size,
                        1,
                        y_size,
                        uv_kernel_size,
                        uv_kernel_size,
                        uv_kernel_oversampling,
                        w_kernel_size,
                        w_kernel_oversampling,
                        theta,
                        wstep,
                        inv_wavelength * baseline_u_metres,
                        inv_wavelength * baseline_v_metres,
                        inv_wavelength * baseline_w_metres,
                        &grid_offset,
                        &sub_offset_x,
                        &sub_offset_y,
                        &sub_offset_z,
                        &grid_coord_x,
                        &grid_coord_y
                );

                // Check point is fully within the grid.
                if (!(grid_coord_x > half_uv_kernel_size &&
                        grid_coord_x < x_size - half_uv_kernel_size &&
                        grid_coord_y > half_uv_kernel_size &&
                        grid_coord_y < y_size - half_uv_kernel_size))
                {
                    continue;
                }

                // De-grid each polarisation.
                for (int i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    complex<VIS_TYPE> vis_local(0, 0);
                    for (int z = 0; z < w_kernel_size; z++)
                    {
                        complex<VIS_TYPE> visz(0, 0);
                        for (int y = 0; y < uv_kernel_size; y++)
                        {
                            const int64_t i_grid_y = grid_coord_y + y -
                                    half_uv_kernel_size;

                            complex<VIS_TYPE> visy(0, 0);
                            for (int x = 0; x < uv_kernel_size; x++)
                            {
                                const int64_t i_grid_x = grid_coord_x + x -
                                        half_uv_kernel_size;

                                const int64_t i_grid = INDEX_5D(
                                        num_channels,
                                        z_size,
                                        y_size,
                                        x_size,
                                        num_pols,
                                        i_channel,
                                        z,
                                        i_grid_y,
                                        i_grid_x,
                                        i_pol
                                );

                                const complex<VIS_TYPE> value = grid[i_grid];

                                visy += uv_kernel[sub_offset_x + x] * value;
                            }
                            visz += uv_kernel[sub_offset_y + y] * visy;
                        }
                        vis_local += w_kernel[sub_offset_z + z] * visz;
                    }
                    if (conjugate) vis_local = std::conj(vis_local);

                    const int64_t i_out = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol
                    );

                    complex<VIS_TYPE> diff = vis[i_out] - vis_local;
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
        bool conjugate,
        sdp_Error* status
)
{
    // Generate some test data.
    const int num_baselines = 14;
    const int num_times = 4;
    const int uv_kernel_size = 8;
    const int uv_kernel_oversampling = 16000;
    const int w_kernel_size = 4;
    const int w_kernel_oversampling = 16000;
    const double channel_start_hz = 100e6;
    const double channel_step_hz = 0.1e6;
    const double theta = 0.1;
    const double wstep = 250.0;

    sdp_MemType grid_type = input_type_complex;
    int64_t grid_shape[] = {num_channels, 4, 512, 512, num_pols};

    int64_t uvw_shape[] = {num_times, num_baselines, 3};
    sdp_MemType uvw_type = input_type;

    sdp_MemType uv_kernel_type = input_type;
    int64_t uv_kernel_shape[] = {uv_kernel_oversampling, uv_kernel_size};

    sdp_MemType w_kernel_type = input_type;
    int64_t w_kernel_shape[] = {w_kernel_oversampling, w_kernel_size};

    int64_t vis_shape[] = {num_times, num_baselines, num_channels, num_pols};
    sdp_MemType vis_type = output_type;

    sdp_Mem* grid = sdp_mem_create(grid_type, SDP_MEM_CPU,
            sizeof(grid_shape) / sizeof(int64_t), grid_shape, status
    );
    sdp_Mem* uvw = sdp_mem_create(uvw_type, SDP_MEM_CPU,
            sizeof(uvw_shape) / sizeof(int64_t), uvw_shape, status
    );
    sdp_Mem* uv_kernel = sdp_mem_create(
            uv_kernel_type, SDP_MEM_CPU, 2, uv_kernel_shape, status
    );
    sdp_Mem* w_kernel = sdp_mem_create(
            w_kernel_type, SDP_MEM_CPU, 2, w_kernel_shape, status
    );
    sdp_Mem* vis = sdp_mem_create(
            vis_type, output_location, 4, vis_shape, status
    );

    sdp_mem_random_fill(grid, status);
    sdp_mem_random_fill(uvw, status);
    sdp_mem_random_fill(uv_kernel, status);
    sdp_mem_random_fill(w_kernel, status);

    sdp_mem_clear_contents(vis, status);
    sdp_mem_set_read_only(vis, read_only_output);

    // Copy inputs to specified location.
    sdp_Mem* grid_in = sdp_mem_create_copy(grid, input_location, status);
    sdp_Mem* uvw_in = sdp_mem_create_copy(uvw, input_location, status);
    sdp_Mem* uv_kernel_in = sdp_mem_create_copy(
            uv_kernel, input_location, status
    );
    sdp_Mem* w_kernel_in = sdp_mem_create_copy(
            w_kernel, input_location, status
    );

    // Call the function to test.
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_degrid_uvw_custom(
            grid_in,
            uvw_in,
            uv_kernel_in,
            w_kernel_in,
            theta,
            wstep,
            channel_start_hz,
            channel_step_hz,
            conjugate,
            vis,
            status
    );

    sdp_mem_ref_dec(grid_in);
    sdp_mem_ref_dec(uvw_in);
    sdp_mem_ref_dec(uv_kernel_in);
    sdp_mem_ref_dec(w_kernel_in);

    // Copy the output for checking.
    sdp_Mem* vis_out = sdp_mem_create_copy(vis, SDP_MEM_CPU, status);
    sdp_mem_ref_dec(vis);

    // Check output
    if (expect_pass)
    {
        int64_t z_size = sdp_mem_shape_dim(grid, 1);
        int64_t y_size = sdp_mem_shape_dim(grid, 2);
        int64_t x_size = sdp_mem_shape_dim(grid, 3);

        check_results(
                test_name,
                uv_kernel_size,
                w_kernel_size,
                x_size,
                y_size,
                z_size,
                num_times,
                num_baselines,
                num_channels,
                num_pols,
                (const complex<double>*)sdp_mem_data_const(grid),
                (const double*)sdp_mem_data_const(uvw),
                (const double*)sdp_mem_data_const(uv_kernel),
                (const double*)sdp_mem_data_const(w_kernel),
                uv_kernel_oversampling,
                w_kernel_oversampling,
                theta,
                wstep,
                channel_start_hz,
                channel_step_hz,
                conjugate,
                (complex<double>*)sdp_mem_data(vis_out),
                status
        );
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
        run_and_check("CPU run, 1 polarisation, 1 channel",
                true, false, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, false, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU run, 4 polarisations, 1 channel",
                true, false, 1, 4,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, false, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU run - complex conjugate, 1 polarisation, 1 channel",
                true, false, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, true, &status
        );
        assert(status == SDP_SUCCESS);
    }

#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU run, 1 polarisation, 1 channel",
                true, false, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, false, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU run, 4 polarisations, 1 channel",
                true, false, 1, 4,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, false, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU run - complex conjugate, 1 polarisation, 1 channel",
                true, false, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_GPU, SDP_MEM_GPU, true, &status
        );
        assert(status == SDP_SUCCESS);
    }
#endif

    // Unhappy paths.
    {
        sdp_Error status = SDP_ERR_RUNTIME;
        run_and_check("Error status set on function entry",
                false, false, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, false, &status
        );
        assert(status == SDP_ERR_RUNTIME);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Read only output",
                false, true, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, false, &status
        );
        assert(status != SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Unsupported data type",
                false, false, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_CHAR,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, false, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Unsupported grid data type",
                false, false, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, false, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Unsupported visibility data type",
                false, false, 1, 1,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_CPU, false, &status
        );
        assert(status == SDP_ERR_DATA_TYPE);
    }

#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("Memory location mismatch",
                false, false, 1, 1,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_CPU, SDP_MEM_GPU, false, &status
        );
        assert(status == SDP_ERR_MEM_LOCATION);
    }
#endif

    return 0;
}
