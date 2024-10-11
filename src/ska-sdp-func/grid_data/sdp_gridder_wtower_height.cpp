/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_height.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using std::complex;

// Begin anonymous namespace for file-local functions.
namespace {

// Local function to find the accuracy of the supplied gridder configuration.
double find_gridder_accuracy(
        sdp_GridderWtowerUVW* kernel,
        double fov,
        double subgrid_frac,
        int num_samples,
        double w,
        sdp_Error* status
)
{
    if (*status) return 0;

    // Get parameters from gridding kernel.
    const int image_size = sdp_gridder_wtower_uvw_image_size(kernel);
    const int subgrid_size = sdp_gridder_wtower_uvw_subgrid_size(kernel);
    const double theta = sdp_gridder_wtower_uvw_theta(kernel);
    const double shear_u = sdp_gridder_wtower_uvw_shear_u(kernel);
    const double shear_v = sdp_gridder_wtower_uvw_shear_v(kernel);

    // Allocate local scratch arrays.
    if (num_samples == 0) num_samples = 3;
    const int64_t num_rows = num_samples * num_samples;
    const int64_t num_src = 4;
    const int64_t image_shape[] = {image_size, image_size};
    const int64_t src_lmn_shape[] = {num_src, 3};
    const int64_t subgrid_shape[] = {subgrid_size, subgrid_size};
    const int64_t uvws_shape[] = {num_rows, 3};
    const int64_t vis_shape[] = {num_rows, 1};
    sdp_Mem* image = sdp_mem_create(
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, image_shape, status
    );
    sdp_Mem* src_flux = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, &num_src, status
    );
    sdp_Mem* src_lmn = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, src_lmn_shape, status
    );
    sdp_Mem* start_chs = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 1, &num_rows, status
    );
    sdp_Mem* end_chs = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 1, &num_rows, status
    );
    sdp_Mem* subgrid = sdp_mem_create(
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, subgrid_shape, status
    );
    sdp_Mem* uvws = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, uvws_shape, status
    );
    sdp_Mem* vis_ref = sdp_mem_create(
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, vis_shape, status
    );
    sdp_Mem* vis_test = sdp_mem_create(
            SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, vis_shape, status
    );
    if (*status || !image || !src_flux || !src_lmn || !start_chs || !end_chs ||
            !subgrid || !uvws || !vis_ref || !vis_test)
    {
        sdp_mem_free(image);
        sdp_mem_free(src_flux);
        sdp_mem_free(src_lmn);
        sdp_mem_free(start_chs);
        sdp_mem_free(end_chs);
        sdp_mem_free(subgrid);
        sdp_mem_free(uvws);
        sdp_mem_free(vis_ref);
        sdp_mem_free(vis_test);
        return 0;
    }

    // Make the worst-case image, with sources in the corners.
    // Note that this needs to be of complex type for the FFT.
    sdp_gridder_worst_case_image(theta, fov, image, status);
    sdp_gridder_image_to_flmn(
            image, theta, shear_u, shear_v, NULL, src_flux, src_lmn, status
    );

    // Apply correction.
    sdp_gridder_wtower_uvw_degrid_correct(kernel, image, 0, 0, 0, status);

    // Extract subgrid:
    // i.e. subgrid = ifft(subgrid_cut_out(fft(image), subgrid_size))
    sdp_Fft* fft_grid = sdp_fft_create(image, image, 2, true, status);
    sdp_Fft* ifft_subgrid = sdp_fft_create(subgrid, subgrid, 2, false, status);
    sdp_fft_phase(image, status);
    sdp_fft_exec(fft_grid, image, image, status);
    sdp_fft_phase(image, status);
    sdp_gridder_subgrid_cut_out(image, 0, 0, subgrid, status);
    sdp_mem_free(image);
    sdp_fft_phase(subgrid, status);
    sdp_fft_exec(ifft_subgrid, subgrid, subgrid, status);
    sdp_fft_phase(subgrid, status);
    sdp_fft_norm(subgrid, status); // To match numpy's ifft.
    sdp_fft_free(fft_grid);
    sdp_fft_free(ifft_subgrid);

    // Determine error at random points with (u, v) less than
    // subgrid_size / 3 distance from centre (optimal w-tower size -
    // we assume that this is always greater than any support the
    // gridding kernel might need, so we can see it as included).
    // First calculate start, end, and step size for u and v,
    // then fill the uvws array using the provided value of w.
    if (subgrid_frac == 0.0) subgrid_frac = 2.0 / 3.0;
    const double start = -subgrid_size * subgrid_frac / theta / 2;
    const double end = subgrid_size * subgrid_frac / theta / 2;
    const double step = (end - start) / (num_samples - 1);
    sdp_MemViewCpu<double, 2> uvws_;
    sdp_mem_check_and_view(uvws, &uvws_, status);
    for (int i = 0, index = 0; i < num_samples; ++i)
    {
        for (int j = 0; j < num_samples; ++j, ++index)
        {
            uvws_(index, 0) = start + j * step;
            uvws_(index, 1) = start + i * step;
            uvws_(index, 2) = w;
        }
    }

    // Test gridder at these points.
    sdp_MemViewCpu<int, 1> end_chs_;
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    for (int64_t i = 0; i < num_rows; ++i)
    {
        end_chs_(i) = 1;
    }
    sdp_mem_clear_contents(vis_test, status);
    sdp_mem_clear_contents(start_chs, status);
    sdp_gridder_wtower_uvw_degrid(kernel, subgrid, 0, 0, 0, C_0, C_0,
            uvws, start_chs, end_chs, vis_test, status
    );
    sdp_mem_free(subgrid);
    sdp_mem_free(start_chs);
    sdp_mem_free(end_chs);

    // Generate reference data using a DFT.
    sdp_MemViewCpu<const double, 1> flux_;
    sdp_MemViewCpu<const double, 2> lmn_;
    sdp_MemViewCpu<complex<double>, 2> vis_ref_;
    sdp_mem_check_and_view(src_flux, &flux_, status);
    sdp_mem_check_and_view(src_lmn, &lmn_, status);
    sdp_mem_check_and_view(vis_ref, &vis_ref_, status);
    sdp_mem_clear_contents(vis_ref, status);
    #pragma omp parallel for
    for (int64_t i_row = 0; i_row < num_rows; ++i_row)
    {
        double u = uvws_(i_row, 0), v = uvws_(i_row, 1), w = uvws_(i_row, 2);
        for (int64_t i_src = 0; i_src < num_src; ++i_src)
        {
            const double phase = -2.0 * M_PI * (
                u * lmn_(i_src, 0) + v * lmn_(i_src, 1) + w * lmn_(i_src, 2)
            );
            const complex<double> phasor(cos(phase), sin(phase));
            vis_ref_(i_row, 0) += flux_(i_src) * phasor;
        }
    }
    sdp_mem_free(src_flux);
    sdp_mem_free(src_lmn);
    sdp_mem_free(uvws);

    // Calculate and return root mean square error.
    const double rmse = sdp_gridder_rms_diff(vis_test, vis_ref, status);
    sdp_mem_free(vis_ref);
    sdp_mem_free(vis_test);
    return rmse;
}

} // End anonymous namespace for file-local functions.


double sdp_gridder_determine_max_w_tower_height(
        int image_size,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int oversampling,
        int w_support,
        int w_oversampling,
        double fov,
        double subgrid_frac,
        int num_samples,
        double target_err,
        sdp_Error* status
)
{
    if (*status) return 0.0;

    // Create a gridding kernel to use for the assessment.
    sdp_GridderWtowerUVW* kernel = sdp_gridder_wtower_uvw_create(
            image_size, subgrid_size, theta, w_step, shear_u, shear_v,
            support, oversampling, w_support, w_oversampling, status
    );
    if (*status)
    {
        sdp_gridder_wtower_uvw_free(kernel);
        return 0.0;
    }

    // If no target error is specified, default to twice the error at w = 0.
    if (target_err == 0.0)
    {
        target_err = 2 * find_gridder_accuracy(
                kernel, fov, subgrid_frac, num_samples, 0.0, status
        );
    }

    // Start a simple binary search.
    double return_val = 0.0;
    int iw = 1;
    int diw = 1;
    bool accelerate = true;
    while (!(*status))  // while(true), but with error checking.
    {
        // Determine error.
        const double err = find_gridder_accuracy(
                kernel, fov, subgrid_frac, num_samples, iw * w_step, status
        );

        // Below? Advance. Above? Go back.
        if (err < target_err)
        {
            if (accelerate)
            {
                diw *= 2;
            }
            else if (diw > 1)
            {
                diw /= 2;
            }
            else
            {
                return_val = 2 * iw;
                break;
            }
            iw += diw;
        }
        else if (diw > 1)
        {
            diw /= 2;
            iw -= diw;
            accelerate = false;
        }
        else
        {
            return_val = 2 * (iw - 1);
            break;
        }
    }
    sdp_gridder_wtower_uvw_free(kernel);
    return return_val;
}


void sdp_gridder_worst_case_image(
        double theta,
        double fov,
        sdp_Mem* image,
        sdp_Error* status
)
{
    if (*status) return;
    const int image_size = (int) sdp_mem_shape_dim(image, 0);
    if (sdp_mem_num_dims(image) != 2 ||
            image_size != sdp_mem_shape_dim(image, 1))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Image must be square");
        return;
    }

    // Make sources / image with source in corners of fov. Make sure it doesn't
    // divide the subgrid size equally (then there's a good chance we're
    // operating on-grid, and therefore not actually testing the window
    // function).
    int fov_edge = int(image_size / theta * fov / 2);
    while (image_size % fov_edge == 0)
    {
        fov_edge -= 1;
    }

    // Put sources into corners, careful not to generate actual symmetries.
    sdp_mem_clear_contents(image, status);
    if (sdp_mem_type(image) == SDP_MEM_COMPLEX_DOUBLE)
    {
        sdp_MemViewCpu<complex<double>, 2> image_;
        sdp_mem_check_and_view(image, &image_, status);
        image_(image_size / 2 + fov_edge, image_size / 2 + fov_edge) = 0.3;
        image_(image_size / 2 - fov_edge, image_size / 2 - fov_edge) = 0.2;
        image_(image_size / 2 + fov_edge, image_size / 2 - fov_edge - 1) = 0.3;
        image_(image_size / 2 - fov_edge - 1, image_size / 2 + fov_edge) = 0.2;
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }
}
