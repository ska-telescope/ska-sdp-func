/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/grid_data/sdp_grid_wstack_wtower.h"
#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"


void sdp_grid_wstack_wtower_degrid_all(
        const sdp_Mem* image,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvw,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int oversampling,
        int w_support,
        int w_oversampling,
        double subgrid_frac,
        double w_tower_height,
        int verbosity,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(vis);
    if (sdp_mem_location(image) != loc || sdp_mem_location(uvw) != loc)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        return;
    }
    if (loc != SDP_MEM_CPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR(
                "Only a CPU version of sdp_grid_wstack_wtower_degrid_all "
                "is currently implemented"
        );
        return;
    }
    if (sdp_mem_num_dims(vis) != 2 || sdp_mem_num_dims(uvw) != 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Visibilities and (u,v,w)-coordinates must be 2D");
        return;
    }
#ifdef _OPENMP
    omp_set_nested(0);
#endif
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    const int64_t num_chan = sdp_mem_shape_dim(vis, 1);

    // Create the gridder kernel.
    const int image_size = (int) sdp_mem_shape_dim(image, 0);
    sdp_GridderWtowerUVW* kernel = sdp_gridder_wtower_uvw_create(
            image_size, subgrid_size, theta, w_step, shear_u, shear_v,
            support, oversampling, w_support, w_oversampling, status
    );

    // Assume we're using all visibilities.
    sdp_Mem* start_ch_orig = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 1, &num_rows, status
    );
    sdp_Mem* end_ch_orig = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 1, &num_rows, status
    );
    sdp_mem_clear_contents(start_ch_orig, status);
    int* end_ch_ = (int*) sdp_mem_data(end_ch_orig);
    for (int64_t i = 0; i < num_rows; ++i)
    {
        end_ch_[i] = num_chan;
    }

    // Determine effective size of sub-grids and w-tower height
    // (both depend critically on how much we want to "use" of the sub-grid).
    const int eff_sg_size = int(floor(subgrid_size * subgrid_frac));
    const double eff_sg_distance = eff_sg_size / theta;
    const double w_stack_distance = w_tower_height * w_step;

    // Determine (maximum) number of sub-grids and w-stacking planes needed.
    const double eta = 1e-5;
    double uvw_min[3] = {0.0, 0.0, 0.0}, uvw_max[3] = {0.0, 0.0, 0.0};
    sdp_gridder_uvw_bounds_all(
            uvw, freq0_hz, dfreq_hz, start_ch_orig, end_ch_orig,
            uvw_min, uvw_max, status
    );
    int64_t min_iu = int64_t(floor(uvw_min[0] / eff_sg_distance + 0.5 - eta));
    int64_t max_iu = int64_t(floor(uvw_max[0] / eff_sg_distance + 0.5 + eta));
    int64_t min_iv = int64_t(floor(uvw_min[1] / eff_sg_distance + 0.5 - eta));
    int64_t max_iv = int64_t(floor(uvw_max[1] / eff_sg_distance + 0.5 + eta));
    int64_t min_iw = int64_t(floor(uvw_min[2] / w_stack_distance + 0.5 - eta));
    int64_t max_iw = int64_t(floor(uvw_max[2] / w_stack_distance + 0.5 + eta));

    // Create an FFT plan for the whole grid / image.
    sdp_Fft* fft = sdp_fft_create(image, image, 2, true, status);

    // Clear output visibilities.
    sdp_mem_clear_contents(vis, status);

    // Loop over w-planes.
    for (int64_t iw = min_iw; iw <= max_iw; ++iw)
    {
        // Select visibilities on w-plane.
        sdp_Mem* start_ch_w = sdp_mem_create_copy(start_ch_orig, loc, status);
        sdp_Mem* end_ch_w = sdp_mem_create_copy(end_ch_orig, loc, status);
        double min_w = iw * w_stack_distance - w_stack_distance / 2;
        double max_w = (iw + 1) * w_stack_distance - w_stack_distance / 2;
        sdp_gridder_clamp_channels_single(uvw, 2, freq0_hz, dfreq_hz,
                start_ch_w, end_ch_w, min_w, max_w, status
        );
        double num_vis = 0;
        sdp_gridder_sum_diff(end_ch_w, start_ch_w, &num_vis, status);
        if (num_vis == 0)
        {
            sdp_mem_free(start_ch_w);
            sdp_mem_free(end_ch_w);
            continue;
        }

        // Do image correction / w-stacking.
        sdp_Mem* grid = sdp_mem_create_copy(image, SDP_MEM_CPU, status);
        sdp_gridder_wtower_uvw_degrid_correct(
                kernel, grid, 0, 0, iw * w_tower_height, status
        );
        sdp_fft_phase(grid, status);
        sdp_fft_exec(fft, grid, grid, status);
        sdp_fft_phase(grid, status);

        // Loop over sub-grid (towers) in u.
        #pragma omp parallel for
        for (int64_t iu = min_iu; iu <= max_iu; ++iu)
        {
            // Select visibilities in column.
            sdp_Mem* start_ch_u = sdp_mem_create_copy(start_ch_w, loc, status);
            sdp_Mem* end_ch_u = sdp_mem_create_copy(end_ch_w, loc, status);
            double min_u = iu * eff_sg_distance - eff_sg_distance / 2;
            double max_u = (iu + 1) * eff_sg_distance - eff_sg_distance / 2;
            sdp_gridder_clamp_channels_single(uvw, 0, freq0_hz, dfreq_hz,
                    start_ch_u, end_ch_u, min_u, max_u, status
            );
            double num_vis = 0;
            sdp_gridder_sum_diff(end_ch_u, start_ch_u, &num_vis, status);
            if (num_vis == 0)
            {
                sdp_mem_free(start_ch_u);
                sdp_mem_free(end_ch_u);
                continue;
            }

            // Allocate a sub-grid and prepare an iFFT plan for it.
            const int64_t subgrid_shape[] = {subgrid_size, subgrid_size};
            sdp_Mem* subgrid_image = sdp_mem_create(
                    sdp_mem_type(grid), loc, 2, subgrid_shape, status
            );
            sdp_Fft* ifft_subgrid = sdp_fft_create(
                    subgrid_image, subgrid_image, 2, false, status
            );

            // Loop over sub-grid (towers) in v.
            for (int64_t iv = min_iv; iv <= max_iv; ++iv)
            {
                // Select visibilities in sub-grid.
                sdp_Mem* start_ch_v = sdp_mem_create_copy(
                        start_ch_u, loc, status
                );
                sdp_Mem* end_ch_v = sdp_mem_create_copy(end_ch_u, loc, status);
                double min_v = iv * eff_sg_distance - eff_sg_distance / 2;
                double max_v = (iv + 1) * eff_sg_distance - eff_sg_distance / 2;
                sdp_gridder_clamp_channels_single(uvw, 1, freq0_hz, dfreq_hz,
                        start_ch_v, end_ch_v, min_v, max_v, status
                );
                double num_vis = 0;
                sdp_gridder_sum_diff(end_ch_v, start_ch_v, &num_vis, status);
                if (num_vis == 0)
                {
                    sdp_mem_free(start_ch_v);
                    sdp_mem_free(end_ch_v);
                    continue;
                }
                if (verbosity > 0)
                {
                    SDP_LOG_INFO("subgrid %d/%d/%d: %.0f visibilities",
                            iu, iv, iw, num_vis
                    );
                }

                // Prepare sub-grid.
                // subgrid_image = ifft(
                //     subgrid_cut_out(
                //         shift_grid(grid, iu * eff_sg_size, iv * eff_sg_size),
                //         kernel.subgrid_size
                //     )
                // )
                sdp_gridder_subgrid_cut_out(
                        grid, iu * eff_sg_size, iv * eff_sg_size,
                        subgrid_image, status
                );
                sdp_fft_phase(subgrid_image, status);
                sdp_fft_exec(
                        ifft_subgrid, subgrid_image, subgrid_image, status
                );
                sdp_fft_norm(subgrid_image, status); // To match numpy's ifft.
                sdp_fft_phase(subgrid_image, status);

                // Degrid visibilities.
                sdp_gridder_wtower_uvw_degrid(kernel, subgrid_image,
                        iu * eff_sg_size, iv * eff_sg_size, iw * w_tower_height,
                        freq0_hz, dfreq_hz, uvw, start_ch_v, end_ch_v,
                        vis, status
                );
                sdp_mem_free(start_ch_v);
                sdp_mem_free(end_ch_v);
            }
            sdp_fft_free(ifft_subgrid);
            sdp_mem_free(subgrid_image);
            sdp_mem_free(start_ch_u);
            sdp_mem_free(end_ch_u);
        }
        sdp_mem_free(grid);
        sdp_mem_free(start_ch_w);
        sdp_mem_free(end_ch_w);
    }
    sdp_fft_free(fft);
    sdp_mem_free(start_ch_orig);
    sdp_mem_free(end_ch_orig);
    sdp_gridder_wtower_uvw_free(kernel);
}


void sdp_grid_wstack_wtower_grid_all(
        const sdp_Mem* vis,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvw,
        int subgrid_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int oversampling,
        int w_support,
        int w_oversampling,
        double subgrid_frac,
        double w_tower_height,
        int verbosity,
        sdp_Mem* image,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(vis);
    if (sdp_mem_location(image) != loc || sdp_mem_location(uvw) != loc)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        return;
    }
    if (loc != SDP_MEM_CPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR(
                "Only a CPU version of sdp_grid_wstack_wtower_grid_all "
                "is currently implemented"
        );
        return;
    }
    if (sdp_mem_num_dims(vis) != 2 || sdp_mem_num_dims(uvw) != 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Visibilities and (u,v,w)-coordinates must be 2D");
        return;
    }
#ifdef _OPENMP
    omp_set_nested(0);
#endif
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    const int64_t num_chan = sdp_mem_shape_dim(vis, 1);

    // Create the gridder kernel.
    const int image_size = (int) sdp_mem_shape_dim(image, 0);
    sdp_GridderWtowerUVW* kernel = sdp_gridder_wtower_uvw_create(
            image_size, subgrid_size, theta, w_step, shear_u, shear_v,
            support, oversampling, w_support, w_oversampling, status
    );

    // Assume we're using all visibilities.
    sdp_Mem* start_ch_orig = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 1, &num_rows, status
    );
    sdp_Mem* end_ch_orig = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 1, &num_rows, status
    );
    sdp_mem_clear_contents(start_ch_orig, status);
    int* end_ch_ = (int*) sdp_mem_data(end_ch_orig);
    for (int64_t i = 0; i < num_rows; ++i)
    {
        end_ch_[i] = num_chan;
    }

    // Determine effective size of sub-grids and w-tower height
    // (both depend critically on how much we want to "use" of the sub-grid).
    const int eff_sg_size = int(floor(subgrid_size * subgrid_frac));
    const double eff_sg_distance = eff_sg_size / theta;
    const double w_stack_distance = w_tower_height * w_step;

    // Determine (maximum) number of sub-grids and w-stacking planes needed
    const double eta = 1e-5;
    double uvw_min[3] = {0.0, 0.0, 0.0}, uvw_max[3] = {0.0, 0.0, 0.0};
    sdp_gridder_uvw_bounds_all(
            uvw, freq0_hz, dfreq_hz, start_ch_orig, end_ch_orig,
            uvw_min, uvw_max, status
    );
    int64_t min_iu = int64_t(floor(uvw_min[0] / eff_sg_distance + 0.5 - eta));
    int64_t max_iu = int64_t(floor(uvw_max[0] / eff_sg_distance + 0.5 + eta));
    int64_t min_iv = int64_t(floor(uvw_min[1] / eff_sg_distance + 0.5 - eta));
    int64_t max_iv = int64_t(floor(uvw_max[1] / eff_sg_distance + 0.5 + eta));
    int64_t min_iw = int64_t(floor(uvw_min[2] / w_stack_distance + 0.5 - eta));
    int64_t max_iw = int64_t(floor(uvw_max[2] / w_stack_distance + 0.5 + eta));

    // Create an iFFT plan for the whole grid / image.
    sdp_Fft* ifft = sdp_fft_create(image, image, 2, false, status);

    // Clear the output image.
    sdp_mem_clear_contents(image, status);

    // Loop over w-planes.
    for (int64_t iw = min_iw; iw <= max_iw; ++iw)
    {
        // Select visibilities on w-plane.
        sdp_Mem* start_ch_w = sdp_mem_create_copy(start_ch_orig, loc, status);
        sdp_Mem* end_ch_w = sdp_mem_create_copy(end_ch_orig, loc, status);
        double min_w = iw * w_stack_distance - w_stack_distance / 2;
        double max_w = (iw + 1) * w_stack_distance - w_stack_distance / 2;
        sdp_gridder_clamp_channels_single(uvw, 2, freq0_hz, dfreq_hz,
                start_ch_w, end_ch_w, min_w, max_w, status
        );
        double num_vis = 0;
        sdp_gridder_sum_diff(end_ch_w, start_ch_w, &num_vis, status);
        if (num_vis == 0)
        {
            sdp_mem_free(start_ch_w);
            sdp_mem_free(end_ch_w);
            continue;
        }

        // Allocate an empty grid for this plane.
        const int64_t grid_shape[] = {image_size, image_size};
        sdp_Mem* grid = sdp_mem_create(
                sdp_mem_type(vis), loc, 2, grid_shape, status
        );
        sdp_mem_clear_contents(grid, status);

        // Loop over sub-grid (towers) in u.
        #pragma omp parallel for
        for (int64_t iu = min_iu; iu <= max_iu; ++iu)
        {
            // Select visibilities in column.
            sdp_Mem* start_ch_u = sdp_mem_create_copy(start_ch_w, loc, status);
            sdp_Mem* end_ch_u = sdp_mem_create_copy(end_ch_w, loc, status);
            double min_u = iu * eff_sg_distance - eff_sg_distance / 2;
            double max_u = (iu + 1) * eff_sg_distance - eff_sg_distance / 2;
            sdp_gridder_clamp_channels_single(uvw, 0, freq0_hz, dfreq_hz,
                    start_ch_u, end_ch_u, min_u, max_u, status
            );
            double num_vis = 0;
            sdp_gridder_sum_diff(end_ch_u, start_ch_u, &num_vis, status);
            if (num_vis == 0)
            {
                sdp_mem_free(start_ch_u);
                sdp_mem_free(end_ch_u);
                continue;
            }

            // Allocate a sub-grid and prepare an FFT plan for it.
            const int64_t subgrid_shape[] = {subgrid_size, subgrid_size};
            sdp_Mem* subgrid_image = sdp_mem_create(
                    sdp_mem_type(grid), loc, 2, subgrid_shape, status
            );
            sdp_Fft* fft_subgrid = sdp_fft_create(
                    subgrid_image, subgrid_image, 2, true, status
            );
            const double sg_factor = pow(image_size / (double)subgrid_size, 2);

            // Loop over sub-grid (towers) in v.
            for (int64_t iv = min_iv; iv <= max_iv; ++iv)
            {
                // Select visibilities in sub-grid.
                sdp_Mem* start_ch_v = sdp_mem_create_copy(
                        start_ch_u, loc, status
                );
                sdp_Mem* end_ch_v = sdp_mem_create_copy(end_ch_u, loc, status);
                double min_v = iv * eff_sg_distance - eff_sg_distance / 2;
                double max_v = (iv + 1) * eff_sg_distance - eff_sg_distance / 2;
                sdp_gridder_clamp_channels_single(uvw, 1, freq0_hz, dfreq_hz,
                        start_ch_v, end_ch_v, min_v, max_v, status
                );
                double num_vis = 0;
                sdp_gridder_sum_diff(end_ch_v, start_ch_v, &num_vis, status);
                if (num_vis == 0)
                {
                    sdp_mem_free(start_ch_v);
                    sdp_mem_free(end_ch_v);
                    continue;
                }
                if (verbosity > 0)
                {
                    SDP_LOG_INFO("subgrid %d/%d/%d: %.0f visibilities",
                            iu, iv, iw, num_vis
                    );
                }

                // Grid visibilities.
                sdp_mem_clear_contents(subgrid_image, status);
                sdp_gridder_wtower_uvw_grid(kernel, vis, uvw,
                        start_ch_v, end_ch_v, freq0_hz, dfreq_hz, subgrid_image,
                        iu * eff_sg_size, iv * eff_sg_size, iw * w_tower_height,
                        status
                );

                // Add to grid.
                // grid += shift_grid(
                //     grid_pad(fft(subgrid_image), kernel.image_size),
                //     -iu * eff_sg_size, -iv * eff_sg_size
                // )
                sdp_fft_phase(subgrid_image, status);
                sdp_fft_exec(fft_subgrid, subgrid_image, subgrid_image, status);
                sdp_fft_phase(subgrid_image, status);
                #pragma omp critical
                sdp_gridder_subgrid_add(
                        grid, -iu * eff_sg_size, -iv * eff_sg_size,
                        subgrid_image, sg_factor, status
                );

                sdp_mem_free(start_ch_v);
                sdp_mem_free(end_ch_v);
            }
            sdp_fft_free(fft_subgrid);
            sdp_mem_free(subgrid_image);
            sdp_mem_free(start_ch_u);
            sdp_mem_free(end_ch_u);
        }

        // Do image correction / w-stacking.
        // image += kernel.grid_correct(ifft(grid), 0, 0, iw * w_tower_height)
        sdp_fft_phase(grid, status);
        sdp_fft_exec(ifft, grid, grid, status);
        sdp_fft_norm(grid, status); // To match numpy's ifft.
        sdp_fft_phase(grid, status);
        sdp_gridder_wtower_uvw_grid_correct(
                kernel, grid, 0, 0, iw * w_tower_height, status
        );
        sdp_gridder_accumulate_scaled_arrays(image, grid, NULL, 0, status);

        sdp_mem_free(grid);
        sdp_mem_free(start_ch_w);
        sdp_mem_free(end_ch_w);
    }
    sdp_fft_free(ifft);
    sdp_mem_free(start_ch_orig);
    sdp_mem_free(end_ch_orig);
    sdp_gridder_wtower_uvw_free(kernel);
}
