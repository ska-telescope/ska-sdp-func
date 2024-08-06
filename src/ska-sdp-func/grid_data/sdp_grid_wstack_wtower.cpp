/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/grid_data/sdp_grid_wstack_wtower.h"
#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"
#include "ska-sdp-func/utility/sdp_timer.h"


static void report_timing(
        const sdp_Mem* vis,
        const sdp_Mem* image,
        sdp_Timer* tmr_total,
        sdp_Fft* fft,
        int num_threads,
        sdp_GridderWtowerUVW** kernel,
        sdp_Fft** fft_subgrid,
        int grid
)
{
    double t_total = sdp_timer_elapsed(tmr_total);
    double t_fft_grid = sdp_fft_elapsed_time(fft, SDP_FFT_TMR_EXEC);
    double t_fft_subgrid = 0;
    double t_kernel_fft = 0;
    double t_kernel_grid = 0;
    double t_kernel_total = 0;
    double t_kernel_grid_correct = sdp_gridder_wtower_uvw_elapsed_time(
            kernel[0], SDP_WTOWER_TMR_GRID_CORRECT, grid
    );
    for (int i = 0; i < num_threads; ++i)
    {
        t_fft_subgrid += sdp_fft_elapsed_time(
                fft_subgrid[i], SDP_FFT_TMR_EXEC
        );
        t_kernel_fft += sdp_gridder_wtower_uvw_elapsed_time(
                kernel[i], SDP_WTOWER_TMR_FFT, grid
        );
        t_kernel_grid += sdp_gridder_wtower_uvw_elapsed_time(
                kernel[i], SDP_WTOWER_TMR_KERNEL, grid
        );
        t_kernel_total += sdp_gridder_wtower_uvw_elapsed_time(
                kernel[i], SDP_WTOWER_TMR_TOTAL, grid
        );
    }
    t_fft_subgrid /= num_threads;
    t_kernel_fft /= num_threads;
    t_kernel_grid /= num_threads;
    t_kernel_total /= num_threads;
    const char* name = grid ? "grid_all,  " : "degrid_all,";
    SDP_LOG_INFO("w-towers timing report");
    SDP_LOG_INFO("vis shape (rows, chans)  : (%d, %d)",
            sdp_mem_shape_dim(vis, 0), sdp_mem_shape_dim(vis, 1)
    );
    SDP_LOG_INFO("image size (pixels)      : (%d, %d)",
            sdp_mem_shape_dim(image, 0), sdp_mem_shape_dim(image, 1)
    );
    SDP_LOG_INFO("%s % 3d threads  : %.3f sec",
            name, num_threads, t_total
    );
    SDP_LOG_INFO("  + (de)grid correct     : %.3f sec (%.1f%%)",
            t_kernel_grid_correct, 100 * t_kernel_grid_correct / t_total
    );
    SDP_LOG_INFO("  + FFT(grid)            : %.3f sec (%.1f%%)",
            t_fft_grid, 100 * t_fft_grid / t_total
    );
    SDP_LOG_INFO("  + FFT(subgrid)         : %.3f sec (%.1f%%)",
            t_fft_subgrid, 100 * t_fft_subgrid / t_total
    );
    SDP_LOG_INFO("  + kernel               : %.3f sec (%.1f%%)",
            t_kernel_total, 100 * t_kernel_total / t_total
    );
    SDP_LOG_INFO("    - kernel FFT         : %.3f sec (%.1f%%; overall %.1f%%)",
            t_kernel_fft, 100 * t_kernel_fft / t_kernel_total,
            100 * t_kernel_fft / t_total
    );
    SDP_LOG_INFO("    - kernel (de)grid    : %.3f sec (%.1f%%; overall %.1f%%)",
            t_kernel_grid, 100 * t_kernel_grid / t_kernel_total,
            100 * t_kernel_grid / t_total
    );
}


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
    if (sdp_mem_num_dims(vis) != 2 || sdp_mem_num_dims(uvw) != 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Visibilities and (u,v,w)-coordinates must be 2D");
        return;
    }
    int num_threads = 1;
#ifdef _OPENMP
    omp_set_max_active_levels(1);
    num_threads = (loc == SDP_MEM_CPU) ? omp_get_max_threads() : 1;
#endif

    // Set up timers.
    const sdp_TimerType tmr_type = (
        loc == SDP_MEM_CPU ? SDP_TIMER_NATIVE : SDP_TIMER_CUDA
    );
    sdp_Timer* tmr_total = sdp_timer_create(tmr_type);
    sdp_timer_resume(tmr_total);

    // Create gridder kernels, sub-grids and sub-grid FFT plans, per thread.
    std::vector<sdp_GridderWtowerUVW*> kernel(num_threads);
    std::vector<sdp_Mem*> subgrid(num_threads);
    std::vector<sdp_Fft*> ifft_subgrid(num_threads);
    const int image_size = (int) sdp_mem_shape_dim(image, 0);
    const int64_t subgrid_shape[] = {subgrid_size, subgrid_size};
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i)
    {
        kernel[i] = sdp_gridder_wtower_uvw_create(
                image_size, subgrid_size, theta, w_step, shear_u, shear_v,
                support, oversampling, w_support, w_oversampling, status
        );
        subgrid[i] = sdp_mem_create(
                sdp_mem_type(vis), loc, 2, subgrid_shape, status
        );
        ifft_subgrid[i] = sdp_fft_create(
                subgrid[i], subgrid[i], 2, false, status
        );
    }

    // Assume we're using all visibilities.
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    const int64_t num_chan = sdp_mem_shape_dim(vis, 1);
    sdp_Mem* start_ch_orig = sdp_mem_create(
            SDP_MEM_INT, loc, 1, &num_rows, status
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
    if (loc == SDP_MEM_GPU)
    {
        sdp_Mem* tmp = sdp_mem_create_copy(end_ch_orig, loc, status);
        sdp_mem_free(end_ch_orig);
        end_ch_orig = tmp;
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
        sdp_Mem* grid = sdp_mem_create_copy(image, loc, status);
        sdp_gridder_wtower_uvw_degrid_correct(
                kernel[0], grid, 0, 0, iw * w_tower_height, status
        );
        sdp_fft_phase(grid, status);
        sdp_fft_exec(fft, grid, grid, status);
        sdp_fft_phase(grid, status);

        // Loop over sub-grid (towers) in u.
        #pragma omp parallel for num_threads(num_threads)
        for (int64_t iu = min_iu; iu <= max_iu; ++iu)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
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
                if (verbosity > 1)
                {
                    SDP_LOG_INFO("subgrid %d/%d/%d: %.0f visibilities",
                            iu, iv, iw, num_vis
                    );
                }

                // Prepare sub-grid.
                // subgrid = ifft(
                //     subgrid_cut_out(
                //         shift_grid(grid, iu * eff_sg_size, iv * eff_sg_size),
                //         kernel.subgrid_size
                //     )
                // )
                sdp_gridder_subgrid_cut_out(
                        grid, iu * eff_sg_size, iv * eff_sg_size,
                        subgrid[tid], status
                );
                sdp_fft_phase(subgrid[tid], status);
                sdp_fft_exec(
                        ifft_subgrid[tid], subgrid[tid], subgrid[tid], status
                );
                sdp_fft_norm(subgrid[tid], status); // To match numpy's ifft.
                sdp_fft_phase(subgrid[tid], status);

                // Degrid visibilities.
                sdp_gridder_wtower_uvw_degrid(kernel[tid], subgrid[tid],
                        iu * eff_sg_size, iv * eff_sg_size, iw * w_tower_height,
                        freq0_hz, dfreq_hz, uvw, start_ch_v, end_ch_v,
                        vis, status
                );
                sdp_mem_free(start_ch_v);
                sdp_mem_free(end_ch_v);
            }
            sdp_mem_free(start_ch_u);
            sdp_mem_free(end_ch_u);
        }
        sdp_mem_free(grid);
        sdp_mem_free(start_ch_w);
        sdp_mem_free(end_ch_w);
    }
    sdp_mem_free(start_ch_orig);
    sdp_mem_free(end_ch_orig);

    // Report timing.
    if (verbosity > 0)
    {
        report_timing(vis, image, tmr_total, fft,
                num_threads, &kernel[0], &ifft_subgrid[0], 0
        );
    }
    for (int i = 0; i < num_threads; ++i)
    {
        sdp_gridder_wtower_uvw_free(kernel[i]);
        sdp_mem_free(subgrid[i]);
        sdp_fft_free(ifft_subgrid[i]);
    }
    sdp_fft_free(fft);
    sdp_timer_free(tmr_total);
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
    if (sdp_mem_num_dims(vis) != 2 || sdp_mem_num_dims(uvw) != 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Visibilities and (u,v,w)-coordinates must be 2D");
        return;
    }
    int num_threads = 1;
#ifdef _OPENMP
    omp_set_max_active_levels(1);
    num_threads = (loc == SDP_MEM_CPU) ? omp_get_max_threads() : 1;
#endif

    // Set up timers.
    const sdp_TimerType tmr_type = (
        loc == SDP_MEM_CPU ? SDP_TIMER_NATIVE : SDP_TIMER_CUDA
    );
    sdp_Timer* tmr_total = sdp_timer_create(tmr_type);
    sdp_timer_resume(tmr_total);

    // Create gridder kernels, sub-grids and sub-grid FFT plans, per thread.
    std::vector<sdp_GridderWtowerUVW*> kernel(num_threads);
    std::vector<sdp_Mem*> subgrid(num_threads);
    std::vector<sdp_Fft*> fft_subgrid(num_threads);
    const int image_size = (int) sdp_mem_shape_dim(image, 0);
    const int64_t subgrid_shape[] = {subgrid_size, subgrid_size};
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i)
    {
        kernel[i] = sdp_gridder_wtower_uvw_create(
                image_size, subgrid_size, theta, w_step, shear_u, shear_v,
                support, oversampling, w_support, w_oversampling, status
        );
        subgrid[i] = sdp_mem_create(
                sdp_mem_type(vis), loc, 2, subgrid_shape, status
        );
        fft_subgrid[i] = sdp_fft_create(
                subgrid[i], subgrid[i], 2, true, status
        );
    }

    // Assume we're using all visibilities.
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    const int64_t num_chan = sdp_mem_shape_dim(vis, 1);
    sdp_Mem* start_ch_orig = sdp_mem_create(
            SDP_MEM_INT, loc, 1, &num_rows, status
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
    if (loc == SDP_MEM_GPU)
    {
        sdp_Mem* tmp = sdp_mem_create_copy(end_ch_orig, loc, status);
        sdp_mem_free(end_ch_orig);
        end_ch_orig = tmp;
    }

    // Determine effective size of sub-grids and w-tower height
    // (both depend critically on how much we want to "use" of the sub-grid).
    const int eff_sg_size = int(floor(subgrid_size * subgrid_frac));
    const double eff_sg_distance = eff_sg_size / theta;
    const double w_stack_distance = w_tower_height * w_step;
    const double sg_factor = pow(image_size / (double)subgrid_size, 2);

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
        #pragma omp parallel for num_threads(num_threads)
        for (int64_t iu = min_iu; iu <= max_iu; ++iu)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
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
                if (verbosity > 1)
                {
                    SDP_LOG_INFO("subgrid %d/%d/%d: %.0f visibilities",
                            iu, iv, iw, num_vis
                    );
                }

                // Grid visibilities.
                sdp_mem_clear_contents(subgrid[tid], status);
                sdp_gridder_wtower_uvw_grid(kernel[tid], vis, uvw,
                        start_ch_v, end_ch_v, freq0_hz, dfreq_hz, subgrid[tid],
                        iu * eff_sg_size, iv * eff_sg_size, iw * w_tower_height,
                        status
                );

                // Add to grid.
                // grid += shift_grid(
                //     grid_pad(fft(subgrid), kernel.image_size),
                //     -iu * eff_sg_size, -iv * eff_sg_size
                // )
                sdp_fft_phase(subgrid[tid], status);
                sdp_fft_exec(
                        fft_subgrid[tid], subgrid[tid], subgrid[tid], status
                );
                sdp_fft_phase(subgrid[tid], status);
                #pragma omp critical
                sdp_gridder_subgrid_add(
                        grid, -iu * eff_sg_size, -iv * eff_sg_size,
                        subgrid[tid], sg_factor, status
                );

                sdp_mem_free(start_ch_v);
                sdp_mem_free(end_ch_v);
            }
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
                kernel[0], grid, 0, 0, iw * w_tower_height, status
        );
        sdp_gridder_accumulate_scaled_arrays(image, grid, NULL, 0, status);

        sdp_mem_free(grid);
        sdp_mem_free(start_ch_w);
        sdp_mem_free(end_ch_w);
    }
    sdp_mem_free(start_ch_orig);
    sdp_mem_free(end_ch_orig);

    // Report timing.
    if (verbosity > 0)
    {
        report_timing(vis, image, tmr_total, ifft,
                num_threads, &kernel[0], &fft_subgrid[0], 1
        );
    }
    for (int i = 0; i < num_threads; ++i)
    {
        sdp_gridder_wtower_uvw_free(kernel[i]);
        sdp_mem_free(subgrid[i]);
        sdp_fft_free(fft_subgrid[i]);
    }
    sdp_fft_free(ifft);
    sdp_timer_free(tmr_total);
}
