/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <inttypes.h>
#include <vector>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/grid_data/sdp_grid_wstack_wtower.h"
#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"
#include "ska-sdp-func/utility/sdp_timer.h"

using std::vector;

static void report_timing(
        int num_w_planes,
        int num_subgrids_u,
        int num_subgrids_v,
        int image_size,
        int subgrid_size,
        double w_step,
        double w_tower_height,
        const sdp_Mem* vis,
        sdp_Timer* tmr_fft_grid,
        sdp_Timer* tmr_total,
        sdp_Timer* tmr_w_stack,
        sdp_Timer* tmr_zero,
        int num_threads,
        sdp_GridderWtowerUVW** kernel,
        int gridding,
        sdp_Error* status
);


static int64_t select_in_subgrid(
        const int iu,
        const int iv,
        const int iw,
        const sdp_Mem* uvw,
        const double freq0_hz,
        const double dfreq_hz,
        const double eff_sg_distance,
        const sdp_Mem* start_ch_w,
        const sdp_Mem* end_ch_w,
        sdp_Mem* start_ch_uv,
        sdp_Mem* end_ch_uv,
        int verbosity,
        sdp_Error* status
)
{
    int64_t num_vis = 0;
    const int64_t num_rows = sdp_mem_shape_dim(start_ch_uv, 0);
    sdp_mem_copy_contents(start_ch_uv, start_ch_w, 0, 0, num_rows, status);
    sdp_mem_copy_contents(end_ch_uv, end_ch_w, 0, 0, num_rows, status);
    double min_u = iu * eff_sg_distance - eff_sg_distance / 2;
    double max_u = (iu + 1) * eff_sg_distance - eff_sg_distance / 2;
    double min_v = iv * eff_sg_distance - eff_sg_distance / 2;
    double max_v = (iv + 1) * eff_sg_distance - eff_sg_distance / 2;
    sdp_gridder_clamp_channels_uv(uvw, freq0_hz, dfreq_hz, start_ch_uv,
            end_ch_uv, min_u, max_u, min_v, max_v, status
    );
    sdp_gridder_sum_diff(end_ch_uv, start_ch_uv, &num_vis, status);
    if (verbosity > 1 && num_vis > 0)
    {
        SDP_LOG_INFO("subgrid %d/%d/%d: %" PRId64 " visibilities",
                iu, iv, iw, num_vis
        );
    }
    return num_vis;
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
    if (w_tower_height == 0.0)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Automatic w-tower height not yet implemented");
        return;
    }
    if (subgrid_frac == 0.0) subgrid_frac = 2.0 / 3.0;
    int num_threads = 1;
#ifdef _OPENMP
    omp_set_max_active_levels(1);
    num_threads = (loc == SDP_MEM_CPU) ? omp_get_max_threads() : 1;
#endif

    // Set up timers.
    const sdp_TimerType tmr_type = (
        loc == SDP_MEM_CPU ? SDP_TIMER_NATIVE : SDP_TIMER_CUDA
    );
    sdp_Timer* tmr_fft_grid = sdp_timer_create(tmr_type);
    sdp_Timer* tmr_total = sdp_timer_create(tmr_type);
    sdp_Timer* tmr_w_stack = sdp_timer_create(tmr_type);
    sdp_Timer* tmr_zero = sdp_timer_create(tmr_type);
    sdp_timer_resume(tmr_total);

    // Assume we're using all visibilities.
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    sdp_Mem* start_ch = 0, * end_ch = 0, * start_ch_w = 0, * end_ch_w = 0;
    start_ch = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    end_ch = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    start_ch_w = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    end_ch_w = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    sdp_mem_set_value(start_ch, 0, status);
    sdp_mem_set_value(end_ch, (int) sdp_mem_shape_dim(vis, 1), status);

    // Create gridder kernels, sub-grids and sub-grid FFT plans, per thread.
    vector<sdp_GridderWtowerUVW*> kernel(num_threads, 0);
    vector<sdp_Mem*> subgrid(num_threads, 0);
    vector<sdp_Fft*> ifft_subgrid(num_threads, 0);
    vector<sdp_Mem*> start_ch_uv(num_threads, 0), end_ch_uv(num_threads, 0);
    const int image_size = (int) sdp_mem_shape_dim(image, 0);
    const int64_t grid_shape[] = {image_size, image_size};
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
        start_ch_uv[i] = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
        end_ch_uv[i] = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
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
            uvw, freq0_hz, dfreq_hz, start_ch, end_ch, uvw_min, uvw_max, status
    );
    int64_t min_iu = int64_t(floor(uvw_min[0] / eff_sg_distance + 0.5 - eta));
    int64_t max_iu = int64_t(floor(uvw_max[0] / eff_sg_distance + 0.5 + eta));
    int64_t min_iv = int64_t(floor(uvw_min[1] / eff_sg_distance + 0.5 - eta));
    int64_t max_iv = int64_t(floor(uvw_max[1] / eff_sg_distance + 0.5 + eta));
    int64_t min_iw = int64_t(floor(uvw_min[2] / w_stack_distance + 0.5 - eta));
    int64_t max_iw = int64_t(floor(uvw_max[2] / w_stack_distance + 0.5 + eta));
    const int num_w_planes = int(1 + max_iw - min_iw);
    const int num_subgrids_u = int(1 + max_iu - min_iu);
    const int num_subgrids_v = int(1 + max_iv - min_iv);
    const int num_subgrids = num_subgrids_u * num_subgrids_v;

    // Create an FFT plan for the whole grid / image.
    sdp_Mem* grid = sdp_mem_create(
            sdp_mem_type(vis), loc, 2, grid_shape, status
    );
    sdp_Fft* fft = sdp_fft_create(grid, grid, 2, true, status);

    // Clear output visibilities.
    sdp_timer_resume(tmr_zero);
    sdp_mem_set_value(vis, 0, status);
    sdp_timer_pause(tmr_zero);

    // Loop over w-planes.
    if (verbosity > 0)
    {
        SDP_LOG_INFO("using %d w-planes and %d sub-grids",
                num_w_planes, num_subgrids
        );
    }
    for (int64_t iw = min_iw; iw <= max_iw; ++iw)
    {
        if (*status) break;

        // Select visibilities on w-plane.
        sdp_mem_copy_contents(start_ch_w, start_ch, 0, 0, num_rows, status);
        sdp_mem_copy_contents(end_ch_w, end_ch, 0, 0, num_rows, status);
        double min_w = iw * w_stack_distance - w_stack_distance / 2;
        double max_w = (iw + 1) * w_stack_distance - w_stack_distance / 2;
        sdp_gridder_clamp_channels_single(uvw, 2, freq0_hz, dfreq_hz,
                start_ch_w, end_ch_w, min_w, max_w, status
        );
        int64_t num_vis = 0;
        sdp_gridder_sum_diff(end_ch_w, start_ch_w, &num_vis, status);
        if (num_vis == 0) continue;

        // Do image correction / w-stacking.
        sdp_timer_resume(tmr_zero);
        sdp_mem_set_value(grid, 0, status);
        sdp_timer_pause(tmr_zero);
        sdp_timer_resume(tmr_w_stack);
        sdp_gridder_accumulate_scaled_arrays(grid, image, NULL, 0, status);
        sdp_timer_pause(tmr_w_stack);
        sdp_gridder_wtower_uvw_degrid_correct(
                kernel[0], grid, 0, 0, iw * w_tower_height, status
        );
        sdp_timer_resume(tmr_fft_grid);
        sdp_fft_phase(grid, status);
        sdp_fft_exec(fft, grid, grid, status);
        sdp_fft_phase(grid, status);
        sdp_timer_pause(tmr_fft_grid);

        // Loop over sub-grid (towers) in u and v.
        #pragma \
        omp parallel for schedule(dynamic) num_threads(num_threads) collapse(2)
        for (int64_t iu = min_iu; iu <= max_iu; ++iu)
        {
            for (int64_t iv = min_iv; iv <= max_iv; ++iv)
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                // Select visibilities in sub-grid.
                int64_t num_vis = select_in_subgrid(iu, iv, iw, uvw, freq0_hz,
                        dfreq_hz, eff_sg_distance, start_ch_w, end_ch_w,
                        start_ch_uv[tid], end_ch_uv[tid], verbosity, status
                );
                if (num_vis == 0) continue;

                // Prepare sub-grid (cut-out, and inverse-FFT).
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

                // Degrid visibilities from sub-grid.
                sdp_gridder_wtower_uvw_degrid(kernel[tid], subgrid[tid],
                        iu * eff_sg_size, iv * eff_sg_size, iw * w_tower_height,
                        freq0_hz, dfreq_hz, uvw,
                        start_ch_uv[tid], end_ch_uv[tid], vis, status
                );
            }
        }
    }
    sdp_mem_free(grid);
    sdp_mem_free(start_ch);
    sdp_mem_free(end_ch);
    sdp_mem_free(start_ch_w);
    sdp_mem_free(end_ch_w);

    // Report timing.
    if (verbosity > 0)
    {
        report_timing(num_w_planes, num_subgrids_u, num_subgrids_v,
                image_size, subgrid_size, w_step, w_tower_height,
                vis, tmr_fft_grid, tmr_total, tmr_w_stack, tmr_zero,
                num_threads, &kernel[0], 0, status
        );
    }
    for (int i = 0; i < num_threads; ++i)
    {
        sdp_gridder_wtower_uvw_free(kernel[i]);
        sdp_mem_free(subgrid[i]);
        sdp_fft_free(ifft_subgrid[i]);
        sdp_mem_free(start_ch_uv[i]);
        sdp_mem_free(end_ch_uv[i]);
    }
    sdp_fft_free(fft);
    sdp_timer_free(tmr_fft_grid);
    sdp_timer_free(tmr_total);
    sdp_timer_free(tmr_w_stack);
    sdp_timer_free(tmr_zero);
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
    if (w_tower_height == 0.0)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Automatic w-tower height not yet implemented");
        return;
    }
    if (subgrid_frac == 0.0) subgrid_frac = 2.0 / 3.0;
    int num_threads = 1;
#ifdef _OPENMP
    omp_set_max_active_levels(1);
    num_threads = (loc == SDP_MEM_CPU) ? omp_get_max_threads() : 1;
#endif

    // Set up timers.
    const sdp_TimerType tmr_type = (
        loc == SDP_MEM_CPU ? SDP_TIMER_NATIVE : SDP_TIMER_CUDA
    );
    sdp_Timer* tmr_fft_grid = sdp_timer_create(tmr_type);
    sdp_Timer* tmr_total = sdp_timer_create(tmr_type);
    sdp_Timer* tmr_w_stack = sdp_timer_create(tmr_type);
    sdp_Timer* tmr_zero = sdp_timer_create(tmr_type);
    sdp_timer_resume(tmr_total);

    // Assume we're using all visibilities.
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    sdp_Mem* start_ch = 0, * end_ch = 0, * start_ch_w = 0, * end_ch_w = 0;
    start_ch = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    end_ch = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    start_ch_w = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    end_ch_w = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    sdp_mem_set_value(start_ch, 0, status);
    sdp_mem_set_value(end_ch, (int) sdp_mem_shape_dim(vis, 1), status);

    // Create gridder kernels, sub-grids and sub-grid FFT plans, per thread.
    vector<sdp_GridderWtowerUVW*> kernel(num_threads, 0);
    vector<sdp_Mem*> subgrid(num_threads, 0);
    vector<sdp_Fft*> fft_subgrid(num_threads, 0);
    vector<sdp_Mem*> start_ch_uv(num_threads, 0), end_ch_uv(num_threads, 0);
    const int image_size = (int) sdp_mem_shape_dim(image, 0);
    const int64_t grid_shape[] = {image_size, image_size};
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
        start_ch_uv[i] = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
        end_ch_uv[i] = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
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
            uvw, freq0_hz, dfreq_hz, start_ch, end_ch, uvw_min, uvw_max, status
    );
    int64_t min_iu = int64_t(floor(uvw_min[0] / eff_sg_distance + 0.5 - eta));
    int64_t max_iu = int64_t(floor(uvw_max[0] / eff_sg_distance + 0.5 + eta));
    int64_t min_iv = int64_t(floor(uvw_min[1] / eff_sg_distance + 0.5 - eta));
    int64_t max_iv = int64_t(floor(uvw_max[1] / eff_sg_distance + 0.5 + eta));
    int64_t min_iw = int64_t(floor(uvw_min[2] / w_stack_distance + 0.5 - eta));
    int64_t max_iw = int64_t(floor(uvw_max[2] / w_stack_distance + 0.5 + eta));
    const int num_w_planes = int(1 + max_iw - min_iw);
    const int num_subgrids_u = int(1 + max_iu - min_iu);
    const int num_subgrids_v = int(1 + max_iv - min_iv);
    const int num_subgrids = num_subgrids_u * num_subgrids_v;

    // Create an iFFT plan for the whole grid / image.
    sdp_Mem* grid = sdp_mem_create(
            sdp_mem_type(vis), loc, 2, grid_shape, status
    );
    sdp_Fft* ifft = sdp_fft_create(grid, grid, 2, false, status);

    // Clear the output image.
    sdp_timer_resume(tmr_zero);
    sdp_mem_set_value(image, 0, status);
    sdp_timer_pause(tmr_zero);

    // Loop over w-planes.
    if (verbosity > 0)
    {
        SDP_LOG_INFO("using %d w-planes and %d sub-grids",
                num_w_planes, num_subgrids
        );
    }
    for (int64_t iw = min_iw; iw <= max_iw; ++iw)
    {
        if (*status) break;

        // Select visibilities on w-plane.
        sdp_mem_copy_contents(start_ch_w, start_ch, 0, 0, num_rows, status);
        sdp_mem_copy_contents(end_ch_w, end_ch, 0, 0, num_rows, status);
        double min_w = iw * w_stack_distance - w_stack_distance / 2;
        double max_w = (iw + 1) * w_stack_distance - w_stack_distance / 2;
        sdp_gridder_clamp_channels_single(uvw, 2, freq0_hz, dfreq_hz,
                start_ch_w, end_ch_w, min_w, max_w, status
        );
        int64_t num_vis = 0;
        sdp_gridder_sum_diff(end_ch_w, start_ch_w, &num_vis, status);
        if (num_vis == 0) continue;

        // Clear the grid for this w-plane.
        sdp_timer_resume(tmr_zero);
        sdp_mem_set_value(grid, 0, status);
        sdp_timer_pause(tmr_zero);

        // Loop over sub-grid (towers) in u and v.
        #pragma \
        omp parallel for schedule(dynamic) num_threads(num_threads) collapse(2)
        for (int64_t iu = min_iu; iu <= max_iu; ++iu)
        {
            for (int64_t iv = min_iv; iv <= max_iv; ++iv)
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                // Select visibilities in sub-grid.
                int64_t num_vis = select_in_subgrid(iu, iv, iw, uvw, freq0_hz,
                        dfreq_hz, eff_sg_distance, start_ch_w, end_ch_w,
                        start_ch_uv[tid], end_ch_uv[tid], verbosity, status
                );
                if (num_vis == 0) continue;

                // Grid onto sub-grid.
                sdp_mem_set_value(subgrid[tid], 0, status);
                sdp_gridder_wtower_uvw_grid(kernel[tid], vis, uvw,
                        start_ch_uv[tid], end_ch_uv[tid], freq0_hz, dfreq_hz,
                        subgrid[tid], iu * eff_sg_size, iv * eff_sg_size,
                        iw * w_tower_height, status
                );

                // FFT sub-grid, and add to grid.
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
            }
        }

        // Do image correction / w-stacking.
        // image += kernel.grid_correct(ifft(grid), 0, 0, iw * w_tower_height)
        sdp_timer_resume(tmr_fft_grid);
        sdp_fft_phase(grid, status);
        sdp_fft_exec(ifft, grid, grid, status);
        sdp_fft_norm(grid, status); // To match numpy's ifft.
        sdp_fft_phase(grid, status);
        sdp_timer_pause(tmr_fft_grid);
        sdp_gridder_wtower_uvw_grid_correct(
                kernel[0], grid, 0, 0, iw * w_tower_height, status
        );
        sdp_timer_resume(tmr_w_stack);
        sdp_gridder_accumulate_scaled_arrays(image, grid, NULL, 0, status);
        sdp_timer_pause(tmr_w_stack);
    }
    sdp_mem_free(grid);
    sdp_mem_free(start_ch);
    sdp_mem_free(end_ch);
    sdp_mem_free(start_ch_w);
    sdp_mem_free(end_ch_w);

    // Report timing.
    if (verbosity > 0)
    {
        report_timing(num_w_planes, num_subgrids_u, num_subgrids_v,
                image_size, subgrid_size, w_step, w_tower_height,
                vis, tmr_fft_grid, tmr_total, tmr_w_stack, tmr_zero,
                num_threads, &kernel[0], 1, status
        );
    }
    for (int i = 0; i < num_threads; ++i)
    {
        sdp_gridder_wtower_uvw_free(kernel[i]);
        sdp_mem_free(subgrid[i]);
        sdp_fft_free(fft_subgrid[i]);
        sdp_mem_free(start_ch_uv[i]);
        sdp_mem_free(end_ch_uv[i]);
    }
    sdp_fft_free(ifft);
    sdp_timer_free(tmr_fft_grid);
    sdp_timer_free(tmr_total);
    sdp_timer_free(tmr_w_stack);
    sdp_timer_free(tmr_zero);
}


static void report_timing(
        int num_w_planes,
        int num_subgrids_u,
        int num_subgrids_v,
        int image_size,
        int subgrid_size,
        double w_step,
        double w_tower_height,
        const sdp_Mem* vis,
        sdp_Timer* tmr_fft_grid,
        sdp_Timer* tmr_total,
        sdp_Timer* tmr_w_stack,
        sdp_Timer* tmr_zero,
        int num_threads,
        sdp_GridderWtowerUVW** kernel,
        int gridding,
        sdp_Error* status
)
{
    if (*status) return;
    int total_w_planes = 0;
    double t_total = sdp_timer_elapsed(tmr_total);
    double t_fft_grid = sdp_timer_elapsed(tmr_fft_grid);
    double t_w_stack = sdp_timer_elapsed(tmr_w_stack);
    double t_zeroing = sdp_timer_elapsed(tmr_zero);
    std::vector<double> t_kernel_totals(num_threads, 0.0);
    double t_kernel_grid_correct = sdp_gridder_wtower_uvw_elapsed_time(
            kernel[0], SDP_WTOWER_TMR_GRID_CORRECT, gridding
    );
    for (int i = 0; i < num_threads; ++i)
    {
        total_w_planes += sdp_gridder_wtower_uvw_num_w_planes(
                kernel[i], gridding
        );
        t_kernel_totals[i] = sdp_gridder_wtower_uvw_elapsed_time(
                kernel[i], SDP_WTOWER_TMR_PROCESS_SUBGRID_STACK, gridding
        );
    }
    std::sort(t_kernel_totals.begin(), t_kernel_totals.end());
    const double t_kernel_median = t_kernel_totals[num_threads / 2];
    const double t_kernel_min = t_kernel_totals[0];
    const double t_kernel_max = t_kernel_totals[num_threads - 1];
    const double t_kernel_q1 = t_kernel_totals[1 * num_threads / 4];
    const double t_kernel_q3 = t_kernel_totals[3 * num_threads / 4];
    const double t_kernel_iqr = t_kernel_q3 - t_kernel_q1;

    const char* name = gridding ? "Gridding,  " : "Degridding,";
    SDP_LOG_INFO("Timing report for w-stacking with w-towers");
    SDP_LOG_INFO("| Number of large w-planes  : %d", num_w_planes);
    SDP_LOG_INFO("| Number of small w-planes  : %d (in gridder kernel)",
            total_w_planes
    );
    SDP_LOG_INFO("| Number of sub-grids (u, v): (%d, %d)",
            num_subgrids_u, num_subgrids_v
    );
    SDP_LOG_INFO("| Image size (pixels)       : (%d, %d)",
            image_size, image_size
    );
    SDP_LOG_INFO("| Sub-grid size (pixels)    : (%d, %d)",
            subgrid_size, subgrid_size
    );
    SDP_LOG_INFO("| Vis shape (rows, chans)   : (%d, %d)",
            sdp_mem_shape_dim(vis, 0), sdp_mem_shape_dim(vis, 1)
    );
    SDP_LOG_INFO("| Value of w_step           : %.3e", w_step);
    SDP_LOG_INFO("| Value of w_tower_height   : %.3e", w_tower_height);
    SDP_LOG_INFO("| %s % 3d threads   : %.3f sec",
            name, num_threads, t_total
    );
    SDP_LOG_INFO("|   + (De)grid correct      : %.3f sec (%.1f%%)",
            t_kernel_grid_correct, 100 * t_kernel_grid_correct / t_total
    );
    SDP_LOG_INFO("|   + FFT(grid)             : %.3f sec (%.1f%%)",
            t_fft_grid, 100 * t_fft_grid / t_total
    );
    SDP_LOG_INFO("|   + W-stacking            : %.3f sec (%.1f%%)",
            t_w_stack, 100 * t_w_stack / t_total
    );
    SDP_LOG_INFO("|   + Zeroing arrays        : %.3f sec (%.1f%%)",
            t_zeroing, 100 * t_zeroing / t_total
    );
    SDP_LOG_INFO("|   + Process sub-grid stack thread distribution:");
    SDP_LOG_INFO("|     - Minimum             : %.3f sec (%.1f%%)",
            t_kernel_min, 100 * t_kernel_min / t_total
    );
    SDP_LOG_INFO("|     - Maximum             : %.3f sec (%.1f%%)",
            t_kernel_max, 100 * t_kernel_max / t_total
    );
    SDP_LOG_INFO("|     - Median              : %.3f sec (%.1f%%)",
            t_kernel_median, 100 * t_kernel_median / t_total
    );
    SDP_LOG_INFO("|     - Interquartile range : %.3f sec", t_kernel_iqr);
}
