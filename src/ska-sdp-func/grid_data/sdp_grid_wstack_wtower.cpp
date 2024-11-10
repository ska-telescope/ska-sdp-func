/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef _OPENMP
#include <omp.h>
#endif

#include <inttypes.h>
#include <vector>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/grid_data/sdp_grid_wstack_wtower.h"
#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"
#include "ska-sdp-func/utility/sdp_thread_support.h"
#include "ska-sdp-func/utility/sdp_timer.h"

using std::vector;

// Begin anonymous namespace for file-local functions.
namespace {

struct sdp_SubgridTask
{
    static const int64_t min_chunk_size = 2000;
    int64_t iu;
    int64_t iv;
    int64_t num_chunks;
    int64_t num_rows_per_chunk;
    int64_t num_rows_total;
    int64_t num_vis;
    int64_t row_start;
    int64_t row_end;

    sdp_SubgridTask() :
        iu(0), iv(0), num_chunks(0), num_rows_per_chunk(0),
        num_rows_total(0), num_vis(0), row_start(0), row_end(0)
    {
    }

    sdp_SubgridTask(
            int64_t iu,
            int64_t iv,
            int64_t num_vis,
            int64_t num_rows,
            int64_t num_vis_per_chunk
    ) : iu(iu), iv(iv), num_chunks(0), num_rows_per_chunk(0),
        num_rows_total(num_rows), num_vis(num_vis), row_start(0), row_end(0)
    {
        if (num_vis_per_chunk < min_chunk_size)
        {
            num_vis_per_chunk = min_chunk_size;
        }
        num_chunks = (num_vis + num_vis_per_chunk - 1) / num_vis_per_chunk;
        num_rows_per_chunk = num_rows_total;
        if (num_chunks > 0)
        {
            num_rows_per_chunk = (num_rows_total + num_chunks - 1) / num_chunks;
        }
    }
};


// Count visibilities in each sub-grid, create and return a list of tasks.
vector<sdp_SubgridTask> count_visibilities(
        const sdp_Mem* uvw,
        const sdp_Mem* vis,
        const sdp_Mem* start_ch_w,
        const sdp_Mem* end_ch_w,
        vector<sdp_Mem*>& start_ch_uv,
        vector<sdp_Mem*>& end_ch_uv,
        double freq0_hz,
        double dfreq_hz,
        double eff_sg_dist,
        int64_t min_iu,
        int64_t max_iu,
        int64_t min_iv,
        int64_t max_iv,
        int64_t iw,
        int verbosity,
        sdp_Timers* SDP_TMR_HANDLE,
        sdp_Error* status
)
{
    SDP_TMR_PUSH("Count visibilities");
    const int num_threads = (int) start_ch_uv.size();
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    const int64_t num_chan = sdp_mem_shape_dim(vis, 1);
    const int64_t num_vis_per_chunk = (
        (num_rows * num_chan) / (int64_t) num_threads
    );
    const int num_subgrids_u = int(1 + max_iu - min_iu);
    const int num_subgrids_v = int(1 + max_iv - min_iv);
    vector<sdp_SubgridTask> tasks(num_subgrids_u * num_subgrids_v);
    sdp_Mutex* mutex = (verbosity > 1) ? sdp_mutex_create() : 0;
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int64_t iu = min_iu; iu <= max_iu; ++iu)
    {
        for (int64_t iv = min_iv; iv <= max_iv; ++iv)
        {
            int tid = SDP_GET_THREAD_NUM;

            // Select visibilities in sub-grid.
            int64_t num_vis = 0;
            const double min_u = iu * eff_sg_dist - eff_sg_dist / 2;
            const double max_u = (iu + 1) * eff_sg_dist - eff_sg_dist / 2;
            const double min_v = iv * eff_sg_dist - eff_sg_dist / 2;
            const double max_v = (iv + 1) * eff_sg_dist - eff_sg_dist / 2;
            sdp_gridder_clamp_channels_uv(uvw, freq0_hz, dfreq_hz,
                    start_ch_w, end_ch_w, min_u, max_u, min_v, max_v,
                    start_ch_uv[tid], end_ch_uv[tid], -1, -1, status
            );
            sdp_gridder_sum_diff(end_ch_uv[tid], start_ch_uv[tid],
                    &num_vis, -1, -1, status
            );
            int64_t subgrid = (iu - min_iu) * num_subgrids_v + iv - min_iv;
            tasks[subgrid] = sdp_SubgridTask(
                    iu, iv, num_vis, num_rows, num_vis_per_chunk
            );
            if (verbosity > 1 && num_vis > 0)
            {
                sdp_mutex_lock(mutex);
                SDP_LOG_INFO(
                        "subgrid %d/%d/%d: "
                        "%" PRId64 " visibilities in %" PRId64 " chunks",
                        iu, iv, iw, num_vis, tasks[subgrid].num_chunks
                );
                sdp_mutex_unlock(mutex);
            }
        }
    }
    SDP_TMR_POP;
    sdp_mutex_free(mutex);
    return tasks;
}


// Pick the next task to work on from the list.
sdp_SubgridTask pick_task(vector<sdp_SubgridTask>& subgrids, sdp_Mutex* mutex)
{
    sdp_SubgridTask task;
    sdp_mutex_lock(mutex);
    for (size_t i = 0; i < subgrids.size(); ++i)
    {
        // Find a subgrid with work still to do.
        if (subgrids[i].num_chunks > 0)
        {
            // Set up the current task.
            task = subgrids[i];
            task.row_end = task.row_start + subgrids[i].num_rows_per_chunk;
            if (task.row_end > task.num_rows_total)
            {
                task.row_end = task.num_rows_total;
            }

            // Update the subgrid data for the next worker.
            subgrids[i].num_chunks--;
            subgrids[i].row_start += subgrids[i].num_rows_per_chunk;
            break;
        }
    }
    sdp_mutex_unlock(mutex);
    return task;
}


// Report run parameters and timing data.
void report_timing(
        const vector<sdp_GridderWtowerUVW*> kernel,
        int verbosity,
        int gridding,
        int num_w_planes,
        int num_subgrids_u,
        int num_subgrids_v,
        double w_tower_height,
        const sdp_Mem* vis,
        const sdp_Timers* SDP_TMR_HANDLE,
        sdp_Error* status
)
{
    if (*status || verbosity == 0) return;
    int total_w_planes = 0;
    for (size_t i = 0; i < kernel.size(); ++i)
    {
        total_w_planes += sdp_gridder_wtower_uvw_num_w_planes(
                kernel[i], gridding
        );
    }
    const int image_size = sdp_gridder_wtower_uvw_image_size(kernel[0]);
    const int subgrid_size = sdp_gridder_wtower_uvw_subgrid_size(kernel[0]);
    const double w_step = sdp_gridder_wtower_uvw_w_step(kernel[0]);
    SDP_LOG_INFO("Timing report for w-stacking with w-towers");
    SDP_LOG_INFO("| Number of large w-planes  : %d", num_w_planes);
    SDP_LOG_INFO("| Total w-layers processed  : %d (in gridder kernels)",
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
    SDP_TMR_REPORT;
}

} // End anonymous namespace for file-local functions.


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
        int num_threads,
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
    int max_threads = 1;
#ifdef _OPENMP
    omp_set_max_active_levels(1);
    max_threads = (loc == SDP_MEM_CPU) ? omp_get_max_threads() : 1;
#endif
    if (num_threads <= 0) num_threads = max_threads;

    // Set up timers.
    SDP_TMR_CREATE("Degridding",
            loc == SDP_MEM_CPU ? SDP_TIMER_NATIVE : SDP_TIMER_CUDA, num_threads
    );
    SDP_TMR_CREATE_SET("Process sub-grid stack", num_threads);

    // Assume we're using all visibilities.
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    const int64_t num_chan = sdp_mem_shape_dim(vis, 1);
    sdp_Mem* start_ch = 0, * end_ch = 0, * start_ch_w = 0, * end_ch_w = 0;
    start_ch = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    end_ch = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    start_ch_w = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    end_ch_w = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    sdp_mem_set_value(start_ch, 0, status);
    sdp_mem_set_value(end_ch, (int) num_chan, status);

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
    const double eff_sg_dist = eff_sg_size / theta;
    const double w_stack_dist = w_tower_height * w_step;

    // Determine (maximum) number of sub-grids and w-stacking planes needed.
    const double eta = 1e-5;
    double uvw_min[3] = {0.0, 0.0, 0.0}, uvw_max[3] = {0.0, 0.0, 0.0};
    sdp_gridder_uvw_bounds_all(
            uvw, freq0_hz, dfreq_hz, start_ch, end_ch, uvw_min, uvw_max, status
    );
    int64_t min_iu = int64_t(floor(uvw_min[0] / eff_sg_dist + 0.5 - eta));
    int64_t max_iu = int64_t(floor(uvw_max[0] / eff_sg_dist + 0.5 + eta));
    int64_t min_iv = int64_t(floor(uvw_min[1] / eff_sg_dist + 0.5 - eta));
    int64_t max_iv = int64_t(floor(uvw_max[1] / eff_sg_dist + 0.5 + eta));
    int64_t min_iw = int64_t(floor(uvw_min[2] / w_stack_dist + 0.5 - eta));
    int64_t max_iw = int64_t(floor(uvw_max[2] / w_stack_dist + 0.5 + eta));
    const int num_w_planes = int(1 + max_iw - min_iw);
    const int num_subgrids_u = int(1 + max_iu - min_iu);
    const int num_subgrids_v = int(1 + max_iv - min_iv);

    // Create an FFT plan for the whole grid / image.
    sdp_Mem* grid = sdp_mem_create(
            sdp_mem_type(vis), loc, 2, grid_shape, status
    );
    sdp_Fft* fft = sdp_fft_create(grid, grid, 2, true, status);

    // Clear output visibilities.
    SDP_TMR_PUSH("Zeroing arrays");
    sdp_mem_set_value(vis, 0, status);
    SDP_TMR_POP;

    // Loop over w-planes.
    if (verbosity > 0)
    {
        SDP_LOG_INFO("using %d w-planes and %d sub-grids",
                num_w_planes, num_subgrids_u * num_subgrids_v
        );
    }
    for (int64_t iw = min_iw; iw <= max_iw; ++iw)
    {
        if (*status) break;

        // Select visibilities on w-plane.
        double min_w = iw * w_stack_dist - w_stack_dist / 2;
        double max_w = (iw + 1) * w_stack_dist - w_stack_dist / 2;
        sdp_gridder_clamp_channels_single(uvw, 2, freq0_hz, dfreq_hz,
                start_ch, end_ch, min_w, max_w, start_ch_w, end_ch_w,
                -1, -1, status
        );
        int64_t num_vis = 0;
        sdp_gridder_sum_diff(end_ch_w, start_ch_w, &num_vis, -1, -1, status);
        if (num_vis == 0) continue;

        // Do image correction / w-stacking.
        SDP_TMR_PUSH("Zeroing arrays");
        sdp_mem_set_value(grid, 0, status);
        SDP_TMR_POP_PUSH("W-stacking");
        sdp_gridder_accumulate_scaled_arrays(grid, image, NULL, 0, status);
        SDP_TMR_POP_PUSH("Degrid correct");
        sdp_gridder_wtower_uvw_degrid_correct(
                kernel[0], grid, 0, 0, iw * w_tower_height, status
        );
        SDP_TMR_POP_PUSH("FFT(grid)");
        sdp_fft_exec_shift(fft, grid, 0, status);
        SDP_TMR_POP;

        // Count visibilities in each sub-grid and create tasks.
        vector<sdp_SubgridTask> tasks = count_visibilities(
                uvw, vis, start_ch_w, end_ch_w, start_ch_uv, end_ch_uv,
                freq0_hz, dfreq_hz, eff_sg_dist, min_iu, max_iu, min_iv, max_iv,
                iw, verbosity, SDP_TMR_HANDLE, status
        );

        // Start a parallel region to process all tasks.
        sdp_Mutex* mutex = sdp_mutex_create();
        int64_t vis_count_check = 0;
        #pragma omp parallel num_threads(num_threads)
        {
            for (;;)
            {
                if (*status) break;
                sdp_TimerNode* node = SDP_TMR_ROOT;
                int tid = SDP_GET_THREAD_NUM;

                // Pick up a task from the subgrid list.
                sdp_SubgridTask task = pick_task(tasks, mutex);
                if (task.num_chunks <= 0) break;

                // Select visibilities in sub-grid.
                int64_t num_vis = 0;
                const int64_t iu = task.iu;
                const int64_t iv = task.iv;
                const int64_t start_row = task.row_start;
                const int64_t end_row = task.row_end;
                const double min_u = iu * eff_sg_dist - eff_sg_dist / 2;
                const double max_u = (iu + 1) * eff_sg_dist - eff_sg_dist / 2;
                const double min_v = iv * eff_sg_dist - eff_sg_dist / 2;
                const double max_v = (iv + 1) * eff_sg_dist - eff_sg_dist / 2;
                sdp_gridder_clamp_channels_uv(uvw, freq0_hz, dfreq_hz,
                        start_ch_w, end_ch_w, min_u, max_u, min_v, max_v,
                        start_ch_uv[tid], end_ch_uv[tid],
                        start_row, end_row, status
                );
                sdp_gridder_sum_diff(end_ch_uv[tid], start_ch_uv[tid],
                        &num_vis, start_row, end_row, status
                );
                if (num_vis == 0) continue;
                #pragma omp atomic
                vis_count_check += num_vis;

                // Prepare sub-grid (cut-out, and inverse-FFT).
                sdp_gridder_subgrid_cut_out(
                        grid, iu * eff_sg_size, iv * eff_sg_size,
                        subgrid[tid], status
                );
                sdp_fft_exec_shift(ifft_subgrid[tid], subgrid[tid], 1, status);

                // Degrid visibilities from sub-grid.
                node = sdp_timers_push(
                        SDP_TMR_HANDLE, "Process sub-grid stack", tid, node
                );
                sdp_gridder_wtower_uvw_degrid(kernel[tid], subgrid[tid],
                        iu * eff_sg_size, iv * eff_sg_size, iw * w_tower_height,
                        freq0_hz, dfreq_hz, uvw, start_ch_uv[tid],
                        end_ch_uv[tid], vis, start_row, end_row, status
                );
                node = sdp_timers_pop(SDP_TMR_HANDLE, tid, node);
            }
        }
        sdp_mutex_free(mutex);
        if (vis_count_check != num_vis)
        {
            SDP_LOG_CRITICAL("Processed %d but expected %d visibilities",
                    (int) vis_count_check, (int) num_vis
            );
            exit(1);
        }
    }

    // Report timing.
    report_timing(
            kernel, verbosity, 0, num_w_planes, num_subgrids_u,
            num_subgrids_v, w_tower_height, vis, SDP_TMR_HANDLE, status
    );

    for (int i = 0; i < num_threads; ++i)
    {
        sdp_fft_free(ifft_subgrid[i]);
        sdp_gridder_wtower_uvw_free(kernel[i]);
        sdp_mem_free(subgrid[i]);
        sdp_mem_free(start_ch_uv[i]);
        sdp_mem_free(end_ch_uv[i]);
    }
    sdp_fft_free(fft);
    sdp_mem_free(grid);
    sdp_mem_free(start_ch);
    sdp_mem_free(end_ch);
    sdp_mem_free(start_ch_w);
    sdp_mem_free(end_ch_w);
    SDP_TMR_FREE;
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
        int num_threads,
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
    int max_threads = 1;
#ifdef _OPENMP
    omp_set_max_active_levels(1);
    max_threads = (loc == SDP_MEM_CPU) ? omp_get_max_threads() : 1;
#endif
    if (num_threads <= 0) num_threads = max_threads;

    // Set up timers.
    SDP_TMR_CREATE("Gridding",
            loc == SDP_MEM_CPU ? SDP_TIMER_NATIVE : SDP_TIMER_CUDA, num_threads
    );
    SDP_TMR_CREATE_SET("Process sub-grid stack", num_threads);

    // Assume we're using all visibilities.
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    const int64_t num_chan = sdp_mem_shape_dim(vis, 1);
    sdp_Mem* start_ch = 0, * end_ch = 0, * start_ch_w = 0, * end_ch_w = 0;
    start_ch = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    end_ch = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    start_ch_w = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    end_ch_w = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    sdp_mem_set_value(start_ch, 0, status);
    sdp_mem_set_value(end_ch, (int) num_chan, status);

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
    const double eff_sg_dist = eff_sg_size / theta;
    const double w_stack_dist = w_tower_height * w_step;
    const double sg_factor = pow(image_size / (double)subgrid_size, 2);

    // Determine (maximum) number of sub-grids and w-stacking planes needed
    const double eta = 1e-5;
    double uvw_min[3] = {0.0, 0.0, 0.0}, uvw_max[3] = {0.0, 0.0, 0.0};
    sdp_gridder_uvw_bounds_all(
            uvw, freq0_hz, dfreq_hz, start_ch, end_ch, uvw_min, uvw_max, status
    );
    int64_t min_iu = int64_t(floor(uvw_min[0] / eff_sg_dist + 0.5 - eta));
    int64_t max_iu = int64_t(floor(uvw_max[0] / eff_sg_dist + 0.5 + eta));
    int64_t min_iv = int64_t(floor(uvw_min[1] / eff_sg_dist + 0.5 - eta));
    int64_t max_iv = int64_t(floor(uvw_max[1] / eff_sg_dist + 0.5 + eta));
    int64_t min_iw = int64_t(floor(uvw_min[2] / w_stack_dist + 0.5 - eta));
    int64_t max_iw = int64_t(floor(uvw_max[2] / w_stack_dist + 0.5 + eta));
    const int num_w_planes = int(1 + max_iw - min_iw);
    const int num_subgrids_u = int(1 + max_iu - min_iu);
    const int num_subgrids_v = int(1 + max_iv - min_iv);

    // Create an iFFT plan for the whole grid / image.
    sdp_Mem* grid = sdp_mem_create(
            sdp_mem_type(vis), loc, 2, grid_shape, status
    );
    sdp_Fft* ifft = sdp_fft_create(grid, grid, 2, false, status);

    // Clear the output image.
    SDP_TMR_PUSH("Zeroing arrays");
    sdp_mem_set_value(image, 0, status);
    SDP_TMR_POP;

    // Loop over w-planes.
    if (verbosity > 0)
    {
        SDP_LOG_INFO("using %d w-planes and %d sub-grids",
                num_w_planes, num_subgrids_u * num_subgrids_v
        );
    }
    for (int64_t iw = min_iw; iw <= max_iw; ++iw)
    {
        if (*status) break;

        // Select visibilities on w-plane.
        double min_w = iw * w_stack_dist - w_stack_dist / 2;
        double max_w = (iw + 1) * w_stack_dist - w_stack_dist / 2;
        sdp_gridder_clamp_channels_single(uvw, 2, freq0_hz, dfreq_hz,
                start_ch, end_ch, min_w, max_w, start_ch_w, end_ch_w,
                -1, -1, status
        );
        int64_t num_vis = 0;
        sdp_gridder_sum_diff(end_ch_w, start_ch_w, &num_vis, -1, -1, status);
        if (num_vis == 0) continue;

        // Clear the grid for this w-plane.
        SDP_TMR_PUSH("Zeroing arrays");
        sdp_mem_set_value(grid, 0, status);
        SDP_TMR_POP;

        // Count visibilities in each sub-grid and create tasks.
        vector<sdp_SubgridTask> tasks = count_visibilities(
                uvw, vis, start_ch_w, end_ch_w, start_ch_uv, end_ch_uv,
                freq0_hz, dfreq_hz, eff_sg_dist, min_iu, max_iu, min_iv, max_iv,
                iw, verbosity, SDP_TMR_HANDLE, status
        );

        // Start a parallel region to process all tasks.
        sdp_Mutex* mutex = sdp_mutex_create();
        int64_t vis_count_check = 0;
        #pragma omp parallel num_threads(num_threads)
        {
            for (;;)
            {
                if (*status) break;
                sdp_TimerNode* node = SDP_TMR_ROOT;
                int tid = SDP_GET_THREAD_NUM;

                // Pick up a task from the subgrid list.
                sdp_SubgridTask task = pick_task(tasks, mutex);
                if (task.num_chunks <= 0) break;

                // Select visibilities in sub-grid.
                int64_t num_vis = 0;
                const int64_t iu = task.iu;
                const int64_t iv = task.iv;
                const int64_t start_row = task.row_start;
                const int64_t end_row = task.row_end;
                const double min_u = iu * eff_sg_dist - eff_sg_dist / 2;
                const double max_u = (iu + 1) * eff_sg_dist - eff_sg_dist / 2;
                const double min_v = iv * eff_sg_dist - eff_sg_dist / 2;
                const double max_v = (iv + 1) * eff_sg_dist - eff_sg_dist / 2;
                sdp_gridder_clamp_channels_uv(uvw, freq0_hz, dfreq_hz,
                        start_ch_w, end_ch_w, min_u, max_u, min_v, max_v,
                        start_ch_uv[tid], end_ch_uv[tid],
                        start_row, end_row, status
                );
                sdp_gridder_sum_diff(end_ch_uv[tid], start_ch_uv[tid],
                        &num_vis, start_row, end_row, status
                );
                if (num_vis == 0) continue;
                #pragma omp atomic
                vis_count_check += num_vis;

                // Grid onto sub-grid.
                sdp_mem_set_value(subgrid[tid], 0, status);
                node = sdp_timers_push(
                        SDP_TMR_HANDLE, "Process sub-grid stack", tid, node
                );
                sdp_gridder_wtower_uvw_grid(kernel[tid], vis, uvw,
                        start_ch_uv[tid], end_ch_uv[tid], freq0_hz, dfreq_hz,
                        subgrid[tid], iu * eff_sg_size, iv * eff_sg_size,
                        iw * w_tower_height, start_row, end_row, status
                );
                node = sdp_timers_pop(SDP_TMR_HANDLE, tid, node);

                // FFT sub-grid, and add to grid.
                sdp_fft_exec_shift(fft_subgrid[tid], subgrid[tid], 0, status);
                #pragma omp critical(subgrid_add)
                sdp_gridder_subgrid_add(
                        grid, -iu * eff_sg_size, -iv * eff_sg_size,
                        subgrid[tid], sg_factor, status
                );
            }
        }
        sdp_mutex_free(mutex);
        if (vis_count_check != num_vis)
        {
            SDP_LOG_CRITICAL("Processed %d but expected %d visibilities",
                    (int) vis_count_check, (int) num_vis
            );
            exit(1);
        }

        // Do image correction / w-stacking.
        // image += kernel.grid_correct(ifft(grid), 0, 0, iw * w_tower_height)
        SDP_TMR_PUSH("FFT(grid)");
        sdp_fft_exec_shift(ifft, grid, 1, status);
        SDP_TMR_POP_PUSH("Grid correct");
        sdp_gridder_wtower_uvw_grid_correct(
                kernel[0], grid, 0, 0, iw * w_tower_height, status
        );
        SDP_TMR_POP_PUSH("W-stacking");
        sdp_gridder_accumulate_scaled_arrays(image, grid, NULL, 0, status);
        SDP_TMR_POP;
    }

    // Report timing.
    report_timing(
            kernel, verbosity, 1, num_w_planes, num_subgrids_u,
            num_subgrids_v, w_tower_height, vis, SDP_TMR_HANDLE, status
    );

    for (int i = 0; i < num_threads; ++i)
    {
        sdp_fft_free(fft_subgrid[i]);
        sdp_gridder_wtower_uvw_free(kernel[i]);
        sdp_mem_free(subgrid[i]);
        sdp_mem_free(start_ch_uv[i]);
        sdp_mem_free(end_ch_uv[i]);
    }
    sdp_fft_free(ifft);
    sdp_mem_free(grid);
    sdp_mem_free(start_ch);
    sdp_mem_free(end_ch);
    sdp_mem_free(start_ch_w);
    sdp_mem_free(end_ch_w);
    SDP_TMR_FREE;
}
