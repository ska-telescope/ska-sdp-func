/* See the LICENSE file at the top-level directory of this distribution. */

#include <complex>
#include <cstdlib>
#include <cstring>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/grid_data/sdp_gridder_uvw_es_fft.h"
#include "ska-sdp-func/grid_data/sdp_gridder_uvw_es_fft_utils.h"
#include "ska-sdp-func/math/sdp_prefix_sum.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_timer.h"

struct sdp_GridderUvwEsFft
{
    double pixsize_x_rad;
    double pixsize_y_rad;
    double epsilon;
    bool do_wstacking;
    bool do_vis_count;
    int num_rows;
    int num_chan;
    int image_size;
    int grid_size;
    int support;
    double beta;
    float beta_f;

    double pixel_size;
    float pixel_size_f;

    double uv_scale;
    float uv_scale_f;

    double min_plane_w;
    double max_plane_w;
    double min_abs_w;
    double max_abs_w;
    int num_total_w_grids;
    double w_scale; // scaling factor for converting w coord to signed w grid index

    double inv_w_scale;
    double inv_w_range; // final scaling factor for scaling dirty image by w grid accumulation

    double conv_corr_norm_factor;

    float inv_w_scale_f;
    float inv_w_range_f;
    float w_scale_f;
    float min_plane_w_f;
    float max_plane_w_f;
    float conv_corr_norm_factor_f;

    // allocated memory
    sdp_Mem* w_grid_stack;
    sdp_Mem* quadrature_kernel;
    sdp_Mem* quadrature_nodes;
    sdp_Mem* quadrature_weights;
    sdp_Mem* conv_corr_kernel;
    sdp_Mem* vis_count;
    sdp_Mem* vis_counter;

    sdp_Timer* timer_overall;
    sdp_Timer* timer_fft;
    sdp_Timer* timer_grid;
};


static void report_info(
        sdp_GridderUvwEsFft* plan,
        const char* function_name
)
{
    sdp_log_message(
            SDP_LOG_LEVEL_INFO, stdout, function_name, FILENAME, __LINE__,
            " Sizes: image=(%d x %d), grid=(%d x %d x %d w-planes)",
            plan->image_size, plan->image_size,
            plan->grid_size, plan->grid_size, plan->num_total_w_grids
    );
    sdp_log_message(
            SDP_LOG_LEVEL_INFO, stdout, function_name, FILENAME, __LINE__,
            "Params: epsilon=%.2e, support=%d, beta=%.3f",
            plan->epsilon, plan->support, plan->beta
    );
    sdp_log_message(
            SDP_LOG_LEVEL_INFO, stdout, function_name, FILENAME, __LINE__,
            "Scales: uv_scale=%.3e, w_scale=%.3e, min_plane_w=%.3e",
            plan->uv_scale, plan->w_scale, plan->min_plane_w
    );
}


static void report_timings(
        sdp_GridderUvwEsFft* plan,
        const char* function_name
)
{
    const double tm_overall = sdp_timer_elapsed(plan->timer_overall);
    const double tm_fft = sdp_timer_elapsed(plan->timer_fft);
    const double tm_grid = sdp_timer_elapsed(plan->timer_grid);
    const double pc_fft = 100.0 * tm_fft / tm_overall;
    const double pc_grid = 100.0 * tm_grid / tm_overall;
    sdp_log_message(
            SDP_LOG_LEVEL_INFO, stdout, function_name, FILENAME, __LINE__,
            "In '%s' for %.3f s (%.1f%% gridding/degridding, %.1f%% FFT)",
            function_name, tm_overall, pc_grid, pc_fft
    );
}


void sdp_gridder_uvw_es_fft_free_plan(sdp_GridderUvwEsFft* plan)
{
    if (!plan) return;
    sdp_mem_free(plan->w_grid_stack);
    sdp_mem_free(plan->quadrature_kernel);
    sdp_mem_free(plan->quadrature_nodes);
    sdp_mem_free(plan->quadrature_weights);
    sdp_mem_free(plan->conv_corr_kernel);
    sdp_mem_free(plan->vis_count);
    sdp_mem_free(plan->vis_counter);
    sdp_timer_free(plan->timer_overall);
    sdp_timer_free(plan->timer_fft);
    sdp_timer_free(plan->timer_grid);
    free(plan);
}


void sdp_gridder_check_buffers(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        const sdp_Mem* dirty_image,
        bool do_degridding,
        sdp_Error* status
)
{
    // check location of parameters (CPU or GPU)
    const sdp_MemLocation location = sdp_mem_location(uvw);
    if (location != sdp_mem_location(freq_hz) ||
            location != sdp_mem_location(vis) ||
            location != sdp_mem_location(weight) ||
            location != sdp_mem_location(dirty_image))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch.");
        return;
    }

    // check types of parameters (real or complex)
    if (sdp_mem_is_complex(uvw))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("uvw values must be real.");
        return;
    }
    if (sdp_mem_is_complex(freq_hz))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Frequency values must be real.");
        return;
    }
    if (!sdp_mem_is_complex(vis))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Visibility values must be complex.");
        return;
    }
    if (sdp_mem_is_complex(weight))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Weight values must be real.");
        return;
    }
    if (sdp_mem_is_complex(dirty_image))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Dirty image must be real");
        return;
    }

    // check shapes of parameters
    const int64_t num_vis      = sdp_mem_shape_dim(vis, 0);
    const int64_t num_channels = sdp_mem_shape_dim(vis, 1);

    if (sdp_mem_shape_dim(uvw, 0) != num_vis)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("The number of rows in uvw and vis must match.");
        SDP_LOG_ERROR("uvw is %i by %i",
                sdp_mem_shape_dim(uvw, 0),
                sdp_mem_shape_dim(uvw, 1)
        );
        SDP_LOG_ERROR("vis is %i by %i", num_vis, num_channels);
        return;
    }
    if (sdp_mem_shape_dim(uvw, 1) != 3)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("uvw must be N x 3.");
        SDP_LOG_ERROR("uvw is %i by %i",
                sdp_mem_shape_dim(uvw, 0),
                sdp_mem_shape_dim(uvw, 1)
        );
        return;
    }
    if (sdp_mem_shape_dim(freq_hz, 0) != num_channels)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("The number of channels in vis and freq_hz must match.");
        SDP_LOG_ERROR("freq_hz is %i by %i",
                sdp_mem_shape_dim(freq_hz, 0),
                sdp_mem_shape_dim(freq_hz, 1)
        );
        SDP_LOG_ERROR("vis is %i by %i", num_vis, num_channels);
        return;
    }
    if (sdp_mem_shape_dim(weight, 0) != num_vis ||
            sdp_mem_shape_dim(weight, 1) != num_channels)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("weight and vis must be the same size.");
        SDP_LOG_ERROR("weight is %i by %i",
                sdp_mem_shape_dim(weight, 0),
                sdp_mem_shape_dim(weight, 1)
        );
        SDP_LOG_ERROR("vis is %i by %i", num_vis, num_channels);
        return;
    }

    if (sdp_mem_shape_dim(dirty_image, 0) != sdp_mem_shape_dim(dirty_image, 1))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Dirty image must be square.");
        SDP_LOG_ERROR("dirty_image is %i by %i",
                sdp_mem_shape_dim(dirty_image, 0),
                sdp_mem_shape_dim(dirty_image, 1)
        );
        return;
    }

    // check precision consistency
    if (sdp_mem_type(uvw) == SDP_MEM_DOUBLE)
    {
        if ((sdp_mem_type(freq_hz) != SDP_MEM_DOUBLE) ||
                (sdp_mem_type(vis) != SDP_MEM_COMPLEX_DOUBLE) ||
                (sdp_mem_type(weight) != SDP_MEM_DOUBLE) ||
                (sdp_mem_type(dirty_image) != SDP_MEM_DOUBLE))
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("All buffers must be the same precision.");
            return;
        }
    }
    else
    {
        if ((sdp_mem_type(freq_hz) != SDP_MEM_FLOAT) ||
                (sdp_mem_type(vis) != SDP_MEM_COMPLEX_FLOAT) ||
                (sdp_mem_type(weight) != SDP_MEM_FLOAT) ||
                (sdp_mem_type(dirty_image) != SDP_MEM_FLOAT))
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("All buffers must be the same precision.");
            return;
        }
    }

    // check contiguity
    if (!sdp_mem_is_c_contiguous(uvw) ||
            !sdp_mem_is_c_contiguous(freq_hz) ||
            !sdp_mem_is_c_contiguous(vis) ||
            !sdp_mem_is_c_contiguous(weight) ||
            !sdp_mem_is_c_contiguous(dirty_image))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("All input arrays must be C contiguous");
        return;
    }

    // check output is writeable
    if (do_degridding)
    {
        if (sdp_mem_is_read_only(vis))
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("Visibility data must be writable.");
            return;
        }
    }
    else
    {
        if (sdp_mem_is_read_only(dirty_image))
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            SDP_LOG_ERROR("Dirty image must be writable.");
            return;
        }
    }
}


void sdp_gridder_check_parameters(
        const double pixsize_x_rad,
        const double pixsize_y_rad,
        sdp_Error* status
)
{
    if (pixsize_x_rad != pixsize_y_rad)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Only square images supported, so pixsize_x_rad and "
                "pixsize_y_rad must be equal."
        );
        SDP_LOG_ERROR("pixsize_x_rad is %.12e", pixsize_x_rad);
        SDP_LOG_ERROR("pixsize_y_rad is %.12e", pixsize_y_rad);
        return;
    }
}


void sdp_gridder_check_plan(
        sdp_GridderUvwEsFft* plan,
        sdp_Error* status
)
{
    sdp_gridder_check_parameters(plan->pixsize_x_rad,
            plan->pixsize_y_rad, status
    );
}


sdp_GridderUvwEsFft* sdp_gridder_uvw_es_fft_create_plan(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        const sdp_Mem* dirty_image,
        const double pixsize_x_rad,
        const double pixsize_y_rad,
        const double epsilon,
        const double min_abs_w,
        const double max_abs_w,
        const int do_wstacking,
        sdp_Error* status
)
{
    if (*status) return NULL;

    sdp_gridder_check_parameters(pixsize_x_rad, pixsize_y_rad, status);
    if (*status) return NULL;

    sdp_gridder_check_buffers(
            uvw, freq_hz, vis, weight, dirty_image, false, status
    );
    if (*status) return NULL;

    sdp_GridderUvwEsFft* plan = (sdp_GridderUvwEsFft*) calloc(1,
            sizeof(sdp_GridderUvwEsFft)
    );

    plan->timer_overall = sdp_timer_create(SDP_TIMER_CUDA);
    plan->timer_fft = sdp_timer_create(SDP_TIMER_CUDA);
    plan->timer_grid = sdp_timer_create(SDP_TIMER_CUDA);
    plan->pixsize_x_rad = pixsize_x_rad;
    plan->pixsize_y_rad = pixsize_y_rad;
    plan->pixel_size = pixsize_x_rad;  // only square pixels supported
    plan->pixel_size_f = (float) plan->pixel_size;
    plan->epsilon = epsilon;
    plan->do_wstacking = do_wstacking;
    plan->num_rows = sdp_mem_shape_dim(vis, 0);
    plan->num_chan = sdp_mem_shape_dim(vis, 1);
    plan->image_size = sdp_mem_shape_dim(dirty_image, 0);
    plan->beta = NAN;

    const sdp_MemType vis_type = sdp_mem_type(vis);
    const int dbl_vis = (vis_type & SDP_MEM_DOUBLE);
    const int vis_precision = (vis_type & SDP_MEM_DOUBLE) ?
                SDP_MEM_DOUBLE : SDP_MEM_FLOAT;

    sdp_calculate_params_from_epsilon(plan->epsilon, plan->image_size,
            vis_precision, plan->grid_size, plan->support, plan->beta, status
    );
    if (*status)
    {
        sdp_gridder_uvw_es_fft_free_plan(plan);
        return NULL;
    }

    plan->beta *= plan->support;
    plan->beta_f = (float) plan->beta;
    plan->uv_scale = plan->grid_size * plan->pixel_size;
    plan->uv_scale_f = (float) plan->uv_scale;

    // do_wstacking-dependent parameters
    if (plan->do_wstacking)
    {
        // Determine number of w-planes required.
        const double x0 = -0.5 * plan->image_size * plan->pixel_size;
        const double y0 = -0.5 * plan->image_size * plan->pixel_size;
        double nmin = sqrt(std::max(1.0 - x0 * x0 - y0 * y0, 0.0)) - 1.0;
        if (x0 * x0 + y0 * y0 > 1.0)
        {
            nmin = -sqrt(fabs(1.0 - x0 * x0 - y0 * y0)) - 1.0;
        }
        double w_scale = 0.25 / fabs(nmin); // scaling factor for converting w coord to signed w grid index
        int num_total_w_grids = (max_abs_w - min_abs_w) / w_scale + 2; // number of w grids required
        w_scale = 1.0 / ((1.0 + 1e-13) * (max_abs_w - min_abs_w) /
                (num_total_w_grids - 1));
        const double min_plane_w =
                min_abs_w - (0.5 * plan->support - 1.0) / w_scale;
        const double max_plane_w =
                max_abs_w + (0.5 * plan->support - 1.0) / w_scale;
        num_total_w_grids += plan->support - 2;

        plan->min_abs_w = min_abs_w;
        plan->max_abs_w = max_abs_w;
        plan->min_plane_w = min_plane_w;
        plan->max_plane_w = max_plane_w;
        plan->num_total_w_grids = num_total_w_grids;
        plan->w_scale = w_scale;
        plan->inv_w_range = (plan->max_plane_w - plan->min_plane_w);
    }
    else
    {
        plan->min_abs_w = 0.0;
        plan->max_abs_w = 0.0;
        plan->min_plane_w = 0.0;
        plan->max_plane_w = 0.0;
        plan->num_total_w_grids = 1;
        plan->inv_w_range = 1.0;
        plan->w_scale = 1.0;
    }

    plan->inv_w_scale = 1.0 / plan->w_scale;
    plan->inv_w_scale_f = (float) plan->inv_w_scale;
    plan->w_scale_f     = (float) plan->w_scale;
    plan->min_plane_w_f = (float) plan->min_plane_w;
    plan->max_plane_w_f = (float) plan->max_plane_w;
    plan->inv_w_range_f = (float) plan->inv_w_range;

    sdp_gridder_check_plan(plan, status);
    if (*status)
    {
        sdp_gridder_uvw_es_fft_free_plan(plan);
        return NULL;
    }

    // Generate Gauss Legendre kernel for convolution correction.
    double* quadrature_kernel =
            (double*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(double));
    double* quadrature_nodes =
            (double*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(double));
    double* quadrature_weights =
            (double*) calloc(QUADRATURE_SUPPORT_BOUND, sizeof(double));
    double* conv_corr_kernel =
            (double*) calloc(plan->image_size / 2 + 1, sizeof(double));
    sdp_generate_gauss_legendre_conv_kernel(
            plan->image_size, plan->grid_size, plan->support, plan->beta,
            quadrature_kernel, quadrature_nodes, quadrature_weights,
            conv_corr_kernel
    );

    // Need to determine normalisation factor for scaling runtime calculated
    // conv correction values for coordinate n (where n = sqrt(1 - l^2 - m^2) - 1)
    uint32_t p = (uint32_t)(int(1.5 * plan->support + 2.0));
    plan->conv_corr_norm_factor = 0.0;
    for (uint32_t i = 0; i < p; i++)
    {
        plan->conv_corr_norm_factor += quadrature_kernel[i] *
                quadrature_weights[i];
    }
    plan->conv_corr_norm_factor *= (double)plan->support;
    plan->conv_corr_norm_factor_f = (float)plan->conv_corr_norm_factor;

    // create (temp) CPU buffers
    const int64_t qsb[] = {QUADRATURE_SUPPORT_BOUND};
    // const int64_t stride[] = {1};
    const int64_t len[] = {plan->image_size / 2 + 1};

    sdp_MemType mem_type = dbl_vis ? SDP_MEM_DOUBLE : SDP_MEM_FLOAT;
    sdp_Mem* m_quadrature_kernel =
            sdp_mem_create(mem_type, SDP_MEM_CPU, 1, qsb, status);
    sdp_Mem* m_quadrature_nodes =
            sdp_mem_create(mem_type, SDP_MEM_CPU, 1, qsb, status);
    sdp_Mem* m_quadrature_weights =
            sdp_mem_create(mem_type, SDP_MEM_CPU, 1, qsb, status);
    sdp_Mem* m_conv_corr_kernel =
            sdp_mem_create(mem_type, SDP_MEM_CPU, 1, len, status);

    if (dbl_vis)
    {
        // just copy
        double* p_quadrature_kernel =
                (double*)sdp_mem_data(m_quadrature_kernel);
        double* p_quadrature_nodes =
                (double*)sdp_mem_data(m_quadrature_nodes);
        double* p_quadrature_weights =
                (double*)sdp_mem_data(m_quadrature_weights);
        double* p_conv_corr_kernel =
                (double*)sdp_mem_data(m_conv_corr_kernel);

        for (int i = 0; i < QUADRATURE_SUPPORT_BOUND; ++i)
        {
            p_quadrature_kernel[ i] = (double)(quadrature_kernel[ i]);
            p_quadrature_nodes[  i] = (double)(quadrature_nodes[  i]);
            p_quadrature_weights[i] = (double)(quadrature_weights[i]);
        }
        for (int i = 0; i < plan->image_size / 2 + 1; ++i)
        {
            p_conv_corr_kernel[i] = (double)(conv_corr_kernel[i]);
        }
    }
    else
    {
        // Cast to float.
        float* p_quadrature_kernel = (float*)sdp_mem_data(m_quadrature_kernel);
        float* p_quadrature_nodes = (float*)sdp_mem_data(m_quadrature_nodes);
        float* p_quadrature_weights =
                (float*)sdp_mem_data(m_quadrature_weights);
        float* p_conv_corr_kernel = (float*)sdp_mem_data(m_conv_corr_kernel);

        for (int i = 0; i < QUADRATURE_SUPPORT_BOUND; ++i)
        {
            p_quadrature_kernel[ i] = (float)(quadrature_kernel[ i]);
            p_quadrature_nodes[  i] = (float)(quadrature_nodes[  i]);
            p_quadrature_weights[i] = (float)(quadrature_weights[i]);
        }
        for (int i = 0; i < plan->image_size / 2 + 1; ++i)
        {
            p_conv_corr_kernel[i] = (float)(conv_corr_kernel[i]);
        }
    }

    free(quadrature_kernel);
    free(quadrature_nodes);
    free(quadrature_weights);
    free(conv_corr_kernel);

    // Copy arrays to GPU.
    plan->quadrature_kernel = sdp_mem_create_copy(m_quadrature_kernel,
            SDP_MEM_GPU, status
    );
    plan->quadrature_nodes = sdp_mem_create_copy(m_quadrature_nodes,
            SDP_MEM_GPU, status
    );
    plan->quadrature_weights = sdp_mem_create_copy(m_quadrature_weights,
            SDP_MEM_GPU, status
    );
    plan->conv_corr_kernel = sdp_mem_create_copy(m_conv_corr_kernel,
            SDP_MEM_GPU, status
    );

    sdp_mem_free(m_quadrature_kernel);
    sdp_mem_free(m_quadrature_nodes);
    sdp_mem_free(m_quadrature_weights);
    sdp_mem_free(m_conv_corr_kernel);

    // Visibility counter.
    const char* env_counter = getenv("SDP_GRIDDER_COUNT");
    plan->do_vis_count = false;
    if (env_counter)
    {
        plan->do_vis_count = !strncmp(env_counter, "1", 1) ||
                !strncmp(env_counter, "t", 1) || !strncmp(env_counter, "T", 1);
    }
    int64_t vis_count_size[] = {1};
    plan->vis_count = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_GPU, 1, vis_count_size, status
    );
    plan->vis_counter = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_GPU, 1, vis_count_size, status
    );

    // allocate memory
    int64_t w_grid_stack_shape[] = {plan->grid_size, plan->grid_size};
    plan->w_grid_stack = sdp_mem_create(
            vis_type, SDP_MEM_GPU, 2, w_grid_stack_shape, status
    );
    if (*status)
    {
        sdp_gridder_uvw_es_fft_free_plan(plan);
        return NULL;
    }
    sdp_gridder_check_buffers(
            uvw, freq_hz, vis, weight, dirty_image, false, status
    );
    if (*status)
    {
        sdp_gridder_uvw_es_fft_free_plan(plan);
        return NULL;
    }

    return plan;
}


void sdp_grid_uvw_es_fft(
        sdp_GridderUvwEsFft* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        sdp_Mem* dirty_image,
        sdp_Error* status
)
{
    SDP_LOG_DEBUG("Executing %s...", __func__);
    if (*status || !plan) return;

    // Report plan info.
    report_info(plan, __func__);

    sdp_timer_start(plan->timer_overall);
    sdp_timer_reset(plan->timer_fft);
    sdp_timer_reset(plan->timer_grid);
    sdp_gridder_check_plan(plan, status);
    if (*status) return;

    sdp_gridder_check_buffers(
            uvw, freq_hz, vis, weight, dirty_image, false, status
    );
    if (*status) return;

    const int npix_x = (int)sdp_mem_shape_dim(dirty_image, 0);
    const int npix_y = (int)sdp_mem_shape_dim(dirty_image, 1);  // this should be the same as checked by check_params()

    uint64_t num_threads[] = {128, 2, 1}, num_blocks[] = {1, 1, 1};

    const sdp_MemType vis_type = sdp_mem_type(vis);

    const int chunk_size = plan->num_rows;
    const sdp_MemType coord_type = sdp_mem_type(uvw);
    const int dbl_vis = (vis_type & SDP_MEM_DOUBLE);
    const int dbl_coord = (coord_type & SDP_MEM_DOUBLE);
    num_blocks[0] = (chunk_size + num_threads[0] - 1) / num_threads[0];
    num_blocks[1] = (plan->num_chan + num_threads[1] - 1) / num_threads[1];

    // Create the FFT plan.
    sdp_Fft* fft = sdp_fft_create(
            plan->w_grid_stack, plan->w_grid_stack, 2, 0, status
    );
    if (*status) return;

    // Define the tile size and number of tiles in each direction.
    // A tile consists of SHMSZ grid cells per thread in shared memory
    // and REGSZ grid cells per thread in registers.
    const int SHMSZ = 8;
    const int REGSZ = 8;
    const int grid_centre = plan->grid_size / 2;
    const int tile_size_v = 32;
    const int tile_size_u = (SHMSZ + REGSZ);
    const int num_tiles_u = (plan->grid_size + tile_size_u - 1) / tile_size_u;
    const int num_tiles_v = (plan->grid_size + tile_size_v - 1) / tile_size_v;
    const int num_tiles = num_tiles_u * num_tiles_v;

    // Which tile contains the grid centre?
    const int c_tile_u = grid_centre / tile_size_u;
    const int c_tile_v = grid_centre / tile_size_v;

    // Compute difference between centre of centre tile and grid centre
    // to ensure the centre of the grid is in the centre of a tile.
    const int top_left_u = grid_centre -
            c_tile_u * tile_size_u - tile_size_u / 2;
    const int top_left_v = grid_centre -
            c_tile_v * tile_size_v - tile_size_v / 2;

    // Set up scratch memory.
    const int64_t shape_num_points_in_tiles[] = {num_tiles + 1};
    const int64_t shape_num_vis[] = {1};
    sdp_Mem* num_points_in_tiles = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_GPU, 1, shape_num_points_in_tiles, status
    );
    sdp_Mem* tile_offsets = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_GPU, 1, shape_num_points_in_tiles, status
    );
    sdp_Mem* num_vis_mem = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_CPU, 1, shape_num_vis, status
    );
    sdp_mem_clear_contents(num_points_in_tiles, status);
    sdp_mem_clear_contents(tile_offsets, status);
    sdp_mem_clear_contents(num_vis_mem, status);

    // Count points in tiles.
    {
        const char* kernel_name = 0;
        if (dbl_coord)
        {
            kernel_name = "sdp_cuda_tile_count<double, double2, double3>";
        }
        else if (!dbl_coord)
        {
            kernel_name = "sdp_cuda_tile_count<float, float2, float3>";
        }
        if (kernel_name)
        {
            const void* args[] = {
                &plan->support,
                &chunk_size,
                &plan->num_chan,
                sdp_mem_gpu_buffer_const(freq_hz, status),
                sdp_mem_gpu_buffer_const(uvw, status),
                &plan->grid_size,
                dbl_coord ?
                    (const void*)&plan->uv_scale :
                    (const void*)&plan->uv_scale_f,
                &tile_size_u,
                &tile_size_v,
                &num_tiles_v,
                &top_left_u,
                &top_left_v,
                sdp_mem_gpu_buffer(num_points_in_tiles, status)
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, args, status
            );
        }
    }

#if 0
    sdp_Mem* cpu_num_pts_in_tiles = sdp_mem_create_copy(
            num_points_in_tiles, SDP_MEM_CPU, status
    );
    int* p = (int*)sdp_mem_data(cpu_num_pts_in_tiles);
    for (int j = 0; j < num_tiles_v; ++j)
    {
        for (int i = 0; i < num_tiles_u; ++i)
        {
            printf("%03d ", p[j * num_tiles_u + i]);
        }
        printf("\n");
    }
    sdp_mem_free(cpu_num_pts_in_tiles);
#endif

    // Get the offsets for each tile using prefix sum.
    sdp_prefix_sum(num_tiles, num_points_in_tiles, tile_offsets, status);

    // Get the total number of visibilities to process.
    sdp_mem_copy_contents(num_vis_mem, tile_offsets, 0, num_tiles, 1, status);
    int num_total = *((int*)sdp_mem_data(num_vis_mem));
    SDP_LOG_DEBUG("Number of visibilities to process: %d", num_total);

    // Bucket sort the data into tiles.
    const int64_t sorted_shape[] = {num_total};
    sdp_Mem* sorted_uu = sdp_mem_create(
            coord_type, SDP_MEM_GPU, 1, sorted_shape, status
    );
    sdp_Mem* sorted_vv = sdp_mem_create(
            coord_type, SDP_MEM_GPU, 1, sorted_shape, status
    );
    sdp_Mem* sorted_ww = sdp_mem_create(
            coord_type, SDP_MEM_GPU, 1, sorted_shape, status
    );
    sdp_Mem* sorted_vis = sdp_mem_create(
            vis_type, SDP_MEM_GPU, 1, sorted_shape, status
    );
    sdp_Mem* sorted_tile = sdp_mem_create(
            SDP_MEM_INT, SDP_MEM_GPU, 1, sorted_shape, status
    );
    {
        const char* kernel_name = 0;
        if (dbl_coord)
        {
            kernel_name = "sdp_cuda_tile_bucket_sort<double, double2, double3>";
        }
        else
        {
            kernel_name = "sdp_cuda_tile_bucket_sort<float, float2, float3>";
        }
        if (kernel_name)
        {
            const void* args[] = {
                &plan->support,
                &chunk_size,
                &plan->num_chan,
                sdp_mem_gpu_buffer_const(freq_hz, status),
                sdp_mem_gpu_buffer_const(uvw, status),
                sdp_mem_gpu_buffer_const(vis, status),
                sdp_mem_gpu_buffer_const(weight, status),
                &plan->grid_size,
                dbl_coord ?
                    (const void*)&plan->uv_scale :
                    (const void*)&plan->uv_scale_f,
                &tile_size_u,
                &tile_size_v,
                &num_tiles_v,
                &top_left_u,
                &top_left_v,
                sdp_mem_gpu_buffer(tile_offsets, status),
                sdp_mem_gpu_buffer(sorted_uu, status),
                sdp_mem_gpu_buffer(sorted_vv, status),
                sdp_mem_gpu_buffer(sorted_ww, status),
                sdp_mem_gpu_buffer(sorted_vis, status),
                sdp_mem_gpu_buffer(sorted_tile, status)
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, args, status
            );
        }
    }

    // Grid using w-stacking.
    sdp_mem_clear_contents(plan->vis_count, status);
    for (int grid_w = 0; grid_w < plan->num_total_w_grids; grid_w++)
    {
        sdp_mem_clear_contents(plan->w_grid_stack, status);
        if (*status) break;

        // Perform gridding
        sdp_mem_clear_contents(plan->vis_counter, status);
        sdp_timer_resume(plan->timer_grid);
        {
            const char* kernel_name = 0;
            if (plan->do_wstacking)
            {
                if (dbl_vis && dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_grid_tiled_3d"
                            "<double, double2, 8, 8>";
                }
                else if (!dbl_vis && !dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_grid_tiled_3d"
                            "<float, float2, 8, 8>";
                }
                void* null = 0;
                num_threads[0] = tile_size_v;
                num_threads[1] = 1;
                num_blocks[0] = (num_total + num_threads[0] - 1) / num_threads[0];
                num_blocks[1] = 1;
                if (num_blocks[0] > 10000) num_blocks[0] = 10000;
                const uint64_t sh_mem_size = sdp_mem_type_size(vis_type) *
                        SHMSZ * num_threads[0];

                const void* args[] = {
                    &plan->support,
                    &num_total,
                    sdp_mem_gpu_buffer(sorted_uu, status),
                    sdp_mem_gpu_buffer(sorted_vv, status),
                    sdp_mem_gpu_buffer(sorted_ww, status),
                    sdp_mem_gpu_buffer(sorted_vis, status),
                    sdp_mem_gpu_buffer(sorted_tile, status),
                    &plan->grid_size,
                    &grid_w,
                    &tile_size_u,
                    &tile_size_v,
                    &top_left_u,
                    &top_left_v,
                    dbl_vis ?
                        (const void*)&plan->beta :
                        (const void*)&plan->beta_f,
                    dbl_coord ?
                        (const void*)&plan->w_scale :
                        (const void*)&plan->w_scale_f,
                    dbl_coord ?
                        (const void*)&plan->min_plane_w :
                        (const void*)&plan->min_plane_w_f,
                    sdp_mem_gpu_buffer(plan->vis_counter, status),
                    sdp_mem_gpu_buffer(plan->w_grid_stack, status),
                    plan->do_vis_count ?
                        sdp_mem_gpu_buffer(plan->vis_count, status) : &null
                };
                sdp_launch_cuda_kernel(kernel_name,
                        num_blocks, num_threads, sh_mem_size, 0, args, status
                );
            }
            else
            {
                if (dbl_vis && dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_grid_2d"
                            "<double, double2, double, double2, double3>";
                }
                else if (!dbl_vis && !dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_grid_2d"
                            "<float, float2, float, float2, float3>";
                }
                void* null = 0;
                num_threads[0] = 1;
                num_threads[1] = 256;
                num_blocks[0] =
                        (plan->num_chan + num_threads[0] - 1) / num_threads[0];
                num_blocks[1] =
                        (chunk_size + num_threads[1] - 1) / num_threads[1];
                const void* args[] = {
                    &chunk_size,
                    &plan->num_chan,
                    sdp_mem_gpu_buffer_const(vis, status),
                    sdp_mem_gpu_buffer_const(weight, status),
                    sdp_mem_gpu_buffer_const(uvw, status),
                    sdp_mem_gpu_buffer_const(freq_hz, status),
                    sdp_mem_gpu_buffer(plan->w_grid_stack, status),
                    &plan->grid_size,
                    &grid_w,
                    &plan->support,
                    dbl_vis ?
                        (const void*)&plan->beta :
                        (const void*)&plan->beta_f,
                    dbl_coord ?
                        (const void*)&plan->uv_scale :
                        (const void*)&plan->uv_scale_f,
                    dbl_coord ?
                        (const void*)&plan->w_scale :
                        (const void*)&plan->w_scale_f,
                    dbl_coord ?
                        (const void*)&plan->min_plane_w :
                        (const void*)&plan->min_plane_w_f,
                    plan->do_vis_count ?
                        sdp_mem_gpu_buffer(plan->vis_count, status) : &null
                };
                sdp_launch_cuda_kernel(kernel_name,
                        num_blocks, num_threads, 0, 0, args, status
                );
            }
        }
        sdp_timer_pause(plan->timer_grid);

        // Perform 2D FFT
        sdp_timer_resume(plan->timer_fft);
        sdp_fft_exec(fft, plan->w_grid_stack, plan->w_grid_stack, status);
        sdp_timer_pause(plan->timer_fft);

        // Perform phase shift and sum into single real plane
        {
            const char* kernel_name = dbl_vis ?
                        "apply_w_screen_and_sum<double, double2>" :
                        "apply_w_screen_and_sum<float, float2>";
            num_threads[0] = std::min(32, (npix_x + 1) / 2);
            num_threads[1] = std::min(32, (npix_y + 1) / 2);
            // Allow extra in negative x quadrants, for asymmetric image centre
            num_blocks[0] =
                    (npix_x / 2 + 1 + num_threads[0] - 1) / num_threads[0];
            num_blocks[1] =
                    (npix_y / 2 + 1 + num_threads[1] - 1) / num_threads[1];
            const int num_w_grids_subset = 1;
            const bool do_FFT_shift = true;
            const void* args[] = {
                sdp_mem_gpu_buffer(dirty_image, status),
                &plan->image_size,
                dbl_vis ?
                    (const void*)&plan->pixel_size :
                    (const void*)&plan->pixel_size_f,
                sdp_mem_gpu_buffer_const(plan->w_grid_stack, status),
                &plan->grid_size,
                &grid_w,
                &num_w_grids_subset,
                dbl_vis ?
                    (const void*)&plan->inv_w_scale :
                    (const void*)&plan->inv_w_scale_f,
                dbl_vis ?
                    (const void*)&plan->min_plane_w :
                    (const void*)&plan->min_plane_w_f,
                &do_FFT_shift,
                &plan->do_wstacking
            };
            sdp_launch_cuda_kernel(
                    kernel_name, num_blocks, num_threads, 0, 0, args, status
            );
        }
    } // loop over w-planes.

    // Free FFT plan and data.
    sdp_fft_free(fft);

    // Perform convolution correction and final scaling on single real plane
    // note: can recycle same block/thread dims as w correction kernel
    {
        const char* kernel_name = dbl_vis ?
                    "conv_corr_and_scaling<double>" :
                    "conv_corr_and_scaling<float>";
        num_threads[0] = std::min(32, (npix_x + 1) / 2);
        num_threads[1] = std::min(32, (npix_y + 1) / 2);
        // Allow extra in negative x quadrants, for asymmetric image centre
        num_blocks[0] = (npix_x / 2 + 1 + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (npix_y / 2 + 1 + num_threads[1] - 1) / num_threads[1];
        const bool solving = true;
        const void* args[] = {
            sdp_mem_gpu_buffer(dirty_image, status),
            &plan->image_size,
            dbl_vis ?
                (const void*)&plan->pixel_size :
                (const void*)&plan->pixel_size_f,
            &plan->support,
            dbl_vis ?
                (const void*)&plan->conv_corr_norm_factor :
                (const void*)&plan->conv_corr_norm_factor_f,
            sdp_mem_gpu_buffer_const(plan->conv_corr_kernel, status),
            dbl_vis ?
                (const void*)&plan->inv_w_range :
                (const void*)&plan->inv_w_range_f,
            dbl_vis ?
                (const void*)&plan->inv_w_scale :
                (const void*)&plan->inv_w_scale_f,
            sdp_mem_gpu_buffer_const(plan->quadrature_kernel,  status),
            sdp_mem_gpu_buffer_const(plan->quadrature_nodes,   status),
            sdp_mem_gpu_buffer_const(plan->quadrature_weights, status),
            &solving,
            &plan->do_wstacking
        };
        sdp_launch_cuda_kernel(
                kernel_name, num_blocks, num_threads, 0, 0, args, status
        );
    }
    sdp_timer_pause(plan->timer_overall);

    // Free scratch arrays.
    sdp_mem_free(num_points_in_tiles);
    sdp_mem_free(tile_offsets);
    sdp_mem_free(num_vis_mem);
    sdp_mem_free(sorted_tile);
    sdp_mem_free(sorted_vis);
    sdp_mem_free(sorted_uu);
    sdp_mem_free(sorted_vv);
    sdp_mem_free(sorted_ww);

    // Report visibility count.
    sdp_Mem* vis_count_cpu = sdp_mem_create_copy(
            plan->vis_count, SDP_MEM_CPU, status
    );
    const int count_original = plan->num_rows * plan->num_chan;
    const int count_gridded = *(int*)sdp_mem_data(vis_count_cpu);
    const double ratio = (double)count_gridded / (double)count_original;
    sdp_mem_free(vis_count_cpu);
    if (count_gridded)
    {
        SDP_LOG_INFO("Count: original=%d, gridded=%d, ratio=%.2f",
                count_original, count_gridded, ratio
        );
    }

    // Report timings.
    report_timings(plan, __func__);
}


void sdp_ifft_degrid_uvw_es(
        sdp_GridderUvwEsFft* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        sdp_Mem* vis,
        const sdp_Mem* weight,
        sdp_Mem* dirty_image,   // even though this is an input, it is modified in place so can't be constant
        sdp_Error* status
)
{
    SDP_LOG_DEBUG("Executing %s...", __func__);
    if (*status || !plan) return;

    // Report plan info.
    report_info(plan, __func__);

    sdp_timer_start(plan->timer_overall);
    sdp_timer_reset(plan->timer_fft);
    sdp_timer_reset(plan->timer_grid);
    sdp_gridder_check_plan(plan, status);
    if (*status) return;

    sdp_gridder_check_buffers(
            uvw, freq_hz, vis, weight, dirty_image, true, status
    );
    if (*status) return;

    const int npix_x = (int)sdp_mem_shape_dim(dirty_image, 0);
    const int npix_y = (int)sdp_mem_shape_dim(dirty_image, 1);  // this should be the same as checked by check_params()

    uint64_t num_threads[] = {1, 1, 1}, num_blocks[] = {1, 1, 1};

    const sdp_MemType vis_type = sdp_mem_type(vis);

    const int chunk_size = plan->num_rows;
    const int coord_type = sdp_mem_type(uvw);
    const int dbl_vis = (vis_type & SDP_MEM_DOUBLE);
    const int dbl_coord = (coord_type & SDP_MEM_DOUBLE);

    // Create the FFT plan.
    sdp_Fft* fft = sdp_fft_create(
            plan->w_grid_stack, plan->w_grid_stack, 2, true, status
    );
    if (*status) return;

    // Perform convolution correction and final scaling on single real plane
    // note: can recycle same block/thread dims as w correction kernel
    {
        const char* kernel_name = dbl_vis ?
                    "conv_corr_and_scaling<double>" :
                    "conv_corr_and_scaling<float>";
        num_threads[0] = std::min(32, (npix_x + 1) / 2);
        num_threads[1] = std::min(32, (npix_y + 1) / 2);
        // Allow extra in negative x quadrants, for asymmetric image centre
        num_blocks[0] = (npix_x / 2 + 1 + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (npix_y / 2 + 1 + num_threads[1] - 1) / num_threads[1];
        const bool solving = false;
        const void* args[] = {
            sdp_mem_gpu_buffer(dirty_image, status),
            &plan->image_size,
            dbl_vis ?
                (const void*)&plan->pixel_size :
                (const void*)&plan->pixel_size_f,
            &plan->support,
            dbl_vis ?
                (const void*)&plan->conv_corr_norm_factor :
                (const void*)&plan->conv_corr_norm_factor_f,
            sdp_mem_gpu_buffer_const(plan->conv_corr_kernel, status),
            dbl_vis ?
                (const void*)&plan->inv_w_range :
                (const void*)&plan->inv_w_range_f,
            dbl_vis ?
                (const void*)&plan->inv_w_scale :
                (const void*)&plan->inv_w_scale_f,
            sdp_mem_gpu_buffer_const(plan->quadrature_kernel, status),
            sdp_mem_gpu_buffer_const(plan->quadrature_nodes, status),
            sdp_mem_gpu_buffer_const(plan->quadrature_weights, status),
            &solving,
            &plan->do_wstacking
        };
        sdp_launch_cuda_kernel(
                kernel_name, num_blocks, num_threads, 0, 0, args, status
        );
    }

    for (int grid_w = 0; grid_w < plan->num_total_w_grids; grid_w++)
    {
        sdp_mem_clear_contents(plan->w_grid_stack, status);
        if (*status) break;

        // Undo w-stacking and dirty image accumulation.
        {
            const char* kernel_name = dbl_vis ?
                        "reverse_w_screen_to_stack<double, double2>" :
                        "reverse_w_screen_to_stack<float, float2>";
            num_threads[0] = std::min(32, (npix_x + 1) / 2);
            num_threads[1] = std::min(32, (npix_y + 1) / 2);
            // Allow extra in negative x quadrants, for asymmetric image centre
            num_blocks[0] =
                    (npix_x / 2 + 1 + num_threads[0] - 1) / num_threads[0];
            num_blocks[1] =
                    (npix_y / 2 + 1 + num_threads[1] - 1) / num_threads[1];
            const int num_w_grids_subset = 1;
            const bool do_FFT_shift = true;
            const void* args[] = {
                sdp_mem_gpu_buffer_const(dirty_image, status),
                &plan->image_size,
                dbl_vis ?
                    (const void*)&plan->pixel_size :
                    (const void*)&plan->pixel_size_f,
                sdp_mem_gpu_buffer(plan->w_grid_stack, status),
                &plan->grid_size,
                &grid_w,
                &num_w_grids_subset,
                dbl_vis ?
                    (const void*)&plan->inv_w_scale :
                    (const void*)&plan->inv_w_scale_f,
                dbl_vis ?
                    (const void*)&plan->min_plane_w :
                    (const void*)&plan->min_plane_w_f,
                &do_FFT_shift,
                &plan->do_wstacking
            };
            sdp_launch_cuda_kernel(
                    kernel_name, num_blocks, num_threads, 0, 0, args, status
            );
        }

        // Perform 2D FFT
        sdp_timer_resume(plan->timer_fft);
        sdp_fft_exec(fft, plan->w_grid_stack, plan->w_grid_stack, status);
        sdp_timer_pause(plan->timer_fft);

        // Perform degridding
        sdp_timer_resume(plan->timer_grid);
        {
            const char* kernel_name = 0;
            if (plan->do_wstacking)
            {
                if (dbl_vis && dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_degrid_3d"
                            "<double, double2, double, double2, double3>";
                }
                else if (!dbl_vis && !dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_degrid_3d"
                            "<float, float2, float, float2, float3>";
                }
            }
            else
            {
                if (dbl_vis && dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_degrid_2d"
                            "<double, double2, double, double2, double3>";
                }
                else if (!dbl_vis && !dbl_coord)
                {
                    kernel_name = "sdp_cuda_nifty_degrid_2d"
                            "<float, float2, float, float2, float3>";
                }
            }
            if (kernel_name)
            {
                num_threads[0] = 1;
                num_threads[1] = 256;
                num_blocks[0] =
                        (plan->num_chan + num_threads[0] - 1) / num_threads[0];
                num_blocks[1] =
                        (chunk_size + num_threads[1] - 1) / num_threads[1];
                const void* args[] = {
                    &chunk_size,
                    &plan->num_chan,
                    sdp_mem_gpu_buffer(vis, status),
                    sdp_mem_gpu_buffer_const(weight, status),
                    sdp_mem_gpu_buffer_const(uvw, status),
                    sdp_mem_gpu_buffer_const(freq_hz, status),
                    sdp_mem_gpu_buffer_const(plan->w_grid_stack, status),
                    &plan->grid_size,
                    &grid_w,
                    &plan->support,
                    dbl_vis ?
                        (const void*)&plan->beta :
                        (const void*)&plan->beta_f,
                    dbl_coord ?
                        (const void*)&plan->uv_scale :
                        (const void*)&plan->uv_scale_f,
                    dbl_coord ?
                        (const void*)&plan->w_scale :
                        (const void*)&plan->w_scale_f,
                    dbl_coord ?
                        (const void*)&plan->min_plane_w :
                        (const void*)&plan->min_plane_w_f
                };
                sdp_launch_cuda_kernel(kernel_name,
                        num_blocks, num_threads, 0, 0, args, status
                );
            }
        }
        sdp_timer_pause(plan->timer_grid);
    } // loop over w-planes

    // Free FFT plan and data.
    sdp_fft_free(fft);

    // Report timings.
    sdp_timer_pause(plan->timer_overall);
    report_timings(plan, __func__);
}
