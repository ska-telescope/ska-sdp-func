/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cmath>
#include <complex>
#include <cassert>

#include "ska-sdp-func/clean/sdp_hogbom_clean.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/grid_data/sdp_gridder_uvw_es_fft.h"
#include "ska-sdp-func/visibility/sdp_dft.h"

using std::complex;

#define INDEX_2D(N2, N1, I2, I1) (N1 * I2 + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1) (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

static void create_test_data(
    sdp_Mem *dirty_img,
    sdp_Mem *psf,
    int nxydirty,
    int nxypsf,
    sdp_Error *status)
{

    const int num_components = 10;
    const int num_pols = 1;
    const int num_channels = 1;
    const int num_baselines = 1000;
    const int num_times = 1;
    const double channel_start_hz = 100e6;
    const double channel_step_hz = 100e3;

    // // calculate useful shapes and sizes
    // const int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
    // const int64_t psf_size = psf_dim * psf_dim;
    // const int64_t dirty_img_shape[] = {dirty_img_dim, dirty_img_dim};
    // const int64_t psf_shape[] = {psf_dim, psf_dim};
    const int64_t vis_shape[] = {num_times, num_baselines, num_channels, num_pols};
    const int64_t fluxes_shape[] = {num_components, num_channels, num_pols};
    const int64_t fluxes_psf_shape[] = {1, num_channels, num_pols};
    const int64_t directions_shape[] = {num_components, 3};
    const int64_t directions_psf_shape[] = {1, 3};
    const int64_t uvw_shape[] = {num_times, num_baselines, 3};

    // initialise empty arrays for dirty image generation
    sdp_Mem *vis = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 4, vis_shape, status);
    complex<double> *vis_ptr = (complex<double> *)sdp_mem_data(vis);
    sdp_mem_clear_contents(vis, status);

    sdp_Mem *vis_psf = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 4, vis_shape, status);
    complex<double> *vis_psf_ptr = (complex<double> *)sdp_mem_data(vis_psf);
    sdp_mem_clear_contents(vis_psf, status);

    sdp_Mem *fluxes = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 3, fluxes_shape, status);
    complex<double> *fluxes_ptr = (complex<double> *)sdp_mem_data(fluxes);
    sdp_mem_clear_contents(fluxes, status);

    sdp_Mem *fluxes_psf = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 3, fluxes_psf_shape, status);
    complex<double> *fluxes_psf_ptr = (complex<double> *)sdp_mem_data(fluxes_psf);
    sdp_mem_clear_contents(fluxes_psf, status);
    // Create flux of 1 + 0j's for psf
    fluxes_psf_ptr[0] = std::complex<double>(1,0);

    sdp_Mem *directions = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, directions_shape, status);
    double *directions_ptr = (double *)sdp_mem_data(directions);
    sdp_mem_clear_contents(directions, status);

    sdp_Mem *directions_psf = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, directions_psf_shape, status);
    // double *directions_psf_ptr = (double *)sdp_mem_data(directions_psf);
    sdp_mem_clear_contents(directions_psf, status);

    sdp_Mem *uvw = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 3, uvw_shape, status);
    double *uvw_ptr = (double *)sdp_mem_data(uvw);
    sdp_mem_clear_contents(uvw, status);

    // UVW coverage as a random filled circle of given max radius, centered on (0,0)
    const int y0 = 0, x0 = 0, r_max = 3000;
    double theta = 0, radius = 0, x = 0, y = 0;

    for (int i = 0; i < num_baselines; i++)
    {

        // random angle
        theta = 2 * M_PI * (rand() / ((double)RAND_MAX));

        // random radius
        radius = r_max * (rand() / ((double)RAND_MAX));

        x = x0 + radius * cos(theta);
        y = y0 + radius * sin(theta);

        const unsigned int i_uvw = INDEX_3D(num_times, num_baselines, 3, 0, i, 0);
        uvw_ptr[i_uvw] = x;
        uvw_ptr[i_uvw + 1] = y;
        uvw_ptr[i_uvw + 2] = 0;
    }

    // Create random fluxes between 1 + 0j and 10 + 0j
    int upper = 10, lower = 1;
    double ran_flux = 0;

    for (int i = 0; i < num_components; i++)
    {

        ran_flux = (rand() % (upper - lower + 1)) + lower;

        const unsigned int i_fluxes = INDEX_3D(num_components, num_channels, num_pols, i, 0, 0);
        fluxes_ptr[i_fluxes] = complex<double>(ran_flux, 0);
    }

    // create random lmn co-ordinates between (-0.015,-0.015) and (0.015,0.015)
    upper = 0, upper = 30;
    double ran_l = 0, ran_m = 0, ran_n = 0;
    for (int i = 0; i < num_components; i++)
    {

        ran_l = (double)((rand() % (upper - lower + 1)) + lower - 15) / 100;
        ran_m = (double)((rand() % (upper - lower + 1)) + lower - 15) / 100;
        ran_n = 1 - pow(ran_l, 2) - pow(ran_m, 2);

        const unsigned int i_directions = INDEX_2D(num_components, 3, i, 0);
        directions_ptr[i_directions] = ran_l;
        directions_ptr[i_directions + 1] = ran_m;
        directions_ptr[i_directions + 2] = ran_n;
    }

    // create visabilities
    sdp_dft_point_v01(directions, fluxes, uvw, channel_start_hz, channel_step_hz, vis, status);

    // create PSF visabilities
    sdp_dft_point_v01(directions_psf, fluxes_psf, uvw, channel_start_hz, channel_step_hz, vis_psf, status);

    // initialise settings for gridder
    int fov = 2; // degrees
    double pixel_size_rad = fov * M_PI / 180 / nxydirty;
    double pixel_size_rad_psf = fov * M_PI / 180 / nxypsf;
    bool do_w_stacking = false;
    double epsilon = 1e-5;
    int num_vis = num_baselines * num_channels;

    const int64_t freqs_shape[] = {num_channels};
    sdp_Mem *freqs = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, freqs_shape, status);
    double *freqs_ptr = (double *)sdp_mem_data(freqs);

    for (int i = 0; i < num_channels; i++)
    {
        freqs_ptr[i] = channel_start_hz + (i * channel_step_hz);
    }

    // DFT:
    // uvw is 3D and real-valued, with shape: [ num_times, num_baselines, 3 ]
    // vis is 4D and complex-valued, with shape: [ num_times, num_baselines, num_channels, num_pols ]
    // Gridder:
    // uvw [in] The (u,v,w) coordinates.  Must be complex with shape [num_rows, 3]
    // vis [in, out] The visibility data.  Must be complex with shape [num_rows, num_chan]

    // reshape vis output from DFT to fit input of gridder
    const int64_t vis_grid_shape[] = {num_baselines, num_channels}; // num_times = 1 so can be ignored, num_rows = num_baselines
    sdp_Mem *vis_grid = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, vis_grid_shape, status);
    complex<double> *vis_grid_ptr = (complex<double> *)sdp_mem_data(vis_grid);

    for (int i = 0; i < num_baselines; i++)
    {

        const unsigned int i_vis = INDEX_4D(num_times, num_baselines, num_channels, num_pols, 0, i, 0, 0);
        const unsigned int i_vis_grid = INDEX_2D(num_baselines, num_channels, i, 0);
        vis_grid_ptr[i_vis_grid] = vis_ptr[i_vis];
    }

    sdp_Mem *vis_grid_psf = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, vis_grid_shape, status);
    complex<double> *vis_grid_psf_ptr = (complex<double> *)sdp_mem_data(vis_grid_psf);

    for (int i = 0; i < 1; i++)
    {

        const unsigned int i_vis = INDEX_4D(num_times, num_baselines, num_channels, num_pols, 0, i, 0, 0);
        const unsigned int i_vis_grid = INDEX_2D(num_baselines, num_channels, i, 0);
        vis_grid_psf_ptr[i_vis_grid] = vis_psf_ptr[i_vis];
    }

    // reshape UVW to fit input of gridder
    const int64_t uvw_grid_shape[] = {num_baselines, 3}; // num_times = 1 so can be ignored, num_rows = num_baselines
    sdp_Mem *uvw_grid = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, uvw_grid_shape, status);
    double *uvw_grid_ptr = (double *)sdp_mem_data(uvw_grid);

    for (int i = 0; i < num_baselines; i++)
    {

        const unsigned int i_uvw = INDEX_3D(num_times, num_baselines, 3, 0, i, 0);
        const unsigned int i_uvw_grid = INDEX_2D(num_baselines, 3, i, 0);
        uvw_ptr[i_uvw] = uvw_grid_ptr[i_uvw_grid];
        uvw_ptr[i_uvw + 1] = uvw_grid_ptr[i_uvw_grid + 1];
        uvw_ptr[i_uvw + 2] = uvw_grid_ptr[i_uvw_grid + 2];
    }

    // copy to gpu becaause gridder only works on gpu
    sdp_Mem *freqs_gpu = sdp_mem_create_copy(freqs, SDP_MEM_GPU, status);
    sdp_Mem *uvw_gpu = sdp_mem_create_copy(uvw_grid, SDP_MEM_GPU, status);
    sdp_Mem *vis_gpu = sdp_mem_create_copy(vis_grid, SDP_MEM_GPU, status);
    sdp_Mem *vis_psf_gpu = sdp_mem_create_copy(vis_grid_psf, SDP_MEM_GPU, status);
    sdp_Mem *dirty_img_gpu = sdp_mem_create_copy(dirty_img, SDP_MEM_GPU, status);
    sdp_Mem *psf_gpu = sdp_mem_create_copy(psf, SDP_MEM_GPU, status);


    const int64_t weight_shape[] = {num_vis, num_channels};
    sdp_Mem *weight = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, weight_shape, status);
    double *weight_ptr = (double *)sdp_mem_data(weight);

    for (int i = 0; i < (num_vis * num_channels); i++)
    {
        weight_ptr[i] = 1;
    }

    sdp_Mem *weight_gpu = sdp_mem_create_copy(weight, SDP_MEM_GPU, status);

    // create gridder for dirty image
    sdp_GridderUvwEsFft *gridder = sdp_gridder_uvw_es_fft_create_plan(
        uvw_gpu,
        freqs_gpu,
        vis_gpu,
        weight_gpu,
        dirty_img_gpu,
        pixel_size_rad,
        pixel_size_rad,
        epsilon,
        pixel_size_rad,
        pixel_size_rad,
        do_w_stacking,
        status);

    sdp_grid_uvw_es_fft(
        gridder,
        uvw_gpu,
        freqs_gpu,
        vis_gpu,
        weight_gpu,
        dirty_img_gpu,
        status);

    sdp_gridder_uvw_es_fft_free_plan(gridder);

    // create gridder for psf
    sdp_GridderUvwEsFft *gridder2 = sdp_gridder_uvw_es_fft_create_plan(
        uvw_gpu,
        freqs_gpu,
        vis_psf_gpu,
        weight_gpu,
        psf_gpu,
        pixel_size_rad_psf,
        pixel_size_rad_psf,
        epsilon,
        pixel_size_rad_psf,
        pixel_size_rad_psf,
        do_w_stacking,
        status);

    sdp_grid_uvw_es_fft(
        gridder2,
        uvw_gpu,
        freqs_gpu,
        vis_psf_gpu,
        weight_gpu,
        psf_gpu,
        status);

    sdp_gridder_uvw_es_fft_free_plan(gridder2);

    // normalise fluxes by number of baselines
    int64_t num_elements_dirty_img = sdp_mem_num_elements(dirty_img_gpu);

    sdp_mem_copy_contents(dirty_img, dirty_img_gpu, 0, 0, num_elements_dirty_img, status);
    double *dirty_img_ptr = (double *)sdp_mem_data(dirty_img);

    for (int i = 0; i < (nxydirty * nxydirty); i++)
    {
        dirty_img_ptr[i] /= num_baselines;
    }

    int64_t num_elements_psf = sdp_mem_num_elements(psf_gpu);

    sdp_mem_copy_contents(psf, psf_gpu, 0, 0, num_elements_psf, status);
    double *psf_ptr = (double *)sdp_mem_data(psf);

    for (int i = 0; i < (nxypsf * nxypsf); i++)
    {
        psf_ptr[i] /= num_baselines;
    }

    sdp_mem_ref_dec(vis);
    sdp_mem_ref_dec(vis_psf);
    sdp_mem_ref_dec(fluxes);
    sdp_mem_ref_dec(fluxes_psf);
    sdp_mem_ref_dec(directions);
    sdp_mem_ref_dec(directions_psf);
    sdp_mem_ref_dec(uvw);
    sdp_mem_ref_dec(freqs);
    sdp_mem_ref_dec(vis_grid);
    sdp_mem_ref_dec(vis_grid_psf);
    sdp_mem_ref_dec(uvw_grid);
    sdp_mem_ref_dec(freqs_gpu);
    sdp_mem_ref_dec(uvw_gpu);
    sdp_mem_ref_dec(vis_gpu);
    sdp_mem_ref_dec(vis_psf_gpu);
    sdp_mem_ref_dec(dirty_img_gpu);
    sdp_mem_ref_dec(psf_gpu);
    sdp_mem_ref_dec(weight);
    sdp_mem_ref_dec(weight_gpu);
}

static void run_and_check(
    const char *test_name,
    bool read_only_output,
    sdp_MemType input_type,
    sdp_MemType output_type,
    sdp_MemLocation input_location,
    sdp_MemLocation output_location,
    sdp_Error *status)
{

    // settings
    int nxydirty = 1024;
    int nxypsf = 2048;
    double loop_gain = 0.1;
    double threshold = 0.001;
    int cycle_limit = 10000;

    // create test data
    const int64_t diryt_img_shape[] = {nxydirty, nxydirty};
    sdp_Mem *dirty_img = sdp_mem_create(input_type, SDP_MEM_CPU, 2, diryt_img_shape, status);
    sdp_mem_clear_contents(dirty_img, status);

    const int64_t psf_shape[] = {nxypsf, nxypsf};
    sdp_Mem *psf = sdp_mem_create(input_type, SDP_MEM_CPU, 2, psf_shape, status);
    sdp_mem_clear_contents(psf, status);

    const int64_t cbeam_details_shape[] = {3};
    sdp_Mem *cbeam_details = sdp_mem_create(input_type, SDP_MEM_CPU, 1, cbeam_details_shape, status);
    double *cbeam_details_ptr = (double *)sdp_mem_data(cbeam_details);
    // pre-computed variables
    cbeam_details_ptr[0] = 23.9;
    cbeam_details_ptr[1] = 24.6;
    cbeam_details_ptr[2] = 23.6;
    
    create_test_data(
        dirty_img,
        psf,
        nxydirty,
        nxypsf,
        status);

    // create output
    const int64_t skymodel_shape[] = {nxydirty, nxydirty};
    sdp_Mem *skymodel = sdp_mem_create(output_type, SDP_MEM_CPU, 2, skymodel_shape, status);
    sdp_mem_clear_contents(skymodel, status);

    // Copy inputs to specified location.
    sdp_Mem* dirty_img_copy = sdp_mem_create_copy(dirty_img, input_location, status);
    sdp_Mem* psf_copy = sdp_mem_create_copy(psf, input_location, status);
    sdp_Mem* cbeam_details_copy = sdp_mem_create_copy(cbeam_details, input_location, status);
    sdp_Mem* skymodel_copy = sdp_mem_create_copy(skymodel, output_location, status);
    sdp_mem_set_read_only(skymodel_copy, read_only_output);


    // call function to test
    SDP_LOG_INFO("Running test: %s", test_name);
    sdp_hogbom_clean(
        dirty_img_copy,
        psf_copy,
        cbeam_details_copy,
        loop_gain,
        threshold,
        cycle_limit,
        skymodel_copy,
        status);

    sdp_mem_ref_dec(dirty_img);
    sdp_mem_ref_dec(psf);
    sdp_mem_ref_dec(cbeam_details);
    sdp_mem_ref_dec(skymodel);
    sdp_mem_ref_dec(dirty_img_copy);
    sdp_mem_ref_dec(psf_copy);
}

int main()
{
    SDP_LOG_INFO("start of test:");

    // happy paths
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, double precision", false,
                      SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                      SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_SUCCESS);
    }

    // unhappy paths
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, Type Mismatch", false,
                      SDP_MEM_DOUBLE, SDP_MEM_FLOAT,
                      SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_ERR_DATA_TYPE);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, Output set to read-only", true,
                      SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                      SDP_MEM_CPU, SDP_MEM_CPU, &status);
        assert(status == SDP_ERR_RUNTIME);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU, location mis-match", false,
                      SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                      SDP_MEM_CPU, SDP_MEM_GPU, &status);
        assert(status == SDP_ERR_MEM_LOCATION);
    }

        
    #ifdef SDP_HAVE_CUDA
    // Happy paths
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU, double precision", false,
                      SDP_MEM_DOUBLE, SDP_MEM_DOUBLE,
                      SDP_MEM_GPU, SDP_MEM_GPU, &status);
        assert(status == SDP_SUCCESS);
    }

    #endif

    return 0;
}