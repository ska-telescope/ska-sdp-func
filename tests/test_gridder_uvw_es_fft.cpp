/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ska-sdp-func/fft/sdp_fft.h"

#include "ska-sdp-func/gridder_uvw_es_fft/sdp_gridder_uvw_es_fft.h"
#include "ska-sdp-func/gridder_uvw_es_fft/sdp_gridder_uvw_es_fft_utils.h"

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"


#ifndef PI
#define PI 3.1415926535897931
#endif


static void run_and_check(
        const char* test_name,
        const bool do_wstacking,
        const double epsilon,
        sdp_MemType uvw_type,
        sdp_MemType freq_hz_type,
        sdp_MemType vis_type,
        sdp_MemType weight_type,
        sdp_MemType image_type,
        sdp_Error* status)
{
    // Generate some test data.
    const int num_vis = 1024;
    const int num_channels = 10;
    const int im_size = 1024;

    const double fov = 2;
    const double speed_of_light = 299792458.0;

    const double pixel_size_rad = fov * PI / 180.0 / im_size;
    const double f_0 = 1e9;

    SDP_LOG_INFO("Running test: %s", test_name);

    int64_t uvw_shape[] = {num_vis, 3};
    int64_t vis_shape[] = {num_vis, num_channels};
    int64_t image_shape[] = {im_size, im_size};
    int64_t freq_hz_shape[] = {num_channels};

    sdp_Mem* uvw =
            sdp_mem_create(uvw_type, SDP_MEM_CPU, 2, uvw_shape, status);
    sdp_Mem* freq_hz =
            sdp_mem_create(freq_hz_type, SDP_MEM_CPU, 1, freq_hz_shape, status);
    sdp_Mem* weight =
            sdp_mem_create(weight_type, SDP_MEM_CPU, 2, vis_shape, status);
    sdp_Mem* vis =
            sdp_mem_create(vis_type, SDP_MEM_CPU, 2, vis_shape, status);
    sdp_Mem* dirty_image =
            sdp_mem_create(image_type, SDP_MEM_CPU, 2, image_shape, status);

    sdp_Mem* est_vis_gpu =
            sdp_mem_create(vis_type, SDP_MEM_GPU, 2, vis_shape, status);
    sdp_Mem* est_dirty_image_gpu =
            sdp_mem_create(image_type, SDP_MEM_GPU, 2, image_shape, status);

    sdp_mem_clear_contents(uvw, status);
    sdp_mem_clear_contents(freq_hz, status);
    sdp_mem_clear_contents(weight, status);
    sdp_mem_clear_contents(vis, status);
    sdp_mem_clear_contents(dirty_image, status);

    sdp_mem_clear_contents(est_vis_gpu, status);
    sdp_mem_clear_contents(est_dirty_image_gpu, status);

    sdp_mem_random_fill(uvw, status);
    sdp_mem_random_fill(dirty_image, status);
    sdp_mem_random_fill(vis, status);

    double min_freq = 0;
    double max_freq = 0;
    double min_abs_w = 1e19;
    double max_abs_w = 1e-19;

    if (test_name[0] != 'f')  // only prepare data if not a fail test
    {
        SDP_LOG_INFO("Preparing test data");

        // fill weight with ones
        void* weights = (void*)sdp_mem_data(weight);
        for (size_t i = 0; i < num_vis * num_channels; i++)
        {
            if (sdp_mem_type(weight) == SDP_MEM_DOUBLE)
            {
                double* temp = (double*)weights;
                temp[i] = 1.0;
            }
            else
            {
                float* temp = (float*)weights;
                temp[i] = 1.0f;
            }
        }

        // fill freq_hz
        void* freqs = (void*)sdp_mem_data(freq_hz);
        for (size_t i = 0; i < num_channels; i++)
        {
            if (sdp_mem_type(freq_hz) == SDP_MEM_DOUBLE)
            {
                double* temp = (double*)freqs;
                temp[i] = f_0 + double(i) * (f_0 / double(num_channels));
                if (i == 0) min_freq = temp[i];
                if (i == num_channels - 1) max_freq = temp[i];
            }
            else
            {
                float* temp = (float*)freqs;
                temp[i] = (float)f_0 + float(i) *
                        (((float)f_0) / float(num_channels));
                if (i == 0) min_freq = temp[i];
                if (i == num_channels - 1) max_freq = temp[i];
            }
        }

        // modify uvw, vis, and dirty_image from raw random numbers
        void* uvws = (void*)sdp_mem_data(uvw);
        for (size_t i = 0; i < num_vis * 3; i++)
        {
            if (sdp_mem_type(uvw) == SDP_MEM_DOUBLE)
            {
                double* temp = (double*)uvws;
                temp[i] -= 0.5;
                temp[i] /= pixel_size_rad * f_0 / speed_of_light;
            }
            else
            {
                float* temp = (float*)uvws;
                temp[i] -= 0.5;
                temp[i] /= pixel_size_rad * f_0 / speed_of_light;
            }
        }
        void* vis_1 = (void*)sdp_mem_data(vis);
        for (size_t i = 0; i < num_vis * num_channels * 2; i++)
        {
            if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
            {
                double* temp = (double*)vis_1;
                temp[i] -= 0.5;
            }
            else
            {
                float* temp = (float*)vis_1;
                temp[i] -= 0.5;
            }
        }
        void* image = (void*)sdp_mem_data(dirty_image);
        for (size_t i = 0; i < im_size * im_size; i++)
        {
            if (sdp_mem_type(dirty_image) == SDP_MEM_DOUBLE)
            {
                double* temp = (double*)image;
                temp[i] -= 0.5;
            }
            else
            {
                float* temp = (float*)image;
                temp[i] -= 0.5f;
            }
        }

        // find min_abs_w and max_abs_w
        for (size_t i = 0; i < num_vis; i++)
        {
            size_t ind = 3 * i + 2;

            if (sdp_mem_type(uvw) == SDP_MEM_DOUBLE)
            {
                const double* temp = (const double*)uvws;
                if (min_abs_w > fabs(temp[ind])) min_abs_w = fabs(temp[ind]);
                if (max_abs_w < fabs(temp[ind])) max_abs_w = fabs(temp[ind]);
            }
            else
            {
                const float* temp = (const float*)uvws;
                if (min_abs_w > fabs(temp[ind])) min_abs_w = fabs(temp[ind]);
                if (max_abs_w < fabs(temp[ind])) max_abs_w = fabs(temp[ind]);
            }
        }

        min_abs_w *= min_freq / speed_of_light;
        max_abs_w *= max_freq / speed_of_light;
    }

    // create GPU copies
    sdp_Mem* freq_hz_gpu =
            sdp_mem_create_copy(freq_hz, SDP_MEM_GPU, status);
    sdp_Mem* uvw_gpu = sdp_mem_create_copy(uvw, SDP_MEM_GPU, status);
    sdp_Mem* vis_gpu = sdp_mem_create_copy(vis, SDP_MEM_GPU, status);
    sdp_Mem* weight_gpu = sdp_mem_create_copy(weight, SDP_MEM_GPU, status);
    sdp_Mem* dirty_image_gpu =
            sdp_mem_create_copy(dirty_image, SDP_MEM_GPU, status);

    if (test_name[0] == 'f') // is this a fail test?
    {
        // special fail tests
        if (test_name[1] == '0')
        {
            sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    dirty_image, // ON CPU!!
                    pixel_size_rad,
                    pixel_size_rad,
                    epsilon,
                    min_abs_w,
                    max_abs_w,
                    do_wstacking,
                    status);

            if (*status) return;

            sdp_grid_uvw_es_fft(
                    gridder,
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    est_dirty_image_gpu,
                    status);
        }
        else if (test_name[1] == '1')
        {
            sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
                    freq_hz_gpu, // bad uvw for wrong num of rows
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    dirty_image_gpu,
                    pixel_size_rad,
                    pixel_size_rad,
                    epsilon,
                    min_abs_w,
                    max_abs_w,
                    do_wstacking,
                    status);

            if (*status) return;

            sdp_grid_uvw_es_fft(
                    gridder,
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    est_dirty_image_gpu,
                    status);
        }
        else if (test_name[1] == '2')
        {
            sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
                    dirty_image_gpu, // bad uvw for wrong num of cols
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    dirty_image_gpu,
                    pixel_size_rad,
                    pixel_size_rad,
                    epsilon,
                    min_abs_w,
                    max_abs_w,
                    do_wstacking,
                    status);

            if (*status) return;

            sdp_grid_uvw_es_fft(
                    gridder,
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    est_dirty_image_gpu,
                    status);
        }
        else if (test_name[1] == '3')
        {
            sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
                    uvw_gpu,
                    uvw_gpu, // bad freq_hz_gpu for wrong num of chans
                    vis_gpu,
                    weight_gpu,
                    dirty_image_gpu,
                    pixel_size_rad,
                    pixel_size_rad,
                    epsilon,
                    min_abs_w,
                    max_abs_w,
                    do_wstacking,
                    status);

            if (*status) return;

            sdp_grid_uvw_es_fft(
                    gridder,
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    est_dirty_image_gpu,
                    status);
        }
        else if (test_name[1] == '4')
        {
            sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    dirty_image_gpu, // bad weight size
                    dirty_image_gpu,
                    pixel_size_rad,
                    pixel_size_rad,
                    epsilon,
                    min_abs_w,
                    max_abs_w,
                    do_wstacking,
                    status);

            if (*status) return;

            sdp_grid_uvw_es_fft(
                    gridder,
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    est_dirty_image_gpu,
                    status);
        }
        else if (test_name[1] == '5')
        {
            sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    weight_gpu, // bad dirty_image size
                    pixel_size_rad,
                    pixel_size_rad,
                    epsilon,
                    min_abs_w,
                    max_abs_w,
                    do_wstacking,
                    status);

            if (*status) return;

            sdp_grid_uvw_es_fft(
                    gridder,
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    est_dirty_image_gpu,
                    status);
        }
        else if (test_name[1] == '6')
        {
            sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    dirty_image_gpu,
                    pixel_size_rad,
                    pixel_size_rad * 2, // pixels not square!!
                    epsilon,
                    min_abs_w,
                    max_abs_w,
                    do_wstacking,
                    status);

            if (*status) return;

            sdp_grid_uvw_es_fft(
                    gridder,
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    est_dirty_image_gpu,
                    status);
        }
        else  // just a "normal" fail test
        {
            sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    dirty_image_gpu,
                    pixel_size_rad,
                    pixel_size_rad,
                    epsilon,
                    min_abs_w,
                    max_abs_w,
                    do_wstacking,
                    status);

            if (*status) return;

            sdp_grid_uvw_es_fft(
                    gridder,
                    uvw_gpu,
                    freq_hz_gpu,
                    vis_gpu,
                    weight_gpu,
                    est_dirty_image_gpu,
                    status);
        }
    }

    sdp_GridderUvwEsFft* gridder = sdp_gridder_uvw_es_fft_create_plan(
            uvw_gpu,
            freq_hz_gpu, // in Hz
            vis_gpu,
            weight_gpu,
            dirty_image_gpu,
            pixel_size_rad,
            pixel_size_rad,
            epsilon,
            min_abs_w,
            max_abs_w,
            do_wstacking,
            status);

    SDP_LOG_INFO("Running test: %s", test_name);

    sdp_grid_uvw_es_fft(
            gridder,
            uvw_gpu,
            freq_hz_gpu,
            vis_gpu,
            weight_gpu,
            est_dirty_image_gpu,
            status);

    // copy output to CPU
    sdp_Mem* est_dirty_image = sdp_mem_create_copy(est_dirty_image_gpu,
            SDP_MEM_CPU,
            status);

    // calc dot product of dirty_image and est_dirty_image
    double adj1 = 0;
    {
        void* di = (void*)sdp_mem_data(    dirty_image);
        void* est_di = (void*)sdp_mem_data(est_dirty_image);
        if (sdp_mem_type(dirty_image) == SDP_MEM_DOUBLE)
        {
            double* x = (double*)di;
            double* y = (double*)est_di;

            for (size_t i = 0; i < im_size * im_size; i++)
            {
                adj1 += x[i] * y[i];
            }
        }
        else
        {
            float* x = (float*)di;
            float* y = (float*)est_di;

            for (size_t i = 0; i < im_size * im_size; i++)
            {
                adj1 += x[i] * y[i];
            }
        }
    }

    sdp_ifft_degrid_uvw_es(
            gridder,
            uvw_gpu,
            freq_hz_gpu,
            est_vis_gpu,
            weight_gpu,
            dirty_image_gpu,
            status);

    // copy output to CPU
    sdp_Mem* est_vis = sdp_mem_create_copy(est_vis_gpu, SDP_MEM_CPU, status);

    // calc dot product of vis and est_vis
    double adj2 = 0;
    {
        const void* v = (const void*)sdp_mem_data(    vis);
        const void* est_v = (const void*)sdp_mem_data(est_vis);
        if (sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
        {
            const std::complex<double>* x = (const std::complex<double>*)v;
            const std::complex<double>* y = (const std::complex<double>*)est_v;

            for (size_t i = 0; i < num_vis * num_channels; i++)
            {
                adj2 += real(x[i]) * real(y[i]) + imag(x[i]) * imag(y[i]);
            }
        }
        else
        {
            const std::complex<float>* x = (const std::complex<float>*)v;
            const std::complex<float>* y = (const std::complex<float>*)est_v;

            for (size_t i = 0; i < num_vis * num_channels; i++)
            {
                adj2 += real(x[i]) * real(y[i]) + imag(x[i]) * imag(y[i]);
            }
        }
    }

    double max_adj = ((fabs(adj1) > fabs(adj2)) ? fabs(adj1) : fabs(adj2));

    double adj_error = fabs(adj1 - adj2) / max_adj;

    printf("*****************************************\n");
    printf(       "adj1 = %.15e\n", adj1);
    printf(       "adj2 = %.15e\n", adj2);
    printf("adjointness test = %.6e\n", adj_error);
    printf("*****************************************\n");

    // check output
    double threshold = 0;
    if (sdp_mem_type(est_dirty_image) == SDP_MEM_DOUBLE)
    {
        threshold = 1e-12;
    }
    else
    {
        threshold = 1e-5;
    }
    assert(adj_error <= threshold);

    // free memory
    sdp_gridder_uvw_es_fft_free_plan(gridder);

    sdp_mem_free(uvw);
    sdp_mem_free(freq_hz);
    sdp_mem_free(vis);
    sdp_mem_free(weight);
    sdp_mem_free(dirty_image);

    sdp_mem_free(est_vis);
    sdp_mem_free(est_dirty_image);

    sdp_mem_free(uvw_gpu);
    sdp_mem_free(freq_hz_gpu);
    sdp_mem_free(vis_gpu);
    sdp_mem_free(weight_gpu);
    sdp_mem_free(dirty_image_gpu);

    sdp_mem_free(est_vis_gpu);
    sdp_mem_free(est_dirty_image_gpu);
}


int main()
{
#ifdef SDP_HAVE_CUDA
    // Happy paths.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("3D double", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("3D single", true, 1e-5,
                SDP_MEM_FLOAT,
                SDP_MEM_FLOAT,
                SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_FLOAT,
                SDP_MEM_FLOAT,
                &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("2D single", false, 1e-5,
                SDP_MEM_FLOAT,
                SDP_MEM_FLOAT,
                SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_FLOAT,
                SDP_MEM_FLOAT,
                &status);
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("2D double", false, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);

        assert(status == SDP_SUCCESS);
    }
#endif

    // Sad paths

    // These test bad parameters and buffers.
    // Even more exhaustive testing is done in the Python tests.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("f0: dirty_image in CPU", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("f1: uvw bad rows", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("f2: uvw bad cols", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("f3: freq_hz bad chans", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("f4: bad weight size", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("f5: bad dirty_image_gpu size", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("f6: pixels not square", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("fail: uvw complex", true, 1e-12,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("fail: freq_hz complex", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("fail: vis not complex", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("fail: weight complex", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("fail: dirty_image complex", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("fail: inconsistent double precision", true, 1e-12,
                SDP_MEM_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_COMPLEX_DOUBLE,
                SDP_MEM_DOUBLE,
                SDP_MEM_FLOAT,
                &status);
        assert(status != SDP_SUCCESS);
    }

    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("fail: inconsistent single precision", true, 1e-12,
                SDP_MEM_FLOAT,
                SDP_MEM_FLOAT,
                SDP_MEM_COMPLEX_FLOAT,
                SDP_MEM_DOUBLE,
                SDP_MEM_FLOAT,
                &status);
        assert(status != SDP_SUCCESS);
    }

    return 0;
}
