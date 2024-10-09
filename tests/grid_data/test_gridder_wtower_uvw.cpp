/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/fourier_transforms/sdp_fft_padded_size.h"
#include "ska-sdp-func/grid_data/sdp_grid_wstack_wtower.h"
#include "ska-sdp-func/grid_data/sdp_gridder_direct.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_height.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"
#include "ska-sdp-func/utility/sdp_timer.h"

#ifdef SDP_HAVE_CFITSIO
#include "fitsio.h"
#endif

using std::complex;
using std::string;


// Convert antenna XYZ coordinates to UVW.
static void xyz_to_uvw(
        const double antenna_xyz[][3],
        const double ha_rad,
        const double dec_rad,
        sdp_Mem* antenna_uvw,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<double, 2> antenna_uvw_;
    sdp_mem_check_and_view(antenna_uvw, &antenna_uvw_, status);
    if (*status) return;
    const int num_antennas = (int) antenna_uvw_.shape[0];
    for (int i = 0; i < num_antennas; i++)
    {
        const double x = antenna_xyz[i][0];
        const double y = antenna_xyz[i][1];
        const double z = antenna_xyz[i][2];
        const double v0 = x * sin(ha_rad) + y * cos(ha_rad);
        antenna_uvw_(i, 0) = x * cos(ha_rad) - y * sin(ha_rad);
        antenna_uvw_(i, 1) = z * cos(dec_rad) + v0 * sin(dec_rad);
        antenna_uvw_(i, 2) = z * sin(dec_rad) - v0 * cos(dec_rad);
    }
}


// Calculate baselines between all antennas.
template<typename T>
static void calculate_baselines(
        const sdp_Mem* antenna_uvw,
        const int row_start,
        sdp_Mem* baselines,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<const double, 2> antenna_uvw_;
    sdp_MemViewCpu<T, 2> baselines_;
    sdp_mem_check_and_view(antenna_uvw, &antenna_uvw_, status);
    sdp_mem_check_and_view(baselines, &baselines_, status);
    if (*status) return;
    const int num_antennas = (int) antenna_uvw_.shape[0];
    for (int i = 0, idx = row_start; i < num_antennas; i++)
    {
        for (int j = i + 1; j < num_antennas; j++, idx++)
        {
            baselines_(idx, 0) = antenna_uvw_(j, 0) - antenna_uvw_(i, 0);
            baselines_(idx, 1) = antenna_uvw_(j, 1) - antenna_uvw_(i, 1);
            baselines_(idx, 2) = antenna_uvw_(j, 2) - antenna_uvw_(i, 2);
        }
    }
}


// Extract real or imaginary parts from data.
static void convert_complex(
        const sdp_Mem* input,
        sdp_Mem* output,
        int offset,
        sdp_Error* status
)
{
    if (*status) return;
    const int64_t num_elements = sdp_mem_num_elements(input);
    if (sdp_mem_type(input) == SDP_MEM_COMPLEX_FLOAT)
    {
        const float* in = (const float*) sdp_mem_data_const(input);
        float* out = (float*) sdp_mem_data(output);
        for (int64_t i = 0; i < num_elements; ++i)
        {
            out[i] = in[2 * i + offset];
        }
    }
    else
    {
        const double* in = (const double*) sdp_mem_data_const(input);
        double* out = (double*) sdp_mem_data(output);
        for (int64_t i = 0; i < num_elements; ++i)
        {
            out[i] = in[2 * i + offset];
        }
    }
}


// Generate a test image with two sources.
static sdp_Mem* generate_model_image(
        sdp_MemType type,
        int image_size,
        sdp_Error* status
)
{
    const int64_t image_shape[] = {image_size, image_size};
    sdp_Mem* image = sdp_mem_create(type, SDP_MEM_CPU, 2, image_shape, status);
    sdp_mem_set_value(image, 0, status);
    const int half = image_size / 2;
    if (type == SDP_MEM_COMPLEX_DOUBLE)
    {
        sdp_MemViewCpu<complex<double>, 2> image_;
        sdp_mem_check_and_view(image, &image_, status);
        image_(half + image_size / 4, half + 2) = 2;
        image_(half - image_size / 4 + 2, half + image_size / 4 - 12) = 1;
    }
    else if (type == SDP_MEM_COMPLEX_FLOAT)
    {
        sdp_MemViewCpu<complex<float>, 2> image_;
        sdp_mem_check_and_view(image, &image_, status);
        image_(half + image_size / 4, half + 2) = 2;
        image_(half - image_size / 4 + 2, half + image_size / 4 - 12) = 1;
    }
    return image;
}


// Generate reference image using a DFT.
static sdp_Mem* generate_reference_image(
        const sdp_Mem* uvws,
        const sdp_Mem* ref_vis,
        double theta,
        double shear_u,
        double shear_v,
        double freq0_hz,
        double dfreq_hz,
        int image_size,
        sdp_MemType type,
        sdp_Error* status
)
{
    const int64_t num_pixels = image_size * image_size;
    const int64_t image_shape[] = {image_size, image_size};
    const int64_t lmn_shape[] = {num_pixels, 3};
    sdp_Mem* lmn = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, lmn_shape, status
    );
#ifdef SDP_HAVE_CUDA
    // Create an image to return.
    sdp_Mem* image = sdp_mem_create(type, SDP_MEM_GPU, 2, image_shape, status);

    // Get lmn coordinates from image.
    sdp_gridder_image_to_flmn(
            image, theta, shear_u, shear_v, NULL, NULL, lmn, status
    );

    // Calculate image using a DFT.
    sdp_Mem* lmn_gpu = sdp_mem_create_copy(lmn, SDP_MEM_GPU, status);
    sdp_Mem* uvws_gpu = sdp_mem_create_copy(uvws, SDP_MEM_GPU, status);
    sdp_Mem* vis_gpu = sdp_mem_create_copy(ref_vis, SDP_MEM_GPU, status);
    sdp_mem_set_value(image, 0, status);
    sdp_gridder_idft(uvws_gpu, vis_gpu, NULL, NULL, lmn_gpu, NULL, 0, 0, 0,
            theta, 0.0, freq0_hz, dfreq_hz, image, status
    );
    sdp_Mem* image_cpu = sdp_mem_create_copy(image, SDP_MEM_CPU, status);
    sdp_mem_free(image);
    sdp_mem_free(lmn);
    sdp_mem_free(lmn_gpu);
    sdp_mem_free(uvws_gpu);
    sdp_mem_free(vis_gpu);
    return image_cpu;
#else
    // Create an image to return.
    sdp_Mem* image = sdp_mem_create(type, SDP_MEM_CPU, 2, image_shape, status);

    // Get lmn coordinates from image.
    sdp_gridder_image_to_flmn(
            image, theta, shear_u, shear_v, NULL, NULL, lmn, status
    );

    // Calculate image using a DFT.
    sdp_mem_set_value(image, 0, status);
    sdp_gridder_idft(uvws, ref_vis, NULL, NULL, lmn, NULL, 0, 0, 0,
            theta, 0.0, freq0_hz, dfreq_hz, image, status
    );

    sdp_mem_free(lmn);
    return image;
#endif
}


// Generate reference visibility data using a DFT.
static sdp_Mem* generate_reference_vis(
        const sdp_Mem* uvws,
        const sdp_Mem* image,
        double theta,
        double shear_u,
        double shear_v,
        double freq0_hz,
        double dfreq_hz,
        int num_chan,
        sdp_MemType vis_type,
        sdp_Error* status
)
{
    // Get flux and lmn coordinates from image.
    const int64_t num_src = sdp_gridder_count_nonzero_pixels(image, status);
    const int64_t lmn_shape[] = {num_src, 3};
    sdp_Mem* flux = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, &num_src, status
    );
    sdp_Mem* lmn = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, lmn_shape, status
    );
    sdp_gridder_image_to_flmn(
            image, theta, shear_u, shear_v, NULL, flux, lmn, status
    );

    // Create a visibility array to return.
    const int64_t num_rows = sdp_mem_shape_dim(uvws, 0);
    const int64_t vis_shape[] = {num_rows, num_chan};
    sdp_Mem* vis = sdp_mem_create(vis_type, SDP_MEM_CPU, 2, vis_shape, status);

    // Calculate visibilities using a DFT.
    sdp_mem_set_value(vis, 0, status);
    sdp_gridder_dft(uvws, NULL, NULL, flux, lmn, 0, 0, 0,
            theta, 0.0, freq0_hz, dfreq_hz, vis, status
    );

    sdp_mem_free(lmn);
    sdp_mem_free(flux);
    return vis;
}


// Generate (u,v,w) baseline coordinates for use in tests.
static sdp_Mem* generate_uvw(sdp_MemType type)
{
    // VLA antenna (x,y,z) coordinates.
    const double antenna_xyz[][3] = {
        {-401.2842, -270.6395, 1.3345},
        {-1317.9926, -889.0279, 2.0336},
        {-2642.9943, -1782.7459, 7.8328},
        {-4329.9414, -2920.6298, 4.217},
        {-6350.012, -4283.1247, -6.0779},
        {-8682.4872, -5856.4585, -7.3861},
        {-11311.4962, -7629.385, -19.3219},
        {-14224.3397, -9594.0268, -32.2199},
        {-17410.1952, -11742.6658, -52.5716},
        {438.6953, -204.4971, -0.1949},
        {1440.9974, -671.8529, 0.6199},
        {2889.4597, -1347.2324, 12.4453},
        {4733.627, -2207.126, 19.9349},
        {6942.0661, -3236.8423, 28.0543},
        {9491.9269, -4425.5098, 19.3104},
        {12366.0731, -5765.3061, 13.8351},
        {15550.4596, -7249.6904, 25.3408},
        {19090.2771, -8748.4418, -53.2768},
        {-38.0377, 434.7135, -0.026},
        {-124.9775, 1428.1567, -1.4012},
        {-259.3684, 2963.3547, -0.0815},
        {-410.6587, 4691.5051, -0.3722},
        {-602.292, 6880.1408, 0.5885},
        {-823.5569, 9407.5172, 0.0647},
        {-1072.9272, 12255.8935, -4.2741},
        {-1349.2489, 15411.7447, -7.7693},
        {-1651.4637, 18863.4683, -9.2248}
    };
    const int num_antennas = 27;
    const int num_times = 32;
    const int num_baselines = num_antennas * (num_antennas - 1) / 2;
    const double ha_inc_rad = (M_PI / 2.0) / num_times;
    const double dec_rad = 40.0 * M_PI / 180.0;

    // Allocate memory for the coordinates.
    sdp_Error status = SDP_SUCCESS;
    const int64_t uvws_shape[] = {num_baselines* num_times, 3};
    const int64_t antenna_uvw_shape[] = {num_antennas, 3};
    sdp_Mem* uvws = sdp_mem_create(type, SDP_MEM_CPU, 2, uvws_shape, &status);
    sdp_Mem* antenna_uvw = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, antenna_uvw_shape, &status
    );

    // Calculate (u,v,w) baseline coordinates.
    for (int t = 0; t < num_times; t++)
    {
        const int start_row = t * num_baselines;
        xyz_to_uvw(antenna_xyz, t * ha_inc_rad, dec_rad, antenna_uvw, &status);
        if (type == SDP_MEM_DOUBLE)
        {
            calculate_baselines<double>(antenna_uvw, start_row, uvws, &status);
        }
        else
        {
            calculate_baselines<float>(antenna_uvw, start_row, uvws, &status);
        }
    }
    sdp_mem_free(antenna_uvw);
    return uvws;
}


// Write a 2D array to a FITS file for inspection.
static void write_data_to_fits(
        const sdp_Mem* image,
        const string& filename
)
{
    const char* fname = filename.c_str();
#ifdef SDP_HAVE_CFITSIO
    long naxes[2] = {0, 0}, firstpix[2] = {1, 1};
    int fits_status = 0;
    fitsfile* f = 0;
    FILE* stream = fopen(fname, "rb");
    if (stream)
    {
        fclose(stream);
        remove(fname);
    }
    fits_create_file(&f, fname, &fits_status);
    naxes[0] = sdp_mem_shape_dim(image, 0);
    naxes[1] = sdp_mem_shape_dim(image, 1);
    long num_pix = naxes[0] * naxes[1];
    fits_create_img(f, sdp_mem_is_double(image) ? DOUBLE_IMG : FLOAT_IMG,
            2, naxes, &fits_status
    );
    fits_movabs_hdu(f, 1, NULL, &fits_status);
    fits_write_pix(f, sdp_mem_is_double(image) ? TDOUBLE : TFLOAT,
            firstpix, num_pix,
            const_cast<void*>(sdp_mem_data_const(image)), &fits_status
    );
    fits_close_file(f, &fits_status);
#else
    (void) image;
    SDP_LOG_ERROR("Cannot save FITS file %s: CFITSIO not available.", fname);
#endif
}


// Write a 2D array to a FITS file for inspection.
static void write_fits_file(
        const sdp_Mem* image,
        const string& filename
)
{
    sdp_Error status = SDP_SUCCESS;
    if (sdp_mem_is_complex(image))
    {
        sdp_MemType type = (
            sdp_mem_is_double(image) ? SDP_MEM_DOUBLE : SDP_MEM_FLOAT
        );
        const int64_t shape[] = {
            sdp_mem_shape_dim(image, 0), sdp_mem_shape_dim(image, 1)
        };
        sdp_Mem* temp = sdp_mem_create(type, SDP_MEM_CPU, 2, shape, &status);
        convert_complex(image, temp, 0, &status);
        write_data_to_fits(temp, filename + "_REAL.fits");
        convert_complex(image, temp, 1, &status);
        write_data_to_fits(temp, filename + "_IMAG.fits");
        sdp_mem_free(temp);
    }
    else
    {
        write_data_to_fits(image, filename + ".fits");
    }
}


static const double cell_size_arcsec = 0.8;
static const double freq0_hz = C_0;
static const double dfreq_hz = C_0 / 100;


static void run_and_check(
        const char* test_name,
        const sdp_Mem* model_img,
        const sdp_Mem* ref_img,
        const sdp_Mem* ref_vis,
        sdp_MemLocation loc,
        sdp_MemType uvw_type,
        sdp_MemType vis_type,
        sdp_Error* status
)
{
    SDP_LOG_INFO("Running test %s", test_name);

    // Imaging parameters.
    int image_size = sdp_mem_shape_dim(ref_img, 0);
    int subgrid_size = 256;
    double shear_u = 0.0;
    double shear_v = 0.0;
    int support = 8;
    int oversampling = 16 * 1024;
    int w_support = 8;
    int w_oversampling = 16 * 1024;
    double padding_factor = 1.2;
    double subgrid_frac = 2.0 / 3.0;
    int verbosity = 0;

    // Get padded grid size.
    const int grid_size = 2 * sdp_fft_padded_size(
            int(image_size * 0.5), padding_factor
    );

    // Get image cell size, imaged FoV and padded FoV.
    double cell_size_rad = (cell_size_arcsec / 3600.0) * (M_PI / 180.0);
    double fov = sin(cell_size_rad) * image_size;
    double theta = sin(cell_size_rad) * grid_size;

    // Determine w-step and maximum w-tower height.
    double w_step = sdp_gridder_determine_w_step(
            theta, fov, shear_u, shear_v, 0
    );
    double w_tower_height = sdp_gridder_find_max_w_tower_height(
            2 * subgrid_size, subgrid_size, theta, w_step, shear_u, shear_v,
            support, oversampling, w_support, w_oversampling,
            fov, subgrid_frac, 0, 0, status
    );

    // Generate baseline coordinates.
    sdp_Mem* uvws = generate_uvw(uvw_type);
    int64_t num_rows = sdp_mem_shape_dim(ref_vis, 0);
    int64_t num_chan = sdp_mem_shape_dim(ref_vis, 1);
    sdp_Mem* start_chs = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    sdp_Mem* end_chs = sdp_mem_create(SDP_MEM_INT, loc, 1, &num_rows, status);
    sdp_mem_set_value(start_chs, 0, status);
    sdp_mem_set_value(end_chs, (int) num_chan, status);

    // Pad the input model image.
    const int64_t grid_shape[] = {grid_size, grid_size};
    sdp_Mem* img_padded = sdp_mem_create(
            sdp_mem_type(model_img), SDP_MEM_CPU, 2, grid_shape, status
    );
    sdp_mem_set_value(img_padded, 0, status);
    sdp_gridder_subgrid_add(img_padded, 0, 0, model_img, 1.0, status);

    // Make sure the reference visibilities have the correct precision.
    sdp_Mem* ref_vis_converted = sdp_mem_convert_precision(
            ref_vis, vis_type, status
    );

    // Copy inputs to GPU as required.
    sdp_Mem* temp_uvws = 0;
    sdp_Mem* temp_image = 0;
    sdp_Mem* temp_vis = 0;
    const sdp_Mem* ptr_img = img_padded;
    const sdp_Mem* ptr_uvws = uvws;
    const sdp_Mem* ptr_vis = ref_vis_converted;
    if (loc == SDP_MEM_GPU)
    {
        ptr_img = temp_image = sdp_mem_create_copy(img_padded, loc, status);
        ptr_uvws = temp_uvws = sdp_mem_create_copy(uvws, loc, status);
        ptr_vis = temp_vis = sdp_mem_create_copy(
                ref_vis_converted, loc, status
        );
    }

    // Call the degridder.
    const int64_t vis_shape[] = {num_rows, num_chan};
    sdp_Mem* vis_wtower = sdp_mem_create(vis_type, loc, 2, vis_shape, status);
    sdp_mem_set_value(vis_wtower, 0, status);
    sdp_grid_wstack_wtower_degrid_all(ptr_img, freq0_hz, dfreq_hz, ptr_uvws,
            subgrid_size, theta, w_step, shear_u, shear_v, support,
            oversampling, w_support, w_oversampling, subgrid_frac,
            w_tower_height, verbosity, vis_wtower, status
    );

    // Check against reference data.
    double rms_vis_wtower = sdp_gridder_rms_diff(ref_vis, vis_wtower, status);
    SDP_LOG_INFO("RMS difference (visibilities): %.3e", rms_vis_wtower);
    assert(rms_vis_wtower < 0.05);

    // Call the gridder.
    sdp_Mem* img_wtower = sdp_mem_create(
            sdp_mem_type(img_padded), loc, 2, grid_shape, status
    );
    sdp_mem_set_value(img_wtower, 0, status);
    sdp_grid_wstack_wtower_grid_all(ptr_vis, freq0_hz, dfreq_hz, ptr_uvws,
            subgrid_size, theta, w_step, shear_u, shear_v, support,
            oversampling, w_support, w_oversampling, subgrid_frac,
            w_tower_height, verbosity, img_wtower, status
    );
    sdp_mem_scale_real(img_wtower, 1.0 / (num_rows * num_chan), status);

    // Trim the image.
    const int64_t image_shape[] = {image_size, image_size};
    sdp_Mem* img_wtower_trimmed = sdp_mem_create(
            sdp_mem_type(img_padded), loc, 2, image_shape, status
    );
    sdp_gridder_subgrid_cut_out(img_wtower, 0, 0, img_wtower_trimmed, status);

    // Check against reference data.
    double rms_img_wtower = sdp_gridder_rms_diff(
            ref_img, img_wtower_trimmed, status
    );
    SDP_LOG_INFO("RMS difference (image): %.3e", rms_img_wtower);
    assert(rms_img_wtower < 1e-3);
#ifdef SDP_HAVE_CFITSIO
    write_fits_file(ref_img, string("test_wtowers_") + test_name);
#endif

    // Free scratch arrays.
    sdp_mem_free(ref_vis_converted);
    sdp_mem_free(temp_image);
    sdp_mem_free(temp_uvws);
    sdp_mem_free(temp_vis);
    sdp_mem_free(img_wtower);
    sdp_mem_free(img_wtower_trimmed);
    sdp_mem_free(vis_wtower);
    sdp_mem_free(img_padded);
    sdp_mem_free(end_chs);
    sdp_mem_free(start_chs);
    sdp_mem_free(uvws);
}


int main()
{
    // Create reference data (only once, as this is relatively expensive).
    sdp_Error status = SDP_SUCCESS;
    sdp_Timer* tmr = sdp_timer_create(SDP_TIMER_NATIVE);
    const int image_size = 512;
    double fov = sin((cell_size_arcsec / 3600.0) * M_PI / 180.0) * image_size;
    sdp_Mem* model_img = generate_model_image(
            SDP_MEM_COMPLEX_DOUBLE, image_size, &status
    );
    sdp_Mem* uvws = generate_uvw(SDP_MEM_DOUBLE);
    int64_t num_rows = sdp_mem_shape_dim(uvws, 0);
    int64_t num_chan = 2;

    SDP_LOG_INFO("Generating reference visibilities using DFT...");
    sdp_timer_start(tmr);
    sdp_Mem* ref_vis = generate_reference_vis(uvws, model_img, fov, 0.0, 0.0,
            freq0_hz, dfreq_hz, num_chan, SDP_MEM_COMPLEX_DOUBLE, &status
    );
    SDP_LOG_INFO("  took %.3f s.", sdp_timer_elapsed(tmr));

    SDP_LOG_INFO("Generating reference image using DFT "
            "(takes around 10 seconds)..."
    );
    sdp_timer_start(tmr);
    sdp_Mem* ref_img = generate_reference_image(uvws, ref_vis, fov, 0.0, 0.0,
            freq0_hz, dfreq_hz, image_size, SDP_MEM_COMPLEX_DOUBLE, &status
    );
    SDP_LOG_INFO("  took %.3f s.", sdp_timer_elapsed(tmr));
    sdp_mem_scale_real(ref_img, 1.0 / (num_rows * num_chan), &status);
#ifdef SDP_HAVE_CFITSIO
    write_fits_file(ref_img, "test_wtowers_reference_DFT");
#endif

    // Test cases.
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU_double_uvw_double_vis",
                model_img, ref_img, ref_vis,
                SDP_MEM_CPU, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU_double_uvw_float_vis",
                model_img, ref_img, ref_vis,
                SDP_MEM_CPU, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_FLOAT, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("CPU_float_uvw_float_vis",
                model_img, ref_img, ref_vis,
                SDP_MEM_CPU, SDP_MEM_FLOAT, SDP_MEM_COMPLEX_FLOAT, &status
        );
        assert(status == SDP_SUCCESS);
    }
#ifdef SDP_HAVE_CUDA
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU_double_uvw_double_vis",
                model_img, ref_img, ref_vis,
                SDP_MEM_GPU, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_DOUBLE, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU_double_uvw_float_vis",
                model_img, ref_img, ref_vis,
                SDP_MEM_GPU, SDP_MEM_DOUBLE, SDP_MEM_COMPLEX_FLOAT, &status
        );
        assert(status == SDP_SUCCESS);
    }
    {
        sdp_Error status = SDP_SUCCESS;
        run_and_check("GPU_float_uvw_float_vis",
                model_img, ref_img, ref_vis,
                SDP_MEM_GPU, SDP_MEM_FLOAT, SDP_MEM_COMPLEX_FLOAT, &status
        );
        assert(status == SDP_SUCCESS);
    }
#endif

    sdp_mem_free(uvws);
    sdp_mem_free(model_img);
    sdp_mem_free(ref_img);
    sdp_mem_free(ref_vis);
    sdp_timer_free(tmr);
    return 0;
}
