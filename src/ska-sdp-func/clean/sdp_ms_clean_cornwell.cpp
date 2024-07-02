/* See the LICENSE file at the top-level directory of this distribution.

MSCLEAN cornwell algorithm taken from:
T. J. Cornwell, "Multiscale CLEAN Deconvolution of Radio Synthesis Images,"
in IEEE Journal of Selected Topics in Signal Processing, vol. 2, no. 5,
pp. 793-801, Oct. 2008, doi: 10.1109/JSTSP.2008.2006388.
https://ieeexplore.ieee.org/document/4703304

and

RASCIL implementation of the algorithm.

*/

#include "ska-sdp-func/clean/sdp_ms_clean_cornwell.h"
#include "ska-sdp-func/numeric_functions/sdp_fft_convolution.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#include <complex>
#include <math.h>

using std::complex;

#define INDEX_2D(N2, N1, I2, I1)            (N1 * I2 + I1)
#define INDEX_3D(N3, N2, N1, I3, I2, I1)    (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, \
            I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)


template<typename T>
inline void sdp_create_cbeam(
        const T* cbeam_details,
        int16_t cbeam_dim,
        complex<T>* cbeam
)
{
    // fit a guassian to the main lobe of the psf

    T A = 1;
    T x0 = cbeam_dim / 2;
    T y0 = cbeam_dim / 2;

    // If the dimension is even, adjust the center to be in the middle of the array
    if (cbeam_dim % 2 == 0)
    {
        x0 -= 0.5;
        y0 -= 0.5;
    }

    T sigma_X = cbeam_details[0];
    T sigma_Y = cbeam_details[1];
    T theta = (M_PI / 180) * cbeam_details[2];

    T a =
            pow(cos(theta),
            2
            ) /
            (2 * pow(sigma_X, 2)) + pow(sin(theta), 2) / (2 * pow(sigma_Y, 2));
    T b = sin(2 * theta) /
            (4 * pow(sigma_X, 2)) - sin(2 * theta) / (4 * pow(sigma_Y, 2));
    T c =
            pow(sin(theta),
            2
            ) /
            (2 * pow(sigma_X, 2)) + pow(cos(theta), 2) / (2 * pow(sigma_Y, 2));

    for (int x = 0; x < cbeam_dim; x++)
    {
        for (int y = 0; y < cbeam_dim; y++)
        {
            const unsigned int i_cbeam = INDEX_2D(cbeam_dim, cbeam_dim, x, y);
            T component = A *
                    exp(-(a *
                    pow(x - x0,
                    2
                    ) + 2 * b * (x - x0) * (y - y0) + c * pow(y - y0, 2))
                    );
            cbeam[i_cbeam] = complex<T>(component, 0.0);
        }
    }
}


template<typename T>
inline void sdp_create_copy_complex(
        const T* in,
        int64_t size,
        complex<T>* out
)
{
    // creates a copy of an array to a complex array
    for (int i = 0; i < size; i++)
    {
        out[i] = complex<T>(in[i], 0.0);
    }
}


template<typename T>
inline void sdp_create_copy_real(
        const complex<T>* in,
        int64_t size,
        T* out
)
{
    // creates a copy of a complex double array to a double array
    for (int i = 0; i < size; i++)
    {
        out[i] = std::real(in[i]);
    }
}


template<typename T>
inline void sdp_create_scale_kern(
        complex<T>* scale_kern_list,
        const int* scales,
        const int64_t num_scales,
        int64_t length
)
{
    T sigma = 0;
    int center_x = length / 2;
    int center_y = length / 2;
    T two_sigma_square = 0;

    for (int scale_idx = 0; scale_idx < num_scales; scale_idx++)
    {
        if (scales[scale_idx] == 0)
        {
            const unsigned int i_list = INDEX_3D(num_scales,
                    length,
                    length,
                    scale_idx,
                    length / 2,
                    length / 2
            );
            scale_kern_list[i_list] = complex<T>(1, 0.0);
        }
        else
        {
            sigma = (3.0 / 16.0) * scales[scale_idx];
            two_sigma_square = 2.0 * sigma * sigma;

            for (int x = 0; x < length; x++)
            {
                for (int y = 0; y < length; y++)
                {
                    double distance = (x - center_x) * (x - center_x) +
                            (y - center_y) * (y - center_y);

                    const unsigned int i_list = INDEX_3D(num_scales,
                            length,
                            length,
                            scale_idx,
                            x,
                            y
                    );
                    scale_kern_list[i_list] =
                            complex<T>(exp(-distance / two_sigma_square) /
                            (M_PI * two_sigma_square), 0.0
                            );
                }
            }
        }
    }
}


template<typename T>
static void ms_clean_cornwell(
        const T* psf_ptr,
        const T* cbeam_details_ptr,
        const int* scale_list_ptr,
        const T loop_gain,
        const T threshold,
        const int cycle_limit,
        const int64_t dirty_img_dim,
        const int64_t psf_dim,
        const int64_t scale_dim,
        const sdp_MemType data_type,
        sdp_Mem* clean_comp_mem,
        sdp_Mem* residual_mem,
        T* skymodel_ptr,
        sdp_Error* status
)
{
    // pointers from inputs
    T* clean_comp_ptr = (T*)sdp_mem_data(clean_comp_mem);
    T* residual_ptr = (T*)sdp_mem_data(residual_mem);

    // calculate useful shapes and sizes
    int64_t dirty_img_size = dirty_img_dim * dirty_img_dim;
    int64_t psf_size = psf_dim * psf_dim;

    int64_t dirty_img_shape[] = {dirty_img_dim, dirty_img_dim};
    int64_t psf_shape[] = {psf_dim, psf_dim};
    int64_t scaled_residuals_shape[] =
    {scale_dim, dirty_img_dim, dirty_img_dim};
    int64_t coupling_matrix_shape[] = {scale_dim, scale_dim};
    int64_t scaled_psf_shape[] = {scale_dim, scale_dim, psf_dim, psf_dim};
    int64_t scale_kern_list_shape[] = {scale_dim, psf_dim, psf_dim};
    int64_t scale_list_shape[] = {scale_dim};

    // choose correct complex data type
    sdp_MemType complex_data_type = SDP_MEM_VOID;

    if (data_type == SDP_MEM_DOUBLE)
    {
        complex_data_type = SDP_MEM_COMPLEX_DOUBLE;
    }
    else if (data_type == SDP_MEM_FLOAT)
    {
        complex_data_type = SDP_MEM_COMPLEX_FLOAT;
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data type");
    }

    // Create intermediate data arrays
    sdp_Mem* psf_complex_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            psf_shape,
            status
    );
    complex<T>* psf_complex_ptr = (complex<T>*)sdp_mem_data(psf_complex_mem);
    sdp_mem_clear_contents(psf_complex_mem, status);

    sdp_Mem* residual_complex_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            dirty_img_shape,
            status
    );
    complex<T>* residual_complex_ptr = (complex<T>*)sdp_mem_data(
            residual_complex_mem
    );
    sdp_mem_clear_contents(residual_complex_mem, status);

    sdp_Mem* clean_comp_complex_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            dirty_img_shape,
            status
    );
    complex<T>* clean_comp_complex_ptr = (complex<T>*)sdp_mem_data(
            clean_comp_complex_mem
    );
    sdp_mem_clear_contents(clean_comp_complex_mem, status);

    sdp_Mem* cbeam_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            psf_shape,
            status
    );
    complex<T>* cbeam_ptr = (complex<T>*)sdp_mem_data(cbeam_mem);
    sdp_mem_clear_contents(cbeam_mem, status);

    sdp_Mem* cur_scaled_residual_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            dirty_img_shape,
            status
    );
    complex<T>* cur_scaled_residual_ptr = (complex<T>*)sdp_mem_data(
            cur_scaled_residual_mem
    );
    sdp_mem_clear_contents(cur_scaled_residual_mem, status);

    sdp_Mem* scaled_residuals_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            3,
            scaled_residuals_shape,
            status
    );
    complex<T>* scaled_residuals_ptr = (complex<T>*)sdp_mem_data(
            scaled_residuals_mem
    );
    sdp_mem_clear_contents(scaled_residuals_mem, status);

    sdp_Mem* scaled_psf_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            4,
            scaled_psf_shape,
            status
    );
    complex<T>* scaled_psf_ptr = (complex<T>*)sdp_mem_data(scaled_psf_mem);
    sdp_mem_clear_contents(scaled_psf_mem, status);

    sdp_Mem* cur_scaled_psf_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            psf_shape,
            status
    );
    complex<T>* cur_scaled_psf_ptr = (complex<T>*)sdp_mem_data(
            cur_scaled_psf_mem
    );
    sdp_mem_clear_contents(cur_scaled_psf_mem, status);

    sdp_Mem* cur_scaled_psf_pre_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            psf_shape,
            status
    );
    sdp_mem_clear_contents(cur_scaled_psf_pre_mem, status);

    sdp_Mem* scale_kern_list_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            3,
            scale_kern_list_shape,
            status
    );
    complex<T>* scale_kern_list_ptr = (complex<T>*)sdp_mem_data(
            scale_kern_list_mem
    );
    sdp_mem_clear_contents(scale_kern_list_mem, status);

    sdp_Mem* cur_1_scale_kern_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            psf_shape,
            status
    );
    complex<T>* cur_1_scale_kern_ptr = (complex<T>*)sdp_mem_data(
            cur_1_scale_kern_mem
    );
    sdp_mem_clear_contents(cur_1_scale_kern_mem, status);

    sdp_Mem* cur_2_scale_kern_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            psf_shape,
            status
    );
    complex<T>* cur_2_scale_kern_ptr = (complex<T>*)sdp_mem_data(
            cur_2_scale_kern_mem
    );
    sdp_mem_clear_contents(cur_2_scale_kern_mem, status);

    sdp_Mem* coupling_matrix_mem = sdp_mem_create(data_type,
            SDP_MEM_CPU,
            2,
            coupling_matrix_shape,
            status
    );
    T* coupling_matrix_ptr = (T*)sdp_mem_data(coupling_matrix_mem);
    sdp_mem_clear_contents(coupling_matrix_mem, status);

    sdp_Mem* peak_per_scale_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            1,
            scale_list_shape,
            status
    );
    complex<T>* peak_per_scale_ptr = (complex<T>*)sdp_mem_data(
            peak_per_scale_mem
    );
    sdp_mem_clear_contents(peak_per_scale_mem, status);

    sdp_Mem* index_per_scale_mem = sdp_mem_create(SDP_MEM_INT,
            SDP_MEM_CPU,
            1,
            scale_list_shape,
            status
    );
    int* index_per_scale_ptr = (int*)sdp_mem_data(index_per_scale_mem);
    sdp_mem_clear_contents(index_per_scale_mem, status);

    sdp_Mem* x_per_scale_mem = sdp_mem_create(SDP_MEM_INT,
            SDP_MEM_CPU,
            1,
            scale_list_shape,
            status
    );
    int* x_per_scale_ptr = (int*)sdp_mem_data(x_per_scale_mem);
    sdp_mem_clear_contents(x_per_scale_mem, status);

    sdp_Mem* y_per_scale_mem = sdp_mem_create(SDP_MEM_INT,
            SDP_MEM_CPU,
            1,
            scale_list_shape,
            status
    );
    int* y_per_scale_ptr = (int*)sdp_mem_data(y_per_scale_mem);
    sdp_mem_clear_contents(y_per_scale_mem, status);

    // Convolution code only works with complex input, so make residual and psf complex
    sdp_create_copy_complex<T>(residual_ptr,
            dirty_img_size,
            residual_complex_ptr
    );
    sdp_create_copy_complex<T>(psf_ptr, psf_size, psf_complex_ptr);

    // create CLEAN beam
    SDP_LOG_DEBUG("Creating CLEAN beam");
    sdp_create_cbeam<T>(cbeam_details_ptr, psf_dim, cbeam_ptr);

    // create scale kernels
    SDP_LOG_DEBUG("Creating Scale Kernels");
    sdp_create_scale_kern<T>(scale_kern_list_ptr,
            scale_list_ptr,
            scale_dim,
            psf_dim
    );

    SDP_LOG_DEBUG("Creating Scaled PSF and Dirty image");

    // scale psf and dirty image with scale kernel
    for (int s = 0; s < scale_dim; s++)
    {
        SDP_LOG_DEBUG("Loading Kernel %d", s);
        // copy first kernel for current operation
        for (int x = 0; x < psf_dim; x++)
        {
            for (int y = 0; y < psf_dim; y++)
            {
                const unsigned int i_list_s = INDEX_3D(scale_dim,
                        psf_dim,
                        psf_dim,
                        s,
                        x,
                        y
                );
                const unsigned int i_cur = INDEX_2D(psf_dim, psf_dim, x, y);
                cur_1_scale_kern_ptr[i_cur] = scale_kern_list_ptr[i_list_s];
            }
        }

        for (int p = 0; p < scale_dim; p++)
        {
            // copy second kernel for current operation
            SDP_LOG_DEBUG("Loading Kernel %d", p);
            for (int x = 0; x < psf_dim; x++)
            {
                for (int y = 0; y < psf_dim; y++)
                {
                    const unsigned int i_list_p = INDEX_3D(scale_dim,
                            psf_dim,
                            psf_dim,
                            p,
                            x,
                            y
                    );
                    const unsigned int i_cur = INDEX_2D(psf_dim, psf_dim, x, y);
                    cur_2_scale_kern_ptr[i_cur] = scale_kern_list_ptr[i_list_p];
                }
            }

            // scale psf twice
            SDP_LOG_DEBUG("Scaling PSF %d, %d", s, p);
            sdp_fft_convolution(psf_complex_mem,
                    cur_1_scale_kern_mem,
                    cur_scaled_psf_pre_mem,
                    status
            );
            sdp_fft_convolution(cur_scaled_psf_pre_mem,
                    cur_2_scale_kern_mem,
                    cur_scaled_psf_mem,
                    status
            );

            // load scaled psf to scaled psf list
            for (int x = 0; x < psf_dim; x++)
            {
                for (int y = 0; y < psf_dim; y++)
                {
                    const unsigned int i_scaled_psf_list = INDEX_4D(scale_dim,
                            scale_dim,
                            psf_dim,
                            psf_dim,
                            s,
                            p,
                            x,
                            y
                    );
                    const unsigned int i_cur = INDEX_2D(psf_dim, psf_dim, x, y);
                    scaled_psf_ptr[i_scaled_psf_list] =
                            cur_scaled_psf_ptr[i_cur];
                }
            }
        }

        // scale dirty image once
        SDP_LOG_DEBUG("Scaling dirty image %d", s);
        sdp_fft_convolution(residual_complex_mem,
                cur_1_scale_kern_mem,
                cur_scaled_residual_mem,
                status
        );

        // load scaled residual to scaled residual list
        for (int x = 0; x < dirty_img_dim; x++)
        {
            for (int y = 0; y < dirty_img_dim; y++)
            {
                const unsigned int i_list = INDEX_3D(scale_dim,
                        dirty_img_dim,
                        dirty_img_dim,
                        s,
                        x,
                        y
                );
                const unsigned int i_cur = INDEX_2D(dirty_img_dim,
                        dirty_img_dim,
                        x,
                        y
                );
                scaled_residuals_ptr[i_list] = cur_scaled_residual_ptr[i_cur];
                // skymodel_ptr[i_cur] = std::real(scaled_residuals_ptr[i_cur]);
            }
        }
    }

    // evaluate the coupling matrix
    SDP_LOG_DEBUG("Evaluate coupling matrix");
    double max_scaled_psf = 0;
    for (int s = 0; s < scale_dim; s++)
    {
        for (int p = 0; p < scale_dim; p++)
        {
            for (int x = 0; x < psf_dim; x++)
            {
                for (int y = 0; y < psf_dim; y++)
                {
                    const unsigned int i_scaled_psf_list = INDEX_4D(scale_dim,
                            scale_dim,
                            psf_dim,
                            psf_dim,
                            s,
                            p,
                            x,
                            y
                    );
                    if (std::real(scaled_psf_ptr[i_scaled_psf_list]) >
                            max_scaled_psf)
                    {
                        max_scaled_psf =
                                std::real(scaled_psf_ptr[i_scaled_psf_list]);
                    }
                }
            }
            const unsigned int i_cur = INDEX_2D(scale_dim, scale_dim, s, p);
            coupling_matrix_ptr[i_cur] = max_scaled_psf;
            max_scaled_psf = 0;
        }
    }

    // set up some loop variables
    int cur_cycle = 0;
    T max_scaled_biased = 0;
    int max_scale = 0;

    // CLEAN loop executes while the stop conditions (threashold and cycle limit) are not met
    while (cur_cycle < cycle_limit)
    {
        SDP_LOG_DEBUG("Current Cycle %d of %d", cur_cycle, cycle_limit);

        sdp_mem_clear_contents(peak_per_scale_mem, status);
        max_scale = 0;
        max_scaled_biased = 0;
        sdp_mem_clear_contents(index_per_scale_mem, status);
        sdp_mem_clear_contents(x_per_scale_mem, status);
        sdp_mem_clear_contents(y_per_scale_mem, status);

        // find the peak at each scale
        for (int i = 0; i < scale_dim; i++)
        {
            for (int x = 0; x < dirty_img_dim; x++)
            {
                for (int y = 0; y < dirty_img_dim; y++)
                {
                    const unsigned int i_list = INDEX_3D(scale_dim,
                            dirty_img_dim,
                            dirty_img_dim,
                            i,
                            x,
                            y
                    );
                    if (std::real(scaled_residuals_ptr[i_list]) >
                            std::real(peak_per_scale_ptr[i]))
                    {
                        peak_per_scale_ptr[i] = scaled_residuals_ptr[i_list];
                        index_per_scale_ptr[i] = i_list;
                        x_per_scale_ptr[i] = x;
                        y_per_scale_ptr[i] = y;
                    }
                }
            }
        }

        // bias with coupling matrix to find overall peak for all scales
        for (int i = 0; i < scale_dim; i++)
        {
            const unsigned int i_cur = INDEX_2D(scale_dim, scale_dim, i, i);
            // SDP_LOG_DEBUG("Current max pre bias %f" , std::real(peak_per_scale_ptr[i]));
            peak_per_scale_ptr[i] /= coupling_matrix_ptr[i_cur];
        }

        // find overall peak
        max_scaled_biased = 0;
        for (int i = 0; i < scale_dim; i++)
        {
            if (std::real(peak_per_scale_ptr[i]) > max_scaled_biased)
            {
                max_scaled_biased = std::real(peak_per_scale_ptr[i]);
                max_scale = i;
            }
        }

        // check maximum value against threshold
        const unsigned int i = INDEX_3D(scale_dim,
                dirty_img_dim,
                dirty_img_dim,
                max_scale,
                x_per_scale_ptr[max_scale],
                y_per_scale_ptr[max_scale]
        );
        if (std::real(scaled_residuals_ptr[i]) < threshold)
        {
            // stop = 1;
            SDP_LOG_DEBUG("msClean stopped at %f",
                    std::real(scaled_residuals_ptr[i])
            );

            break;
        }

        // add fraction of maximum to clean components list
        // identify kernel window to add to the clean components
        int64_t kern_x_start = 0, kern_x_end = 0, kern_y_start = 0,
                kern_y_end = 0;

        kern_x_start = dirty_img_dim - x_per_scale_ptr[max_scale];
        kern_x_end = kern_x_start + dirty_img_dim;
        kern_y_start = dirty_img_dim - y_per_scale_ptr[max_scale];
        kern_y_end = kern_y_start + dirty_img_dim;

        for (int x = kern_x_start, i = 0; x < kern_x_end; x++, i++)
        {
            for (int y = kern_y_start, j = 0; y < kern_y_end; y++, j++)
            {
                const unsigned int i_clean = INDEX_2D(dirty_img_dim,
                        dirty_img_dim,
                        i,
                        j
                );
                const unsigned int i_kern = INDEX_3D(scale_dim,
                        psf_dim,
                        psf_dim,
                        max_scale,
                        x,
                        y
                );

                clean_comp_complex_ptr[i_clean] += loop_gain *
                        max_scaled_biased * scale_kern_list_ptr[i_kern];
            }
        }

        // cross subtract psf from other scales
        // identify psf window to subtract from residual
        int64_t psf_x_start = 0, psf_x_end = 0, psf_y_start = 0, psf_y_end = 0;

        psf_x_start = dirty_img_dim - x_per_scale_ptr[max_scale];
        psf_x_end = psf_x_start + dirty_img_dim;
        psf_y_start = dirty_img_dim - y_per_scale_ptr[max_scale];
        psf_y_end = psf_y_start + dirty_img_dim;

        for (int s = 0; s < scale_dim; s++)
        {
            for (int x = psf_x_start, i = 0; x < psf_x_end; x++, i++)
            {
                for (int y = psf_y_start, j = 0; y < psf_y_end; y++, j++)
                {
                    const unsigned int i_psf = INDEX_4D(scale_dim,
                            scale_dim,
                            psf_dim,
                            psf_dim,
                            s,
                            max_scale,
                            x,
                            y
                    );
                    const unsigned int i_res = INDEX_3D(scale_dim,
                            dirty_img_dim,
                            dirty_img_dim,
                            s,
                            i,
                            j
                    );

                    scaled_residuals_ptr[i_res] -=
                            (complex<T>((loop_gain * max_scaled_biased),
                            0
                            ) * scaled_psf_ptr[i_psf]);
                }
            }
        }

        // SDP_LOG_DEBUG("End of cycle %d", cur_cycle);

        cur_cycle += 1;
    }

    // convolve clean components with clean beam
    sdp_Mem* convolution_result_mem = sdp_mem_create(complex_data_type,
            SDP_MEM_CPU,
            2,
            dirty_img_shape,
            status
    );
    complex<T>* convolution_result_ptr = (complex<T>*)sdp_mem_data(
            convolution_result_mem
    );

    sdp_fft_convolution(clean_comp_complex_mem,
            cbeam_mem,
            convolution_result_mem,
            status
    );

    // complex result to real for the output
    sdp_create_copy_real(convolution_result_ptr, dirty_img_size, skymodel_ptr);
    sdp_create_copy_real(clean_comp_complex_ptr, dirty_img_size,
            clean_comp_ptr
    );

    for (int x = 0; x < dirty_img_dim; x++)
    {
        for (int y = 0; y < dirty_img_dim; y++)
        {
            const unsigned int i_res = INDEX_3D(scale_dim,
                    dirty_img_dim,
                    dirty_img_dim,
                    0,
                    x,
                    y
            );
            const unsigned int i_sky = INDEX_2D(dirty_img_dim,
                    dirty_img_dim,
                    x,
                    y
            );

            skymodel_ptr[i_sky] = skymodel_ptr[i_sky] + std::real(
                    scaled_residuals_ptr[i_res]
            );
            residual_ptr[i_sky] = std::real(scaled_residuals_ptr[i_res]);
        }
    }

    // release memory
    sdp_mem_ref_dec(psf_complex_mem);
    sdp_mem_ref_dec(clean_comp_complex_mem);
    sdp_mem_ref_dec(residual_complex_mem);
    sdp_mem_ref_dec(cbeam_mem);
    sdp_mem_ref_dec(cur_scaled_residual_mem);
    sdp_mem_ref_dec(scaled_residuals_mem);
    sdp_mem_ref_dec(scaled_psf_mem);
    sdp_mem_ref_dec(cur_scaled_psf_mem);
    sdp_mem_ref_dec(cur_scaled_psf_pre_mem);
    sdp_mem_ref_dec(scale_kern_list_mem);
    sdp_mem_ref_dec(cur_1_scale_kern_mem);
    sdp_mem_ref_dec(cur_2_scale_kern_mem);
    sdp_mem_ref_dec(coupling_matrix_mem);
    sdp_mem_ref_dec(peak_per_scale_mem);
    sdp_mem_ref_dec(index_per_scale_mem);
    sdp_mem_ref_dec(x_per_scale_mem);
    sdp_mem_ref_dec(y_per_scale_mem);
    sdp_mem_ref_dec(convolution_result_mem);
}


void sdp_ms_clean_cornwell(
        const sdp_Mem* dirty_img,
        const sdp_Mem* psf,
        const sdp_Mem* cbeam_details,
        const sdp_Mem* scale_list,
        const double loop_gain,
        const double threshold,
        const int cycle_limit,
        sdp_Mem* clean_model,
        sdp_Mem* residual,
        sdp_Mem* skymodel,
        sdp_Error* status
)
{
    if (*status) return;

    const int64_t dirty_img_dim = sdp_mem_shape_dim(dirty_img, 0);
    const int64_t psf_dim = sdp_mem_shape_dim(psf, 0);
    const int64_t scale_dim = sdp_mem_shape_dim(scale_list, 0);
    const int64_t clean_model_dim = sdp_mem_shape_dim(clean_model, 0);
    const int64_t residual_dim = sdp_mem_shape_dim(residual, 0);
    const int64_t skymodel_dim = sdp_mem_shape_dim(skymodel, 0);
    const int64_t cbeam_details_dim = sdp_mem_shape_dim(cbeam_details, 0);

    const sdp_MemLocation location = sdp_mem_location(dirty_img);
    const sdp_MemType data_type = sdp_mem_type(dirty_img);

    if (sdp_mem_is_read_only(skymodel))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output is not writable");
        return;
    }

    if (location != sdp_mem_location(skymodel))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }

    if (data_type != sdp_mem_type(psf))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The Dirty image and PSF must be of the same data type");
        return;
    }

    if (sdp_mem_type(scale_list) != SDP_MEM_INT)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR(
                "The scale list must be a list of 4 byte integers"
        );
        return;
    }

    if (data_type != sdp_mem_type(skymodel))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("The input and output must be of the same data type");
        return;
    }

    if (clean_model_dim != dirty_img_dim)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The CLEAN model and the dirty image must be the same size"
        );
        return;
    }

    if (residual_dim != dirty_img_dim)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The residual image and the dirty image must be the same size"
        );
        return;
    }

    if (skymodel_dim != dirty_img_dim)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The skymodel image and the dirty image must be the same size"
        );
        return;
    }

    if (cbeam_details_dim != 4)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR(
                "The array describing the CLEAN beam must include BMAJ, BMIN, THETA and SIZE"
        );
        return;
    }

    // copy dirty image to starting residual
    sdp_mem_copy_contents(residual, dirty_img, 0, 0,
            (dirty_img_dim * dirty_img_dim), status
    );

    if (sdp_mem_location(dirty_img) == SDP_MEM_CPU)
    {
        if (data_type == SDP_MEM_DOUBLE)
        {
            ms_clean_cornwell<double>(
                    (const double*)sdp_mem_data_const(psf),
                    (const double*)sdp_mem_data_const(cbeam_details),
                    (const int*)sdp_mem_data_const(scale_list),
                    loop_gain,
                    threshold,
                    cycle_limit,
                    dirty_img_dim,
                    psf_dim,
                    scale_dim,
                    data_type,
                    clean_model,
                    residual,
                    (double*)sdp_mem_data(skymodel),
                    status
            );
        }
        else if (data_type == SDP_MEM_FLOAT)
        {
            ms_clean_cornwell<float>(
                    (const float*)sdp_mem_data_const(psf),
                    (const float*)sdp_mem_data_const(cbeam_details),
                    (const int*)sdp_mem_data_const(scale_list),
                    (float)loop_gain,
                    (float)threshold,
                    cycle_limit,
                    dirty_img_dim,
                    psf_dim,
                    scale_dim,
                    data_type,
                    clean_model,
                    residual,
                    (float*)sdp_mem_data(skymodel),
                    status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type");
        }
    }
}
