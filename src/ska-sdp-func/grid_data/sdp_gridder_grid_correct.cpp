/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>

#include "ska-sdp-func/fourier_transforms/private_pswf.h"
#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_grid_correct.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using std::complex;

// Begin anonymous namespace for file-local functions.
namespace {

// Local function to apply grid correction for the PSWF.
template<typename T>
void grid_corr_pswf(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_Pswf* pswf_lm,
        sdp_Pswf* pswf_n,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
)
{
    if (*status) return;

    // Apply portion of shifted PSWF to facet.
    sdp_MemViewCpu<T, 2> facet_;
    sdp_MemViewCpu<const double, 1> pswf_;
    sdp_mem_check_and_view(facet, &facet_, status);
    sdp_mem_check_and_view(
            sdp_pswf_values(pswf_lm, SDP_MEM_CPU, status), &pswf_, status
    );
    if (*status) return;
    const int num_l = (int) sdp_mem_shape_dim(facet, 0);
    const int num_m = (int) sdp_mem_shape_dim(facet, 1);
    const double* pswf_n_coe = (const double*) sdp_mem_data_const(
            sdp_pswf_coeff(pswf_n, SDP_MEM_CPU, status)
    );
    const double pswf_n_c = sdp_pswf_par_c(pswf_n);
    #pragma omp parallel for
    for (int il = 0; il < num_l; ++il)
    {
        const int pl = il - num_l / 2 + facet_offset_l;
        const double l_ = pl * theta / image_size;
        const double pswf_l = pswf_(pl + image_size / 2);
        for (int im = 0; im < num_m; ++im)
        {
            double pswf_n = 1.0;
            const int pm = im - num_m / 2 + facet_offset_m;
            const double m_ = pm * theta / image_size;
            const double pswf_m = pswf_(pm + image_size / 2);
            if (pswf_n_c > 0.0)
            {
                const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
                const double n_x = fabs(n_ * 2.0 * w_step);
                pswf_n = (n_x < 1.0) ?
                            sdp_pswf_aswfa(0, 0, pswf_n_c, pswf_n_coe, n_x) :
                            1.0;
            }
            const double scale = 1.0 / (pswf_l * pswf_m * pswf_n);
            facet_(il, im) *= (T) scale;
        }
    }
}


// Local function to apply w-correction.
template<typename T>
void grid_corr_w_stack(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        int inverse,
        sdp_Error* status
)
{
    if (*status || w_offset == 0) return;
    sdp_MemViewCpu<T, 2> facet_;
    sdp_mem_check_and_view(facet, &facet_, status);
    if (*status) return;
    const int num_l = (int) sdp_mem_shape_dim(facet, 0);
    const int num_m = (int) sdp_mem_shape_dim(facet, 1);
    #pragma omp parallel for
    for (int il = 0; il < num_l; ++il)
    {
        const int pl = il - num_l / 2 + facet_offset_l;
        const double l_ = pl * theta / image_size;
        for (int im = 0; im < num_m; ++im)
        {
            const int pm = im - num_m / 2 + facet_offset_m;
            const double m_ = pm * theta / image_size;
            const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
            const double phase = 2.0 * M_PI * w_step * n_;
            complex<double> w = complex<double>(cos(phase), sin(phase));
            w = std::pow(w, w_offset);
            w = !inverse ? 1.0 / w : w;
            facet_(il, im) *= (T) w;
        }
    }
}

} // End anonymous namespace for file-local functions.


void sdp_gridder_grid_correct_pswf(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        int support,
        int w_support,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_Pswf* pswf_n = sdp_pswf_create(0, w_support * (M_PI / 2));
    sdp_Pswf* pswf_lm = sdp_pswf_create(0, support * (M_PI / 2));
    sdp_pswf_generate(pswf_lm, 0, image_size, 1, status);

    // Apply grid correction for the appropriate data type.
    const sdp_MemLocation loc = sdp_mem_location(facet);
    if (loc == SDP_MEM_CPU)
    {
        switch (sdp_mem_type(facet))
        {
        case SDP_MEM_DOUBLE:
            grid_corr_pswf<double>(
                    image_size, theta, w_step, shear_u, shear_v, pswf_lm,
                    pswf_n, facet, facet_offset_l, facet_offset_m, status
            );
            break;
        case SDP_MEM_COMPLEX_DOUBLE:
            grid_corr_pswf<complex<double> >(
                    image_size, theta, w_step, shear_u, shear_v, pswf_lm,
                    pswf_n, facet, facet_offset_l, facet_offset_m, status
            );
            break;
        case SDP_MEM_FLOAT:
            grid_corr_pswf<float>(
                    image_size, theta, w_step, shear_u, shear_v, pswf_lm,
                    pswf_n, facet, facet_offset_l, facet_offset_m, status
            );
            break;
        case SDP_MEM_COMPLEX_FLOAT:
            grid_corr_pswf<complex<float> >(
                    image_size, theta, w_step, shear_u, shear_v, pswf_lm,
                    pswf_n, facet, facet_offset_l, facet_offset_m, status
            );
            break;
        default:
            *status = SDP_ERR_DATA_TYPE;
            break;
        }
    }
    else if (loc == SDP_MEM_GPU)
    {
        uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
        const char* kernel_name = 0;
        int is_cplx = 0, is_dbl = 0;
        const int num_l = sdp_mem_shape_dim(facet, 0);
        const int num_m = sdp_mem_shape_dim(facet, 1);
        sdp_MemViewGpu<complex<double>, 2> fct_cdbl;
        sdp_MemViewGpu<complex<float>, 2> fct_cflt;
        sdp_MemViewGpu<double, 2> fct_dbl;
        sdp_MemViewGpu<float, 2> fct_flt;
        sdp_MemViewGpu<const double, 1> pswf_;
        sdp_mem_check_and_view(
                sdp_pswf_values(pswf_lm, SDP_MEM_GPU, status), &pswf_, status
        );
        const sdp_Mem* pswf_n_coe = sdp_pswf_coeff(pswf_n, SDP_MEM_GPU, status);
        const double pswf_n_c = sdp_pswf_par_c(pswf_n);
        switch (sdp_mem_type(facet))
        {
        case SDP_MEM_FLOAT:
            is_cplx = 0;
            is_dbl = 0;
            sdp_mem_check_and_view(facet, &fct_flt, status);
            kernel_name = "sdp_gridder_grid_correct_pswf<float>";
            break;
        case SDP_MEM_DOUBLE:
            is_cplx = 0;
            is_dbl = 1;
            sdp_mem_check_and_view(facet, &fct_dbl, status);
            kernel_name = "sdp_gridder_grid_correct_pswf<double>";
            break;
        case SDP_MEM_COMPLEX_FLOAT:
            is_cplx = 1;
            is_dbl = 0;
            sdp_mem_check_and_view(facet, &fct_cflt, status);
            kernel_name = "sdp_gridder_grid_correct_pswf<complex<float> >";
            break;
        case SDP_MEM_COMPLEX_DOUBLE:
            is_cplx = 1;
            is_dbl = 1;
            sdp_mem_check_and_view(facet, &fct_cdbl, status);
            kernel_name = "sdp_gridder_grid_correct_pswf<complex<double> >";
            break;
        default:
            *status = SDP_ERR_DATA_TYPE;
            break;
        }
        const void* arg[] = {
            (const void*) &image_size,
            (const void*) &theta,
            (const void*) &w_step,
            (const void*) &shear_u,
            (const void*) &shear_v,
            (const void*) &pswf_,
            sdp_mem_gpu_buffer_const(pswf_n_coe, status),
            (const void*) &pswf_n_c,
            is_cplx ?
                (is_dbl ? (const void*) &fct_cdbl : (const void*) &fct_cflt) :
                (is_dbl ? (const void*) &fct_dbl : (const void*) &fct_flt),
            (const void*) &facet_offset_l,
            (const void*) &facet_offset_m
        };
        num_blocks[0] = (num_l + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (num_m + num_threads[1] - 1) / num_threads[1];
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
    sdp_pswf_free(pswf_n);
    sdp_pswf_free(pswf_lm);
}


void sdp_gridder_grid_correct_w_stack(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        int inverse,
        sdp_Error* status
)
{
    if (*status || w_offset == 0) return;
    const sdp_MemLocation loc = sdp_mem_location(facet);
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_type(facet) == SDP_MEM_COMPLEX_DOUBLE)
        {
            grid_corr_w_stack<complex<double> >(
                    image_size, theta, w_step, shear_u, shear_v, facet,
                    facet_offset_l, facet_offset_m, w_offset, inverse, status
            );
        }
        else if (sdp_mem_type(facet) == SDP_MEM_COMPLEX_FLOAT)
        {
            grid_corr_w_stack<complex<float> >(
                    image_size, theta, w_step, shear_u, shear_v, facet,
                    facet_offset_l, facet_offset_m, w_offset, inverse, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else if (loc == SDP_MEM_GPU)
    {
        uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
        const char* kernel_name = 0;
        int is_dbl = 0;
        const int num_l = sdp_mem_shape_dim(facet, 0);
        const int num_m = sdp_mem_shape_dim(facet, 1);
        sdp_MemViewGpu<complex<double>, 2> facet_dbl;
        sdp_MemViewGpu<complex<float>, 2> facet_flt;
        if (sdp_mem_type(facet) == SDP_MEM_COMPLEX_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(facet, &facet_dbl, status);
            kernel_name = "sdp_gridder_grid_correct_w_stack<complex<double> >";
        }
        else if (sdp_mem_type(facet) == SDP_MEM_COMPLEX_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(facet, &facet_flt, status);
            kernel_name = "sdp_gridder_grid_correct_w_stack<complex<float> >";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
        const void* arg[] = {
            (const void*) &image_size,
            (const void*) &theta,
            (const void*) &w_step,
            (const void*) &shear_u,
            (const void*) &shear_v,
            is_dbl ? (const void*) &facet_dbl : (const void*) &facet_flt,
            (const void*) &facet_offset_l,
            (const void*) &facet_offset_m,
            (const void*) &w_offset,
            (const void*) &inverse
        };
        num_blocks[0] = (num_l + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (num_m + num_threads[1] - 1) / num_threads[1];
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}
