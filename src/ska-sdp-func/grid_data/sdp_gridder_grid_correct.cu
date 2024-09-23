/* See the LICENSE file at the top-level directory of this distribution. */

#include <thrust/complex.h>

#include "ska-sdp-func/fourier_transforms/private_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using thrust::complex;

template<typename T>
__global__ void sdp_gridder_grid_correct_pswf(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        const sdp_MemViewGpu<const double, 1> pswf,
        const double* const __restrict__ pswf_n_coeff,
        double pswf_n_c,
        sdp_MemViewGpu<T, 2> facet,
        int facet_offset_l,
        int facet_offset_m
)
{
    const int64_t il = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t im = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t num_l = facet.shape[0];
    const int64_t num_m = facet.shape[1];
    if (il >= num_l || im >= num_m) return;
    const int pl = il - num_l / 2 + facet_offset_l;
    const int pm = im - num_m / 2 + facet_offset_m;
    const double l_ = pl * theta / image_size;
    const double m_ = pm * theta / image_size;
    const double pswf_l = pswf(pl + image_size / 2);
    const double pswf_m = pswf(pm + image_size / 2);
    double pswf_n = 1.0;
    if (pswf_n_c > 0.0)
    {
        const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
        const double n_x = fabs(n_ * 2.0 * w_step);
        pswf_n = (n_x < 1.0) ?
                    sdp_pswf_aswfa(0, 0, pswf_n_c, pswf_n_coeff, n_x) : 1.0;
    }
    const double scale = 1.0 / (pswf_l * pswf_m * pswf_n);
    facet(il, im) *= (T) scale;
}


template<typename T>
__global__ void sdp_gridder_grid_correct_w_stack(
        int image_size,
        double theta,
        double w_step,
        double shear_u,
        double shear_v,
        sdp_MemViewGpu<T, 2> facet,
        int facet_offset_l,
        int facet_offset_m,
        int w_offset,
        int inverse
)
{
    const int64_t il = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t im = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t num_l = facet.shape[0];
    const int64_t num_m = facet.shape[1];
    if (il >= num_l || im >= num_m) return;
    const int pl = il - num_l / 2 + facet_offset_l;
    const int pm = im - num_m / 2 + facet_offset_m;
    const double l_ = pl * theta / image_size;
    const double m_ = pm * theta / image_size;
    const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
    const double phase = 2.0 * M_PI * w_step * n_;
    complex<double> w = complex<double>(cos(phase), sin(phase));
    w = pow(w, w_offset);
    w = !inverse ? 1.0 / w : w;
    facet(il, im) *= (T) w;
}


// *INDENT-OFF*
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_pswf<double>)
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_pswf<float>)
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_pswf<complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_pswf<complex<float> >)

SDP_CUDA_KERNEL(sdp_gridder_grid_correct_w_stack<complex<double> >)
SDP_CUDA_KERNEL(sdp_gridder_grid_correct_w_stack<complex<float> >)
// *INDENT-ON*
