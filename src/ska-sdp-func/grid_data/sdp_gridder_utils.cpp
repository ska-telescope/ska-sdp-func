/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using std::complex;

// Begin anonymous namespace for file-local functions.
namespace {

// Local function to multiply every element in an array by those in another
// array raised to the given exponent, accumulating the results in the output:
// out += in1 * in2 ** exponent
template<typename OUT_TYPE, typename IN1_TYPE, typename IN2_TYPE>
void accum_scale_array(
        sdp_Mem* out,
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        int exponent
)
{
    const int64_t num_elements = sdp_mem_num_elements(out);
    OUT_TYPE* out_ = (OUT_TYPE*) sdp_mem_data(out);
    const IN1_TYPE* in1_ = (const IN1_TYPE*) sdp_mem_data_const(in1);
    const IN2_TYPE* in2_ = in2 ? (const IN2_TYPE*) sdp_mem_data_const(in2) : 0;
    if (in2_)
    {
        if (exponent == 1)
        {
            for (int64_t i = 0; i < num_elements; ++i)
                out_[i] += (IN2_TYPE) in1_[i] * in2_[i];
        }
        else
        {
            for (int64_t i = 0; i < num_elements; ++i)
                out_[i] += (IN2_TYPE) in1_[i] * pow(in2_[i], exponent);
        }
    }
    else
    {
        for (int64_t i = 0; i < num_elements; ++i)
            out_[i] += (OUT_TYPE) in1_[i];
    }
}


// Make an oversampled uv-space kernel from an image-space window function.
template<typename T>
void make_kernel(const sdp_Mem* window, sdp_Mem* kernel, sdp_Error* status)
{
    if (*status) return;
    sdp_MemViewCpu<const T, 1> window_;
    sdp_MemViewCpu<T, 2> kernel_;
    sdp_mem_check_and_view(window, &window_, status);
    sdp_mem_check_and_view(kernel, &kernel_, status);
    if (*status) return;
    const int support = (int) sdp_mem_shape_dim(window, 0);
    const int oversampling = (int) sdp_mem_shape_dim(kernel, 0) - 1;
    if (support != (int) sdp_mem_shape_dim(kernel, 1))
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    const double inv_support = 1.0 / support;
    const double inv_oversampling = 1.0 / oversampling;
    const int half_suppport = support / 2;

    // There are (oversampling + 1) rows and (support) columns in the output.
    #pragma omp parallel for
    for (int i = 0; i <= oversampling; ++i)
    {
        for (int s_out = 0; s_out < support; ++s_out)
        {
            // Range of du is [-oversampling, 0].
            // Range of u is [-support // 2, support // 2) - du / oversampling.
            const double du = (double) (i - oversampling);
            const double u = (s_out - half_suppport) - du * inv_oversampling;

            // Take the real part of the DFT of the window.
            double val = 0.0;
            for (int s_in = 0; s_in < support; ++s_in)
            {
                const double l = (s_in - half_suppport) * inv_support;
                val += window_(s_in) * cos(2 * M_PI * u * l); // real part only
            }
            kernel_(i, s_out) = (T) val * inv_support;
        }
    }
}


// Local function to divide every element in an array by those in another array
// raised to the given exponent: out = in1 / in2 ** exponent
template<typename OUT_TYPE, typename IN1_TYPE, typename IN2_TYPE>
void scale_inv_array(
        sdp_Mem* out,
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        int exponent
)
{
    const int64_t num_elements = sdp_mem_num_elements(out);
    OUT_TYPE* out_ = (OUT_TYPE*) sdp_mem_data(out);
    const IN1_TYPE* in1_ = (const IN1_TYPE*) sdp_mem_data_const(in1);
    const IN2_TYPE* in2_ = (const IN2_TYPE*) sdp_mem_data_const(in2);
    if (exponent == 1)
    {
        for (int64_t i = 0; i < num_elements; ++i)
            out_[i] = (IN2_TYPE) in1_[i] / in2_[i];
    }
    else
    {
        for (int64_t i = 0; i < num_elements; ++i)
            out_[i] = (IN2_TYPE) in1_[i] / pow(in2_[i], exponent);
    }
}


// Determine (scaled) min and max values in uvw coordinates.
template<typename UVW_TYPE>
void uvw_bounds_all(
        const sdp_Mem* uvws,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double uvw_min[3],
        double uvw_max[3],
        sdp_Error* status
)
{
    const double SPEED_OF_LIGHT = 299792458.0;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    const int64_t num_uvw = sdp_mem_shape_dim(uvws, 0);
    uvw_min[0] = uvw_min[1] = uvw_min[2] = INFINITY;
    uvw_max[0] = uvw_max[1] = uvw_max[2] = -INFINITY;
    if (*status) return;
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        const int start_ch = start_chs_(i), end_ch = end_chs_(i);
        if (start_ch >= end_ch)
            continue;
        const double uvw[] = {uvws_(i, 0), uvws_(i, 1), uvws_(i, 2)};
        for (int j = 0; j < 3; ++j)
        {
            const double u0 = freq0_hz * uvw[j] / SPEED_OF_LIGHT;
            const double du = dfreq_hz * uvw[j] / SPEED_OF_LIGHT;
            if (uvw[j] >= 0)
            {
                uvw_min[j] = std::min(u0 + start_ch * du, uvw_min[j]);
                uvw_max[j] = std::max(u0 + (end_ch - 1) * du, uvw_max[j]);
            }
            else
            {
                uvw_max[j] = std::max(u0 + start_ch * du, uvw_max[j]);
                uvw_min[j] = std::min(u0 + (end_ch - 1) * du, uvw_min[j]);
            }
        }
    }
}

} // End anonymous namespace for file-local functions.


void sdp_gridder_accumulate_scaled_arrays(
        sdp_Mem* out,
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        int exponent,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemType type_out = sdp_mem_type(out);
    sdp_MemType type1 = sdp_mem_type(in1);
    sdp_MemType type2 = in2 ? sdp_mem_type(in2) : SDP_MEM_COMPLEX_DOUBLE;
    if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
            type1 == SDP_MEM_DOUBLE && type2 == SDP_MEM_COMPLEX_DOUBLE)
    {
        accum_scale_array<complex<double>, double, complex<double> >(
                out, in1, in2, exponent
        );
    }
    else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
            type1 == SDP_MEM_FLOAT && type2 == SDP_MEM_COMPLEX_DOUBLE)
    {
        accum_scale_array<complex<float>, float, complex<double> >(
                out, in1, in2, exponent
        );
    }
    else if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
            type1 == SDP_MEM_COMPLEX_DOUBLE && type2 == SDP_MEM_COMPLEX_DOUBLE)
    {
        accum_scale_array<complex<double>, complex<double>, complex<double> >(
                out, in1, in2, exponent
        );
    }
    else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
            type1 == SDP_MEM_COMPLEX_FLOAT && type2 == SDP_MEM_COMPLEX_DOUBLE)
    {
        accum_scale_array<complex<float>, complex<float>, complex<double> >(
                out, in1, in2, exponent
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
}


void sdp_gridder_make_kernel(
        const sdp_Mem* window,
        sdp_Mem* kernel,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_type(window) == SDP_MEM_DOUBLE &&
            sdp_mem_type(kernel) == SDP_MEM_DOUBLE)
    {
        make_kernel<double>(window, kernel, status);
    }
    else if (sdp_mem_type(window) == SDP_MEM_FLOAT &&
            sdp_mem_type(kernel) == SDP_MEM_FLOAT)
    {
        make_kernel<float>(window, kernel, status);
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
}


void sdp_gridder_make_pswf_kernel(
        int support,
        sdp_Mem* kernel,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(kernel) != 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    const int64_t vr_size = sdp_mem_shape_dim(kernel, 1);
    const int64_t pswf_shape[] = {vr_size};
    sdp_Mem* pswf = sdp_mem_create(
            sdp_mem_type(kernel), SDP_MEM_CPU, 1, pswf_shape, status
    );
    sdp_generate_pswf(0, support * (M_PI / 2), pswf, status);
    if (vr_size % 2 == 0) ((double*) sdp_mem_data(pswf))[0] = 1e-15;
    sdp_gridder_make_kernel(pswf, kernel, status);
    sdp_mem_free(pswf);
}


void sdp_gridder_make_w_pattern(
        int subgrid_size,
        double theta,
        double shear_u,
        double shear_v,
        double w_step,
        sdp_Mem* w_pattern,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<complex<double>, 2> w_pattern_;
    sdp_mem_check_and_view(w_pattern, &w_pattern_, status);
    const int half_size = subgrid_size / 2;
    if (*status) return;
    #pragma omp parallel for
    for (int il = 0; il < subgrid_size; ++il)
    {
        for (int im = 0; im < subgrid_size; ++im)
        {
            const double l_ = (il - half_size) * theta / subgrid_size;
            const double m_ = (im - half_size) * theta / subgrid_size;
            const double n_ = lm_to_n(l_, m_, shear_u, shear_v);
            const double phase = 2.0 * M_PI * w_step * n_;
            w_pattern_(il, im) = complex<double>(cos(phase), sin(phase));
        }
    }
}


void sdp_gridder_scale_inv_array(
        sdp_Mem* out,
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        int exponent,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_type(out) == SDP_MEM_COMPLEX_DOUBLE &&
            sdp_mem_type(in1) == SDP_MEM_DOUBLE &&
            sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
    {
        scale_inv_array<complex<double>, double, complex<double> >(
                out, in1, in2, exponent
        );
    }
    else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_FLOAT &&
            sdp_mem_type(in1) == SDP_MEM_FLOAT &&
            sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
    {
        scale_inv_array<complex<float>, float, complex<double> >(
                out, in1, in2, exponent
        );
    }
    else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_DOUBLE &&
            sdp_mem_type(in1) == SDP_MEM_COMPLEX_DOUBLE &&
            sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
    {
        scale_inv_array<complex<double>, complex<double>, complex<double> >(
                out, in1, in2, exponent
        );
    }
    else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_FLOAT &&
            sdp_mem_type(in1) == SDP_MEM_COMPLEX_FLOAT &&
            sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
    {
        scale_inv_array<complex<float>, complex<float>, complex<double> >(
                out, in1, in2, exponent
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
}


void sdp_gridder_uvw_bounds_all(
        const sdp_Mem* uvws,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double uvw_min[3],
        double uvw_max[3],
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE)
    {
        uvw_bounds_all<double>(
                uvws, freq0_hz, dfreq_hz, start_chs, end_chs, uvw_min, uvw_max,
                status
        );
    }
    else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT)
    {
        uvw_bounds_all<float>(
                uvws, freq0_hz, dfreq_hz, start_chs, end_chs, uvw_min, uvw_max,
                status
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
}
