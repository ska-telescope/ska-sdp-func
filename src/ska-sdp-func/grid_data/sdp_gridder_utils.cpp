/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"

using std::complex;

// Begin anonymous namespace for file-local functions.
namespace {

// Convert (l, m) to (n) directions, allowing for shear.
template<typename T>
T lm_to_n(const T& l, const T& m, const T& h_u, const T& h_v)
{
    // Easy case.
    if (h_u == 0 and h_v == 0)
        return sqrt(1 - l * l - m * m) - 1;

    // Sheared case.
    const T hul_hvm_1 = h_u * l + h_v * m - 1; // = -1 with h_u = h_v = 0
    const T hu2_hv2_1 = h_u * h_u + h_v * h_v + 1; // = 1 with h_u = h_v = 0
    return (
        sqrt(hul_hvm_1 * hul_hvm_1 - hu2_hv2_1 * (l * l + m * m)) +
        hul_hvm_1
    ) / hu2_hv2_1;
}


// Convert all image pixel positions to coordinates.
template<typename T>
void image_to_lmn(
        int image_size,
        double theta,
        double shear_u,
        double shear_v,
        sdp_Mem* l,
        sdp_Mem* m,
        sdp_Mem* n
)
{
    // Store pixel data.
    T* l_ = l ? (T*) sdp_mem_data(l) : NULL;
    T* m_ = m ? (T*) sdp_mem_data(m) : NULL;
    T* n_ = n ? (T*) sdp_mem_data(n) : NULL;
    for (int il = 0, k = 0; il < image_size; ++il)
    {
        for (int im = 0; im < image_size; ++im, ++k)
        {
            const double local_l = (il - image_size / 2) * theta / image_size;
            const double local_m = (im - image_size / 2) * theta / image_size;
            if (l_) l_[k] = local_l;
            if (m_) m_[k] = local_m;
            if (n_) n_[k] = lm_to_n(local_l, local_m, shear_u, shear_v);
        }
    }
}


// Make an oversampled uv-space kernel from an image-space window function.
template<typename T>
void make_kernel(const sdp_Mem* window, sdp_Mem* kernel, sdp_Error* status)
{
    if (*status) return;
    if (sdp_mem_num_dims(window) != 1 || sdp_mem_num_dims(kernel) != 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
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
    const T* window_ = (const T*) sdp_mem_data_const(window);
    T* kernel_ = (T*) sdp_mem_data(kernel);

    // There are (oversampling + 1) rows and (support) columns in the output.
    #pragma omp parallel for collapse(2)
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
                val += window_[s_in] * cos(2 * M_PI * u * l); // real part only
            }
            kernel_[i * support + s_out] = (T) val * inv_support;
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
        double exponent
)
{
    const int64_t num_elements = sdp_mem_num_elements(out);
    OUT_TYPE* out_ = (OUT_TYPE*) sdp_mem_data(out);
    const IN1_TYPE* in1_ = (const IN1_TYPE*) sdp_mem_data_const(in1);
    const IN2_TYPE* in2_ = (const IN2_TYPE*) sdp_mem_data_const(in2);
    if (exponent == 1.0)
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
        double uvw_max[3]
)
{
    const double SPEED_OF_LIGHT = 299792458.0;
    const UVW_TYPE* uvws_ = (const UVW_TYPE*) sdp_mem_data_const(uvws);
    const int* start_chs_ = (const int*) sdp_mem_data_const(start_chs);
    const int* end_chs_ = (const int*) sdp_mem_data_const(end_chs);
    const int64_t num_uvw = sdp_mem_shape_dim(uvws, 0);
    uvw_min[0] = uvw_min[1] = uvw_min[2] = INFINITY;
    uvw_max[0] = uvw_max[1] = uvw_max[2] = -INFINITY;
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        const int start_ch = start_chs_[i], end_ch = end_chs_[i];
        if (start_ch >= end_ch)
            continue;
        const double uvw[] = {uvws_[3 * i], uvws_[3 * i + 1], uvws_[3 * i + 2]};
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


void sdp_gridder_image_to_lmn(
        int image_size,
        double theta,
        double shear_u,
        double shear_v,
        sdp_Mem* l,
        sdp_Mem* m,
        sdp_Mem* n,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemType type = (
        l ? sdp_mem_type(l) : m ? sdp_mem_type(m) :
                n ? sdp_mem_type(n) : SDP_MEM_VOID
    );
    if (type == SDP_MEM_DOUBLE)
    {
        image_to_lmn<double>(image_size, theta, shear_u, shear_v, l, m, n);
    }
    else if (type == SDP_MEM_FLOAT)
    {
        image_to_lmn<float>(image_size, theta, shear_u, shear_v, l, m, n);
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
    if (sdp_mem_location(w_pattern) != SDP_MEM_CPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
        return;
    }
    if (sdp_mem_type(w_pattern) != SDP_MEM_COMPLEX_DOUBLE)
    {
        *status = SDP_ERR_DATA_TYPE;
        return;
    }
    const int num_pix = subgrid_size * subgrid_size;
    const int64_t ns_shape[] = {(int64_t) num_pix};
    sdp_Mem* ns = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, ns_shape, status
    );
    sdp_gridder_image_to_lmn(
            subgrid_size, theta, shear_u, shear_v, 0, 0, ns, status
    );
    double* ns_ = (double*) sdp_mem_data(ns);
    complex<double>* w_pattern_ = (complex<double>*) sdp_mem_data(w_pattern);
    if (!*status)
    {
        #pragma omp parallel for
        for (int i = 0; i < num_pix; ++i)
        {
            const double phase = 2.0 * M_PI * w_step * ns_[i];
            w_pattern_[i] = complex<double>(cos(phase), sin(phase));
        }
    }
    sdp_mem_free(ns);
}


void sdp_gridder_scale_inv_array(
        sdp_Mem* out,
        const sdp_Mem* in1,
        const sdp_Mem* in2,
        double exponent,
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
                uvws, freq0_hz, dfreq_hz, start_chs, end_chs, uvw_min, uvw_max
        );
    }
    else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT)
    {
        uvw_bounds_all<float>(
                uvws, freq0_hz, dfreq_hz, start_chs, end_chs, uvw_min, uvw_max
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }
}
