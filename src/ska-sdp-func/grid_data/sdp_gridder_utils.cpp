/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
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
        int exponent,
        sdp_Error* status
)
{
    sdp_MemViewCpu<OUT_TYPE, 2> out_;
    sdp_MemViewCpu<const IN1_TYPE, 2> in1_;
    sdp_mem_check_and_view(out, &out_, status);
    sdp_mem_check_and_view(in1, &in1_, status);
    const int64_t shape0 = out_.shape[0];
    const int64_t shape1 = out_.shape[1];
    if (in2)
    {
        sdp_MemViewCpu<const IN2_TYPE, 2> in2_;
        sdp_mem_check_and_view(in2, &in2_, status);
        if (exponent == 1)
        {
            for (int64_t i = 0; i < shape0; ++i)
                for (int64_t j = 0; j < shape1; ++j)
                    out_(i, j) += (IN2_TYPE) in1_(i, j) * in2_(i, j);
        }
        else
        {
            for (int64_t i = 0; i < shape0; ++i)
                for (int64_t j = 0; j < shape1; ++j)
                    out_(i, j) += (IN2_TYPE) in1_(i, j) * pow(
                            in2_(i, j), exponent
                    );
        }
    }
    else
    {
        for (int64_t i = 0; i < shape0; ++i)
            for (int64_t j = 0; j < shape1; ++j)
                out_(i, j) += in1_(i, j);
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
        int exponent,
        sdp_Error* status
)
{
    sdp_MemViewCpu<OUT_TYPE, 2> out_;
    sdp_MemViewCpu<const IN1_TYPE, 2> in1_;
    sdp_MemViewCpu<const IN2_TYPE, 2> in2_;
    sdp_mem_check_and_view(out, &out_, status);
    sdp_mem_check_and_view(in1, &in1_, status);
    sdp_mem_check_and_view(in2, &in2_, status);
    const int64_t shape0 = out_.shape[0];
    const int64_t shape1 = out_.shape[1];
    if (exponent == 1)
    {
        for (int64_t i = 0; i < shape0; ++i)
            for (int64_t j = 0; j < shape1; ++j)
                out_(i, j) = (IN2_TYPE) in1_(i, j) / in2_(i, j);
    }
    else
    {
        for (int64_t i = 0; i < shape0; ++i)
            for (int64_t j = 0; j < shape1; ++j)
                out_(i, j) = (IN2_TYPE) in1_(i, j) / pow(in2_(i, j), exponent);
    }
}


// Determine (scaled) min and max values in uvw coordinates.
template<typename UVW_TYPE>
void uvw_bounds_all(
        const sdp_MemViewCpu<const UVW_TYPE, 2>& uvws,
        double freq0_hz,
        double dfreq_hz,
        const sdp_MemViewCpu<const int, 1>& start_chs,
        const sdp_MemViewCpu<const int, 1>& end_chs,
        double uvw_min[3],
        double uvw_max[3],
        sdp_Error* status
)
{
    const int64_t num_uvw = uvws.shape[0];
    if (*status) return;
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        const int start_ch = start_chs(i), end_ch = end_chs(i);
        if (start_ch >= end_ch)
            continue;
        const double uvw[] = {uvws(i, 0), uvws(i, 1), uvws(i, 2)};
        for (int j = 0; j < 3; ++j)
        {
            const double u0 = freq0_hz * uvw[j] / C_0;
            const double du = dfreq_hz * uvw[j] / C_0;
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
    const sdp_MemLocation loc = sdp_mem_location(out);
    if (sdp_mem_location(in1) != loc)
    {
        SDP_LOG_ERROR("Arrays must be co-located");
        *status = SDP_ERR_MEM_LOCATION;
        return;
    }
    sdp_MemType type_out = sdp_mem_type(out);
    sdp_MemType type1 = sdp_mem_type(in1);
    sdp_MemType type2 = in2 ? sdp_mem_type(in2) : SDP_MEM_COMPLEX_DOUBLE;
    if (loc == SDP_MEM_CPU)
    {
        if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_COMPLEX_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_scale_array<
                    complex<double>, complex<double>, complex<double>
            >(out, in1, in2, exponent, status);
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_COMPLEX_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_scale_array<
                    complex<float>, complex<float>, complex<double>
            >(out, in1, in2, exponent, status);
        }
        else if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_COMPLEX_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_scale_array<
                    complex<double>, complex<float>, complex<double>
            >(out, in1, in2, exponent, status);
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_COMPLEX_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_scale_array<
                    complex<float>, complex<double>, complex<double>
            >(out, in1, in2, exponent, status);
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else if (loc == SDP_MEM_GPU)
    {
        uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
        const uint64_t shape0 = (uint64_t) sdp_mem_shape_dim(out, 0);
        const uint64_t shape1 = (uint64_t) sdp_mem_shape_dim(out, 1);
        num_blocks[0] = (shape0 + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (shape1 + num_threads[1] - 1) / num_threads[1];
        const int use_in2 = in2 ? 1 : 0;
        if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_COMPLEX_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_MemViewGpu<complex<double>, 2> out_;
            sdp_MemViewGpu<const complex<double>, 2> in1_;
            sdp_MemViewGpu<const complex<double>, 2> in2_;
            sdp_mem_check_and_view(out, &out_, status);
            sdp_mem_check_and_view(in1, &in1_, status);
            if (in2) sdp_mem_check_and_view(in2, &in2_, status);
            const char* kernel_name = "sdp_gridder_accum_scale_array<"
                    "complex<double>, complex<double>, complex<double> "
                    ">";
            const void* arg[] = {
                (const void*) &out_,
                (const void*) &in1_,
                (const void*) &in2_,
                (const void*) &exponent,
                (const void*) &use_in2
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, arg, status
            );
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_COMPLEX_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_MemViewGpu<complex<float>, 2> out_;
            sdp_MemViewGpu<const complex<float>, 2> in1_;
            sdp_MemViewGpu<const complex<double>, 2> in2_;
            sdp_mem_check_and_view(out, &out_, status);
            sdp_mem_check_and_view(in1, &in1_, status);
            if (in2) sdp_mem_check_and_view(in2, &in2_, status);
            const char* kernel_name = "sdp_gridder_accum_scale_array<"
                    "complex<float>, complex<float>, complex<double> "
                    ">";
            const void* arg[] = {
                (const void*) &out_,
                (const void*) &in1_,
                (const void*) &in2_,
                (const void*) &exponent,
                (const void*) &use_in2
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, arg, status
            );
        }
        else if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_COMPLEX_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_MemViewGpu<complex<double>, 2> out_;
            sdp_MemViewGpu<const complex<float>, 2> in1_;
            sdp_MemViewGpu<const complex<double>, 2> in2_;
            sdp_mem_check_and_view(out, &out_, status);
            sdp_mem_check_and_view(in1, &in1_, status);
            if (in2) sdp_mem_check_and_view(in2, &in2_, status);
            const char* kernel_name = "sdp_gridder_accum_scale_array<"
                    "complex<double>, complex<float>, complex<double> "
                    ">";
            const void* arg[] = {
                (const void*) &out_,
                (const void*) &in1_,
                (const void*) &in2_,
                (const void*) &exponent,
                (const void*) &use_in2
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, arg, status
            );
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_COMPLEX_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_MemViewGpu<complex<float>, 2> out_;
            sdp_MemViewGpu<const complex<double>, 2> in1_;
            sdp_MemViewGpu<const complex<double>, 2> in2_;
            sdp_mem_check_and_view(out, &out_, status);
            sdp_mem_check_and_view(in1, &in1_, status);
            if (in2) sdp_mem_check_and_view(in2, &in2_, status);
            const char* kernel_name = "sdp_gridder_accum_scale_array<"
                    "complex<float>, complex<double>, complex<double> "
                    ">";
            const void* arg[] = {
                (const void*) &out_,
                (const void*) &in1_,
                (const void*) &in2_,
                (const void*) &exponent,
                (const void*) &use_in2
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, arg, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            return;
        }
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
    const sdp_MemLocation loc = sdp_mem_location(out);
    if (sdp_mem_location(in1) != loc || sdp_mem_location(in2) != loc)
    {
        SDP_LOG_ERROR("Arrays must be co-located");
        *status = SDP_ERR_MEM_LOCATION;
        return;
    }
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_type(out) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in1) == SDP_MEM_DOUBLE &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            scale_inv_array<complex<double>, double, complex<double> >(
                    out, in1, in2, exponent, status
            );
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in1) == SDP_MEM_FLOAT &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            scale_inv_array<complex<float>, float, complex<double> >(
                    out, in1, in2, exponent, status
            );
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in1) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            scale_inv_array<complex<double>, complex<double>, complex<double> >(
                    out, in1, in2, exponent, status
            );
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in1) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            scale_inv_array<complex<float>, complex<float>, complex<double> >(
                    out, in1, in2, exponent, status
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
        const uint64_t shape0 = (uint64_t) sdp_mem_shape_dim(out, 0);
        const uint64_t shape1 = (uint64_t) sdp_mem_shape_dim(out, 1);
        num_blocks[0] = (shape0 + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (shape1 + num_threads[1] - 1) / num_threads[1];
        if (sdp_mem_type(out) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in1) == SDP_MEM_DOUBLE &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_MemViewGpu<complex<double>, 2> out_;
            sdp_MemViewGpu<const double, 2> in1_;
            sdp_MemViewGpu<const complex<double>, 2> in2_;
            sdp_mem_check_and_view(out, &out_, status);
            sdp_mem_check_and_view(in1, &in1_, status);
            sdp_mem_check_and_view(in2, &in2_, status);
            const char* kernel_name = "sdp_gridder_scale_inv_array<"
                    "complex<double>, double, complex<double> "
                    ">";
            const void* arg[] = {
                (const void*) &out_,
                (const void*) &in1_,
                (const void*) &in2_,
                (const void*) &exponent
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, arg, status
            );
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in1) == SDP_MEM_FLOAT &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_MemViewGpu<complex<float>, 2> out_;
            sdp_MemViewGpu<const float, 2> in1_;
            sdp_MemViewGpu<const complex<double>, 2> in2_;
            sdp_mem_check_and_view(out, &out_, status);
            sdp_mem_check_and_view(in1, &in1_, status);
            sdp_mem_check_and_view(in2, &in2_, status);
            const char* kernel_name = "sdp_gridder_scale_inv_array<"
                    "complex<float>, float, complex<double> "
                    ">";
            const void* arg[] = {
                (const void*) &out_,
                (const void*) &in1_,
                (const void*) &in2_,
                (const void*) &exponent
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, arg, status
            );
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in1) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_MemViewGpu<complex<double>, 2> out_;
            sdp_MemViewGpu<const complex<double>, 2> in1_;
            sdp_MemViewGpu<const complex<double>, 2> in2_;
            sdp_mem_check_and_view(out, &out_, status);
            sdp_mem_check_and_view(in1, &in1_, status);
            sdp_mem_check_and_view(in2, &in2_, status);
            const char* kernel_name = "sdp_gridder_scale_inv_array<"
                    "complex<double>, complex<double>, complex<double> "
                    ">";
            const void* arg[] = {
                (const void*) &out_,
                (const void*) &in1_,
                (const void*) &in2_,
                (const void*) &exponent
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, arg, status
            );
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in1) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_MemViewGpu<complex<float>, 2> out_;
            sdp_MemViewGpu<const complex<float>, 2> in1_;
            sdp_MemViewGpu<const complex<double>, 2> in2_;
            sdp_mem_check_and_view(out, &out_, status);
            sdp_mem_check_and_view(in1, &in1_, status);
            sdp_mem_check_and_view(in2, &in2_, status);
            const char* kernel_name = "sdp_gridder_scale_inv_array<"
                    "complex<float>, complex<float>, complex<double> "
                    ">";
            const void* arg[] = {
                (const void*) &out_,
                (const void*) &in1_,
                (const void*) &in2_,
                (const void*) &exponent
            };
            sdp_launch_cuda_kernel(kernel_name,
                    num_blocks, num_threads, 0, 0, arg, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            return;
        }
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

    // Initialise the output values.
    uvw_min[0] = uvw_min[1] = uvw_min[2] = INFINITY;
    uvw_max[0] = uvw_max[1] = uvw_max[2] = -INFINITY;

    // Check memory location.
    const sdp_MemLocation loc = sdp_mem_location(uvws);
    if (loc == SDP_MEM_CPU)
    {
        sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
        sdp_mem_check_and_view(start_chs, &start_chs_, status);
        sdp_mem_check_and_view(end_chs, &end_chs_, status);

        if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE)
        {
            sdp_MemViewCpu<const double, 2> uvws_;
            sdp_mem_check_and_view(uvws, &uvws_, status);
            uvw_bounds_all<double>(
                    uvws_, freq0_hz, dfreq_hz, start_chs_, end_chs_,
                    uvw_min, uvw_max, status
            );
        }
        else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT)
        {
            sdp_MemViewCpu<const float, 2> uvws_;
            sdp_mem_check_and_view(uvws, &uvws_, status);
            uvw_bounds_all<float>(
                    uvws_, freq0_hz, dfreq_hz, start_chs_, end_chs_,
                    uvw_min, uvw_max, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else if (loc == SDP_MEM_GPU)
    {
        const int64_t shape[] = {3};
        sdp_Mem* uvw_min_cpu = sdp_mem_create_wrapper(
                uvw_min, SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, shape, NULL, status
        );
        sdp_Mem* uvw_max_cpu = sdp_mem_create_wrapper(
                uvw_max, SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, shape, NULL, status
        );
        sdp_Mem* uvw_min_gpu = sdp_mem_create_copy(uvw_min_cpu, loc, status);
        sdp_Mem* uvw_max_gpu = sdp_mem_create_copy(uvw_max_cpu, loc, status);

        // Call the kernel.
        uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
        const int64_t num_elements = sdp_mem_shape_dim(uvws, 0);
        num_blocks[0] = (num_elements + num_threads[0] - 1) / num_threads[0];
        sdp_MemViewGpu<const double, 2> uvws_dbl;
        sdp_MemViewGpu<const float, 2> uvws_flt;
        sdp_MemViewGpu<const int, 1> start_chs_, end_chs_;
        sdp_mem_check_and_view(start_chs, &start_chs_, status);
        sdp_mem_check_and_view(end_chs, &end_chs_, status);
        const char* kernel_name = 0;
        int is_dbl = 0;
        if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(uvws, &uvws_dbl, status);
            kernel_name = "sdp_gridder_uvw_bounds_all<double>";
        }
        else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(uvws, &uvws_flt, status);
            kernel_name = "sdp_gridder_uvw_bounds_all<float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
        const void* arg[] = {
            is_dbl ? (const void*)&uvws_dbl : (const void*)&uvws_flt,
            (const void*)&freq0_hz,
            (const void*)&dfreq_hz,
            (const void*)&start_chs_,
            (const void*)&end_chs_,
            sdp_mem_gpu_buffer(uvw_min_gpu, status),
            sdp_mem_gpu_buffer(uvw_max_gpu, status)
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );

        // Copy results back.
        sdp_mem_copy_contents(uvw_min_cpu, uvw_min_gpu, 0, 0, shape[0], status);
        sdp_mem_copy_contents(uvw_max_cpu, uvw_max_gpu, 0, 0, shape[0], status);
        sdp_mem_free(uvw_min_cpu);
        sdp_mem_free(uvw_max_cpu);
        sdp_mem_free(uvw_min_gpu);
        sdp_mem_free(uvw_max_gpu);
    }
}
