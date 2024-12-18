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
    const int shape0 = (int) out_.shape[0];
    const int shape1 = (int) out_.shape[1];
    if (exponent == 0 || !in2)
    {
        #pragma omp parallel for
        for (int i = 0; i < shape0; ++i)
        {
            for (int j = 0; j < shape1; ++j)
            {
                out_(i, j) += in1_(i, j);
            }
        }
    }
    else
    {
        sdp_MemViewCpu<const IN2_TYPE, 2> in2_;
        sdp_mem_check_and_view(in2, &in2_, status);
        if (exponent == 1)
        {
            #pragma omp parallel for
            for (int i = 0; i < shape0; ++i)
            {
                for (int j = 0; j < shape1; ++j)
                {
                    out_(i, j) += (IN2_TYPE) in1_(i, j) * in2_(i, j);
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int i = 0; i < shape0; ++i)
            {
                for (int j = 0; j < shape1; ++j)
                {
                    out_(i, j) += (IN2_TYPE) in1_(i, j) * pow(
                            in2_(i, j), exponent
                    );
                }
            }
        }
    }
}


// Local function to accumulate the real part of a complex array in the output:
// out += real(in)
template<typename OUT_TYPE, typename IN_TYPE>
void accum_complex_real_array(
        sdp_Mem* out,
        const sdp_Mem* in,
        sdp_Error* status
)
{
    sdp_MemViewCpu<OUT_TYPE, 2> out_;
    sdp_MemViewCpu<const IN_TYPE, 2> in_;
    sdp_mem_check_and_view(out, &out_, status);
    sdp_mem_check_and_view(in, &in_, status);
    const int shape0 = (int) out_.shape[0];
    const int shape1 = (int) out_.shape[1];
    #pragma omp parallel for
    for (int i = 0; i < shape0; ++i)
    {
        for (int j = 0; j < shape1; ++j)
        {
            out_(i, j) += in_(i, j).real();
        }
    }
}


// Local function to count non-zero pixels in image.
template<typename FLUX_TYPE>
int64_t count_nonzero_pixels(const sdp_Mem* image, sdp_Error* status)
{
    int64_t num_sources = 0;
    const int image_size = (int) sdp_mem_shape_dim(image, 0);
    sdp_MemViewCpu<const FLUX_TYPE, 2> image_;
    sdp_mem_check_and_view(image, &image_, status);
    if (*status) return 0;
    for (int il = 0; il < image_size; ++il)
    {
        for (int im = 0; im < image_size; ++im)
        {
            if (image_(il, im) != (FLUX_TYPE) 0) num_sources++;
        }
    }
    return num_sources;
}


// Local function for prediction of visibilities via direct FT.
template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
void dft(
        const sdp_Mem* uvw,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* flux,
        const sdp_Mem* lmn,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double theta,
        double w_step,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;

    // Get views to data.
    const int64_t num_uvw = sdp_mem_shape_dim(uvw, 0);
    const int num_chan = (int) sdp_mem_shape_dim(vis, 1);
    const int num_sources = (int) sdp_mem_shape_dim(flux, 0);
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvw_;
    sdp_MemViewCpu<const FLUX_TYPE, 1> flux_;
    sdp_MemViewCpu<const DIR_TYPE, 2> lmn_;
    sdp_MemViewCpu<complex<VIS_TYPE>, 2> vis_;
    if (start_chs && end_chs)
    {
        sdp_mem_check_and_view(start_chs, &start_chs_, status);
        sdp_mem_check_and_view(end_chs, &end_chs_, status);
    }
    sdp_mem_check_and_view(uvw, &uvw_, status);
    sdp_mem_check_and_view(flux, &flux_, status);
    sdp_mem_check_and_view(lmn, &lmn_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    if (*status) return;

    // Scale subgrid offset values.
    double du = 0, dv = 0, dw = 0;
    if (theta > 0)
    {
        du = (double) subgrid_offset_u / theta;
        dv = (double) subgrid_offset_v / theta;
        dw = (double) subgrid_offset_w * w_step;
    }

    // Loop over uvw values.
    #pragma omp parallel for
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        // Skip if there's no visibility to degrid.
        if (start_chs && end_chs && start_chs_(i) >= end_chs_(i)) continue;

        // Loop over channels.
        for (int c = 0; c < num_chan; ++c)
        {
            const double inv_wave = (freq0_hz + dfreq_hz * c) / C_0;

            // Scale and shift uvws.
            const double u = uvw_(i, 0) * inv_wave - du;
            const double v = uvw_(i, 1) * inv_wave - dv;
            const double w = uvw_(i, 2) * inv_wave - dw;

            // Loop over sources.
            complex<VIS_TYPE> local_vis = 0;
            for (int s = 0; s < num_sources; ++s)
            {
                const double phase = -2.0 * M_PI *
                        (lmn_(s, 0) * u + lmn_(s, 1) * v + lmn_(s, 2) * w);
                const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                local_vis += (complex<VIS_TYPE>) flux_(s) * phasor;
            }

            // Store local visibility.
            vis_(i, c) += local_vis;
        }
    }
}


// Local function to generate image by direct FT.
template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
void idft(
        const sdp_Mem* uvw,
        const sdp_Mem* vis,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* lmn,
        const sdp_Mem* image_taper_1d,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double theta,
        double w_step,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* image,
        sdp_Error* status
)
{
    if (*status) return;

    // Get views to data.
    const int64_t num_uvw = sdp_mem_shape_dim(uvw, 0);
    const int64_t image_size = sdp_mem_shape_dim(image, 0);
    const int num_chan = (int) sdp_mem_shape_dim(vis, 1);
    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvw_;
    sdp_MemViewCpu<const complex<VIS_TYPE>, 2> vis_;
    sdp_MemViewCpu<const DIR_TYPE, 2> lmn_;
    sdp_MemViewCpu<FLUX_TYPE, 2> image_;
    sdp_MemViewCpu<const double, 1> taper_;
    if (start_chs && end_chs)
    {
        sdp_mem_check_and_view(start_chs, &start_chs_, status);
        sdp_mem_check_and_view(end_chs, &end_chs_, status);
    }
    sdp_mem_check_and_view(uvw, &uvw_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    sdp_mem_check_and_view(lmn, &lmn_, status);
    sdp_mem_check_and_view(image, &image_, status);
    if (image_taper_1d) sdp_mem_check_and_view(image_taper_1d, &taper_, status);
    if (*status) return;

    // Scale subgrid offset values.
    double du = 0, dv = 0, dw = 0;
    if (theta > 0)
    {
        du = (double) subgrid_offset_u / theta;
        dv = (double) subgrid_offset_v / theta;
        dw = (double) subgrid_offset_w * w_step;
    }

    // Loop over pixels.
    #pragma omp parallel for collapse(2)
    for (int64_t il = 0; il < image_size; ++il)
    {
        for (int64_t im = 0; im < image_size; ++im)
        {
            FLUX_TYPE local_pix = 0;
            const int64_t s = il * image_size + im; // Linearised pixel index.

            // Loop over uvw values.
            for (int64_t i = 0; i < num_uvw; ++i)
            {
                // Skip if there's no visibility to grid.
                if (start_chs && end_chs && start_chs_(i) >= end_chs_(i))
                {
                    continue;
                }

                // Loop over channels.
                for (int c = 0; c < num_chan; ++c)
                {
                    const double inv_wave = (freq0_hz + dfreq_hz * c) / C_0;

                    // Scale and shift uvws.
                    const double u = uvw_(i, 0) * inv_wave - du;
                    const double v = uvw_(i, 1) * inv_wave - dv;
                    const double w = uvw_(i, 2) * inv_wave - dw;

                    const double phase = 2.0 * M_PI *
                            (lmn_(s, 0) * u + lmn_(s, 1) * v + lmn_(s, 2) * w);
                    const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                    local_pix += vis_(i, c) * phasor;
                }
            }

            // Store local pixel value, appropriately tapered.
            // We can't taper afterwards, as the input image may be nonzero.
            double taper_val = image_taper_1d ? taper_(il) * taper_(im) : 1.0;
            image_(il, im) += local_pix * (FLUX_TYPE) taper_val;
        }
    }
}


// Local function to convert image pixels to coordinates and optionally, fluxes.
template<typename IMAGE_TYPE, typename FLUX_TYPE, typename DIR_TYPE>
void image_to_flmn(
        const sdp_Mem* image,
        double theta,
        double shear_u,
        double shear_v,
        const sdp_Mem* image_taper_1d,
        sdp_Mem* flux,
        sdp_Mem* lmn,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<DIR_TYPE, 2> lmn_;
    sdp_MemViewCpu<const double, 1> taper_;
    sdp_mem_check_and_view(lmn, &lmn_, status);
    if (image_taper_1d) sdp_mem_check_and_view(image_taper_1d, &taper_, status);

    // Store pixel data.
    if (*status) return;
    int64_t k = 0;
    const int size_l = (int) sdp_mem_shape_dim(image, 0);
    const int size_m = (int) sdp_mem_shape_dim(image, 1);
    if (flux)
    {
        sdp_MemViewCpu<const IMAGE_TYPE, 2> image_;
        sdp_MemViewCpu<FLUX_TYPE, 1> flux_;
        sdp_mem_check_and_view(image, &image_, status);
        sdp_mem_check_and_view(flux, &flux_, status);
        for (int il = 0; il < size_l; ++il)
        {
            const double l = (il - size_l / 2) * theta / size_l;
            for (int im = 0; im < size_m; ++im)
            {
                const double m = (im - size_m / 2) * theta / size_m;
                const IMAGE_TYPE pix_val = image_(il, im);
                if (pix_val != (IMAGE_TYPE) 0)
                {
                    const double taper_val = (
                        image_taper_1d ? taper_(il) * taper_(im) : 1.0
                    );
                    flux_(k) = std::real(pix_val) * taper_val;
                    lmn_(k, 0) = l;
                    lmn_(k, 1) = m;
                    lmn_(k, 2) = lm_to_n(l, m, shear_u, shear_v);
                    k++;
                }
            }
        }
    }
    else
    {
        for (int il = 0; il < size_l; ++il)
        {
            const double l = (il - size_l / 2) * theta / size_l;
            for (int im = 0; im < size_m; ++im, ++k)
            {
                const double m = (im - size_m / 2) * theta / size_m;
                lmn_(k, 0) = l;
                lmn_(k, 1) = m;
                lmn_(k, 2) = lm_to_n(l, m, shear_u, shear_v);
            }
        }
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


// Local function to compute the residual difference between 2D arrays, (a - b).
template<typename TYPE_A, typename TYPE_B>
void residual(
        const sdp_Mem* a,
        const sdp_Mem* b,
        sdp_Mem* out,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<const TYPE_A, 2> a_;
    sdp_MemViewCpu<const TYPE_B, 2> b_;
    sdp_MemViewCpu<TYPE_A, 2> out_;
    sdp_mem_check_and_view(a, &a_, status);
    sdp_mem_check_and_view(b, &b_, status);
    sdp_mem_check_and_view(out, &out_, status);
    if (*status) return;
    const int num_x = (int) a_.shape[0];
    const int num_y = (int) a_.shape[1];
    #pragma omp parallel for
    for (int i = 0; i < num_x; ++i)
    {
        for (int j = 0; j < num_y; ++j)
        {
            out_(i, j) = a_(i, j) - (TYPE_A) b_(i, j);
        }
    }
}


// Local function to compute the RMS difference between 2D arrays, rms(a - b).
template<typename TYPE_A, typename TYPE_B>
double rms_diff(const sdp_Mem* a, const sdp_Mem* b, sdp_Error* status)
{
    if (*status) return INFINITY;
    sdp_MemViewCpu<const TYPE_A, 2> a_;
    sdp_MemViewCpu<const TYPE_B, 2> b_;
    sdp_mem_check_and_view(a, &a_, status);
    sdp_mem_check_and_view(b, &b_, status);
    if (*status) return INFINITY;
    double sum_sq_diff = 0.0;
    const int num_x = (int) a_.shape[0];
    const int num_y = (int) a_.shape[1];
    const int size = num_x * num_y;
    #pragma omp parallel for reduction(+:sum_sq_diff) collapse(2)
    for (int i = 0; i < num_x; ++i)
    {
        for (int j = 0; j < num_y; ++j)
        {
            sum_sq_diff += std::norm(a_(i, j) - (TYPE_A) b_(i, j));
        }
    }
    return sqrt(sum_sq_diff / size);
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
    const int shape0 = (int) out_.shape[0];
    const int shape1 = (int) out_.shape[1];
    if (exponent == 1)
    {
        #pragma omp parallel for
        for (int i = 0; i < shape0; ++i)
        {
            for (int j = 0; j < shape1; ++j)
            {
                out_(i, j) = (IN2_TYPE) in1_(i, j) / in2_(i, j);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int i = 0; i < shape0; ++i)
        {
            for (int j = 0; j < shape1; ++j)
            {
                out_(i, j) = (IN2_TYPE) in1_(i, j) / pow(in2_(i, j), exponent);
            }
        }
    }
}


template<typename T>
void shift_subgrids(sdp_Mem* subgrids, sdp_Error* status)
{
    if (*status) return;
    sdp_MemViewCpu<T, 3> subgrids_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    if (*status) return;
    const int w_support = subgrids_.shape[0];
    const int subgrid_size = subgrids_.shape[1];
    for (int k = 0; k < w_support - 1; ++k)
    {
        #pragma omp parallel for
        for (int j = 0; j < subgrid_size; ++j)
        {
            for (int i = 0; i < subgrid_size; ++i)
            {
                subgrids_(k, j, i) = subgrids_(k + 1, j, i);
            }
        }
    }
}


// Add the supplied sub-grid to the grid.
template<typename GRID_TYPE, typename SUBGRID_TYPE, typename FACTOR_TYPE>
void subgrid_add(
        sdp_Mem* grid,
        int offset_u,
        int offset_v,
        const sdp_Mem* subgrid,
        FACTOR_TYPE factor,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<GRID_TYPE, 2> grid_;
    sdp_MemViewCpu<const SUBGRID_TYPE, 2> sub_;
    sdp_mem_check_and_view(grid, &grid_, status);
    sdp_mem_check_and_view(subgrid, &sub_, status);
    if (*status) return;
    // This does the equivalent of numpy.roll and a shift in two dimensions.
    // The "while" loops below really are needed.
    const int64_t sub_size_u = sub_.shape[0], sub_size_v = sub_.shape[1];
    const int64_t grid_size_u = grid_.shape[0], grid_size_v = grid_.shape[1];
    #pragma omp parallel for
    for (int64_t i = 0; i < sub_size_u; ++i)
    {
        int64_t i1 = i + grid_size_u / 2 - sub_size_u / 2 - offset_u;
        while (i1 < 0)
        {
            i1 += grid_size_u;
        }
        while (i1 >= grid_size_u)
        {
            i1 -= grid_size_u;
        }
        for (int64_t j = 0; j < sub_size_v; ++j)
        {
            int64_t j1 = j + grid_size_v / 2 - sub_size_v / 2 - offset_v;
            while (j1 < 0)
            {
                j1 += grid_size_v;
            }
            while (j1 >= grid_size_v)
            {
                j1 -= grid_size_v;
            }
            grid_(i1, j1) += sub_(i, j) * factor;
        }
    }
}


// Cut out a sub-grid from the supplied grid.
template<typename GRID_TYPE, typename SUBGRID_TYPE>
void subgrid_cut_out(
        const sdp_Mem* grid,
        int offset_u,
        int offset_v,
        sdp_Mem* subgrid,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<const GRID_TYPE, 2> grid_;
    sdp_MemViewCpu<SUBGRID_TYPE, 2> sub_;
    sdp_mem_check_and_view(grid, &grid_, status);
    sdp_mem_check_and_view(subgrid, &sub_, status);
    if (*status) return;
    // This does the equivalent of numpy.roll and a shift in two dimensions.
    // The "while" loops below really are needed.
    const int64_t sub_size_u = sub_.shape[0], sub_size_v = sub_.shape[1];
    const int64_t grid_size_u = grid_.shape[0], grid_size_v = grid_.shape[1];
    #pragma omp parallel for
    for (int64_t i = 0; i < sub_size_u; ++i)
    {
        int64_t i1 = i + grid_size_u / 2 - sub_size_u / 2 + offset_u;
        while (i1 < 0)
        {
            i1 += grid_size_u;
        }
        while (i1 >= grid_size_u)
        {
            i1 -= grid_size_u;
        }
        for (int64_t j = 0; j < sub_size_v; ++j)
        {
            int64_t j1 = j + grid_size_v / 2 - sub_size_v / 2 + offset_v;
            while (j1 < 0)
            {
                j1 += grid_size_v;
            }
            while (j1 >= grid_size_v)
            {
                j1 -= grid_size_v;
            }
            sub_(i, j) = grid_(i1, j1);
        }
    }
}


// Determine sum of element-wise difference: result = sum(a - b).
void sum_diff(
        const sdp_Mem* a,
        const sdp_Mem* b,
        int64_t* result,
        int64_t start_row,
        int64_t end_row,
        sdp_Error* status
)
{
    sdp_MemViewCpu<const int, 1> a_;
    sdp_MemViewCpu<const int, 1> b_;
    sdp_mem_check_and_view(a, &a_, status);
    sdp_mem_check_and_view(b, &b_, status);
    if (*status) return;
    if (a_.shape[0] != b_.shape[0])
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    int64_t res = 0;
    #pragma omp parallel for reduction(+:res)
    for (int64_t i = start_row; i < end_row; ++i)
    {
        res += (a_(i) - b_(i));
    }
    *result = res;
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
        const sdp_Error* status
)
{
    const int64_t num_uvw = uvws.shape[0];
    if (*status) return;
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        const int start_ch = start_chs(i), end_ch = end_chs(i);
        if (start_ch >= end_ch) continue;
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
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        *status = SDP_ERR_MEM_LOCATION;
        return;
    }
    sdp_MemType type_out = sdp_mem_type(out);
    sdp_MemType type1 = sdp_mem_type(in1);
    sdp_MemType type2 = in2 ? sdp_mem_type(in2) : SDP_MEM_COMPLEX_DOUBLE;
    if (!in2) exponent = 0;
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
        else if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_scale_array<
                    complex<double>, double, complex<double>
            >(out, in1, in2, exponent, status);
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_scale_array<
                    complex<float>, float, complex<double>
            >(out, in1, in2, exponent, status);
        }
        else if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_scale_array<
                    complex<double>, float, complex<double>
            >(out, in1, in2, exponent, status);
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_scale_array<
                    complex<float>, double, complex<double>
            >(out, in1, in2, exponent, status);
        }
        else if (type_out == SDP_MEM_DOUBLE && type1 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_complex_real_array<double, complex<double> >(
                    out, in1, status
            );
        }
        else if (type_out == SDP_MEM_FLOAT && type1 == SDP_MEM_COMPLEX_FLOAT)
        {
            accum_complex_real_array<float, complex<float> >(out, in1, status);
        }
        else if (type_out == SDP_MEM_DOUBLE && type1 == SDP_MEM_COMPLEX_FLOAT)
        {
            accum_complex_real_array<double, complex<float> >(out, in1, status);
        }
        else if (type_out == SDP_MEM_FLOAT && type1 == SDP_MEM_COMPLEX_DOUBLE)
        {
            accum_complex_real_array<float, complex<double> >(out, in1, status);
        }
        else
        {
            SDP_LOG_ERROR("Unsupported image data type");
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
        sdp_MemViewGpu<complex<double>, 2> out_cd;
        sdp_MemViewGpu<complex<float>, 2> out_cf;
        sdp_MemViewGpu<double, 2> out_d;
        sdp_MemViewGpu<float, 2> out_f;
        sdp_MemViewGpu<const complex<double>, 2> in1_cd;
        sdp_MemViewGpu<const complex<float>, 2> in1_cf;
        sdp_MemViewGpu<const double, 2> in1_d;
        sdp_MemViewGpu<const float, 2> in1_f;
        sdp_MemViewGpu<const complex<double>, 2> in2_;
        if (in2) sdp_mem_check_and_view(in2, &in2_, status);
        const char* kernel_name = 0;
        const void* arg[] = {NULL, NULL, (const void*) &in2_, &exponent};
        if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_COMPLEX_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cd, status);
            sdp_mem_check_and_view(in1, &in1_cd, status);
            kernel_name = "sdp_gridder_accum_scale_array"
                    "<complex<double>, complex<double>, complex<double> >";
            arg[0] = (const void*) &out_cd;
            arg[1] = (const void*) &in1_cd;
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_COMPLEX_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cf, status);
            sdp_mem_check_and_view(in1, &in1_cf, status);
            kernel_name = "sdp_gridder_accum_scale_array"
                    "<complex<float>, complex<float>, complex<double> >";
            arg[0] = (const void*) &out_cf;
            arg[1] = (const void*) &in1_cf;
        }
        else if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_COMPLEX_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cd, status);
            sdp_mem_check_and_view(in1, &in1_cf, status);
            kernel_name = "sdp_gridder_accum_scale_array"
                    "<complex<double>, complex<float>, complex<double> >";
            arg[0] = (const void*) &out_cd;
            arg[1] = (const void*) &in1_cf;
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_COMPLEX_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cf, status);
            sdp_mem_check_and_view(in1, &in1_cd, status);
            kernel_name = "sdp_gridder_accum_scale_array"
                    "<complex<float>, complex<double>, complex<double> >";
            arg[0] = (const void*) &out_cf;
            arg[1] = (const void*) &in1_cd;
        }
        else if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cd, status);
            sdp_mem_check_and_view(in1, &in1_d, status);
            kernel_name = "sdp_gridder_accum_scale_array"
                    "<complex<double>, double, complex<double> >";
            arg[0] = (const void*) &out_cd;
            arg[1] = (const void*) &in1_d;
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cf, status);
            sdp_mem_check_and_view(in1, &in1_f, status);
            kernel_name = "sdp_gridder_accum_scale_array"
                    "<complex<float>, float, complex<double> >";
            arg[0] = (const void*) &out_cf;
            arg[1] = (const void*) &in1_f;
        }
        else if (type_out == SDP_MEM_COMPLEX_DOUBLE &&
                type1 == SDP_MEM_FLOAT &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cd, status);
            sdp_mem_check_and_view(in1, &in1_f, status);
            kernel_name = "sdp_gridder_accum_scale_array"
                    "<complex<double>, float, complex<double> >";
            arg[0] = (const void*) &out_cd;
            arg[1] = (const void*) &in1_f;
        }
        else if (type_out == SDP_MEM_COMPLEX_FLOAT &&
                type1 == SDP_MEM_DOUBLE &&
                type2 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cf, status);
            sdp_mem_check_and_view(in1, &in1_d, status);
            kernel_name = "sdp_gridder_accum_scale_array"
                    "<complex<float>, double, complex<double> >";
            arg[0] = (const void*) &out_cf;
            arg[1] = (const void*) &in1_d;
        }
        else if (type_out == SDP_MEM_DOUBLE && type1 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_d, status);
            sdp_mem_check_and_view(in1, &in1_cd, status);
            kernel_name = "sdp_gridder_accum_complex_real_array"
                    "<double, complex<double> >";
            arg[0] = (const void*) &out_d;
            arg[1] = (const void*) &in1_cd;
        }
        else if (type_out == SDP_MEM_FLOAT && type1 == SDP_MEM_COMPLEX_FLOAT)
        {
            sdp_mem_check_and_view(out, &out_f, status);
            sdp_mem_check_and_view(in1, &in1_cf, status);
            kernel_name = "sdp_gridder_accum_complex_real_array"
                    "<float, complex<float> >";
            arg[0] = (const void*) &out_f;
            arg[1] = (const void*) &in1_cf;
        }
        else if (type_out == SDP_MEM_DOUBLE && type1 == SDP_MEM_COMPLEX_FLOAT)
        {
            sdp_mem_check_and_view(out, &out_d, status);
            sdp_mem_check_and_view(in1, &in1_cf, status);
            kernel_name = "sdp_gridder_accum_complex_real_array"
                    "<double, complex<float> >";
            arg[0] = (const void*) &out_d;
            arg[1] = (const void*) &in1_cf;
        }
        else if (type_out == SDP_MEM_FLOAT && type1 == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_f, status);
            sdp_mem_check_and_view(in1, &in1_cd, status);
            kernel_name = "sdp_gridder_accum_complex_real_array"
                    "<float, complex<double> >";
            arg[0] = (const void*) &out_f;
            arg[1] = (const void*) &in1_cd;
        }
        else
        {
            SDP_LOG_ERROR("Unsupported image data type");
            *status = SDP_ERR_DATA_TYPE;
        }
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}


int64_t sdp_gridder_count_nonzero_pixels(
        const sdp_Mem* image,
        sdp_Error* status
)
{
    int64_t num_src = 0;
    if (*status) return 0;
    switch (sdp_mem_type(image))
    {
    case SDP_MEM_COMPLEX_DOUBLE:
        num_src = count_nonzero_pixels<complex<double> >(image, status);
        break;
    case SDP_MEM_COMPLEX_FLOAT:
        num_src = count_nonzero_pixels<complex<float> >(image, status);
        break;
    case SDP_MEM_DOUBLE:
        num_src = count_nonzero_pixels<double>(image, status);
        break;
    case SDP_MEM_FLOAT:
        num_src = count_nonzero_pixels<float>(image, status);
        break;
    default:
        *status = SDP_ERR_DATA_TYPE;
        break;
    }
    return num_src;
}


double sdp_gridder_determine_w_step(
        double theta,
        double fov,
        double shear_u,
        double shear_v,
        double x0
)
{
    if (x0 == 0.0) x0 = fov / theta;

    // Determine maximum used n, deduce theta along n-axis.
    const double v1 = lm_to_n(-fov / 2.0, -fov / 2.0, shear_u, shear_v);
    const double v2 = lm_to_n(fov / 2.0, -fov / 2.0, shear_u, shear_v);
    const double v3 = lm_to_n(-fov / 2.0, fov / 2.0, shear_u, shear_v);
    const double v4 = lm_to_n(fov / 2.0, fov / 2.0, shear_u, shear_v);
    const double t1 = std::min(v1, v2);
    const double t2 = std::min(v3, v4);
    const double fov_n = 2.0 * -std::min(t1, t2);
    const double theta_n = fov_n / x0;

    // theta_n is our size in image space,
    // therefore 1 / theta_n is our step length in grid space
    return 1.0 / theta_n;
}


void sdp_gridder_dft(
        const sdp_Mem* uvws,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* flux,
        const sdp_Mem* lmn,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double theta,
        double w_step,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(vis);
    const sdp_MemType flux_type = sdp_mem_type(lmn);
    const sdp_MemType dir_type = sdp_mem_type(lmn);
    const sdp_MemType uvw_type = sdp_mem_type(uvws);
    const sdp_MemType vis_type = sdp_mem_type(vis);
    if (loc == SDP_MEM_CPU)
    {
        if (dir_type == SDP_MEM_DOUBLE &&
                flux_type == SDP_MEM_DOUBLE &&
                uvw_type == SDP_MEM_DOUBLE &&
                vis_type == SDP_MEM_COMPLEX_DOUBLE)
        {
            dft<double, double, double, double>(
                    uvws, start_chs, end_chs, flux, lmn,
                    subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                    theta, w_step, freq0_hz, dfreq_hz, vis, status
            );
        }
        else if (dir_type == SDP_MEM_FLOAT &&
                flux_type == SDP_MEM_DOUBLE &&
                uvw_type == SDP_MEM_FLOAT &&
                vis_type == SDP_MEM_COMPLEX_FLOAT)
        {
            dft<float, double, float, float>(
                    uvws, start_chs, end_chs, flux, lmn,
                    subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                    theta, w_step, freq0_hz, dfreq_hz, vis, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data types");
        }
    }
    else
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported memory location");
    }
}


void sdp_gridder_idft(
        const sdp_Mem* uvws,
        const sdp_Mem* vis,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* lmn,
        const sdp_Mem* image_taper_1d,
        int subgrid_offset_u,
        int subgrid_offset_v,
        int subgrid_offset_w,
        double theta,
        double w_step,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* image,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(vis);
    const sdp_MemType img_type = sdp_mem_type(image);
    const sdp_MemType dir_type = sdp_mem_type(lmn);
    const sdp_MemType uvw_type = sdp_mem_type(uvws);
    const sdp_MemType vis_type = sdp_mem_type(vis);
    if (loc == SDP_MEM_CPU)
    {
        if (dir_type == SDP_MEM_DOUBLE &&
                img_type == SDP_MEM_COMPLEX_DOUBLE &&
                uvw_type == SDP_MEM_DOUBLE &&
                vis_type == SDP_MEM_COMPLEX_DOUBLE)
        {
            idft<double, complex<double>, double, double>(
                    uvws, vis, start_chs, end_chs, lmn, image_taper_1d,
                    subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                    theta, w_step, freq0_hz, dfreq_hz, image, status
            );
        }
        else if (dir_type == SDP_MEM_FLOAT &&
                img_type == SDP_MEM_COMPLEX_FLOAT &&
                uvw_type == SDP_MEM_FLOAT &&
                vis_type == SDP_MEM_COMPLEX_FLOAT)
        {
            idft<float, complex<float>, float, float>(
                    uvws, vis, start_chs, end_chs, lmn, image_taper_1d,
                    subgrid_offset_u, subgrid_offset_v, subgrid_offset_w,
                    theta, w_step, freq0_hz, dfreq_hz, image, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data types");
        }
    }
    else
    {
        uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
        const uint64_t shape0 = (uint64_t) sdp_mem_shape_dim(image, 0);
        const uint64_t shape1 = (uint64_t) sdp_mem_shape_dim(image, 1);
        num_blocks[0] = (shape0 + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (shape1 + num_threads[1] - 1) / num_threads[1];
        sdp_MemViewGpu<complex<double>, 2> img_dbl;
        sdp_MemViewGpu<const complex<double>, 2> vis_dbl;
        sdp_MemViewGpu<complex<float>, 2> img_flt;
        sdp_MemViewGpu<const complex<float>, 2> vis_flt;
        sdp_MemViewGpu<const double, 2> lmn_dbl, uvw_dbl;
        sdp_MemViewGpu<const float, 2> lmn_flt, uvw_flt;
        sdp_MemViewGpu<const int, 1> start_ch, end_ch;
        sdp_MemViewGpu<const double, 1> taper;
        int is_dbl = 0, use_start_end_ch = 0, use_taper = 0;
        const char* kernel_name = 0;
        if (dir_type == SDP_MEM_DOUBLE &&
                img_type == SDP_MEM_COMPLEX_DOUBLE &&
                uvw_type == SDP_MEM_DOUBLE &&
                vis_type == SDP_MEM_COMPLEX_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(image, &img_dbl, status);
            sdp_mem_check_and_view(lmn, &lmn_dbl, status);
            sdp_mem_check_and_view(uvws, &uvw_dbl, status);
            sdp_mem_check_and_view(vis, &vis_dbl, status);
            kernel_name = "sdp_gridder_idft"
                    "<double, complex<double>, double, double>";
        }
        else if (dir_type == SDP_MEM_FLOAT &&
                img_type == SDP_MEM_COMPLEX_FLOAT &&
                uvw_type == SDP_MEM_FLOAT &&
                vis_type == SDP_MEM_COMPLEX_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(image, &img_flt, status);
            sdp_mem_check_and_view(lmn, &lmn_flt, status);
            sdp_mem_check_and_view(uvws, &uvw_flt, status);
            sdp_mem_check_and_view(vis, &vis_flt, status);
            kernel_name = "sdp_gridder_idft"
                    "<float, complex<float>, float, float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data types");
            return;
        }
        if (start_chs && end_chs)
        {
            use_start_end_ch = 1;
            sdp_mem_check_and_view(start_chs, &start_ch, status);
            sdp_mem_check_and_view(end_chs, &end_ch, status);
        }
        if (image_taper_1d)
        {
            use_taper = 1;
            sdp_mem_check_and_view(image_taper_1d, &taper, status);
        }
        const void* arg[] = {
            is_dbl ? (const void*) &uvw_dbl : (const void*) &uvw_flt,
            is_dbl ? (const void*) &vis_dbl : (const void*) &vis_flt,
            (const void*) &start_ch,
            (const void*) &end_ch,
            is_dbl ? (const void*) &lmn_dbl : (const void*) &lmn_flt,
            (const void*) &taper,
            (const void*) &subgrid_offset_u,
            (const void*) &subgrid_offset_v,
            (const void*) &subgrid_offset_w,
            (const void*) &theta,
            (const void*) &w_step,
            (const void*) &freq0_hz,
            (const void*) &dfreq_hz,
            is_dbl ? (const void*) &img_dbl : (const void*) &img_flt,
            (const void*) &use_start_end_ch,
            (const void*) &use_taper
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}


void sdp_gridder_image_to_flmn(
        const sdp_Mem* image,
        double theta,
        double shear_u,
        double shear_v,
        const sdp_Mem* image_taper_1d,
        sdp_Mem* flux,
        sdp_Mem* lmn,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemType image_type = sdp_mem_type(image);
    const sdp_MemType flux_type = flux ? sdp_mem_type(flux) : SDP_MEM_DOUBLE;
    const sdp_MemType dir_type = sdp_mem_type(lmn);
    if (image_type == SDP_MEM_DOUBLE &&
            flux_type == SDP_MEM_DOUBLE &&
            dir_type == SDP_MEM_DOUBLE)
    {
        image_to_flmn<double, double, double>(image, theta,
                shear_u, shear_v, image_taper_1d, flux, lmn, status
        );
    }
    else if (image_type == SDP_MEM_FLOAT &&
            flux_type == SDP_MEM_DOUBLE &&
            dir_type == SDP_MEM_FLOAT)
    {
        image_to_flmn<float, double, float>(image, theta,
                shear_u, shear_v, image_taper_1d, flux, lmn, status
        );
    }
    else if (image_type == SDP_MEM_COMPLEX_DOUBLE &&
            flux_type == SDP_MEM_DOUBLE &&
            dir_type == SDP_MEM_DOUBLE)
    {
        image_to_flmn<complex<double>, double, double>(image, theta,
                shear_u, shear_v, image_taper_1d, flux, lmn, status
        );
    }
    else if (image_type == SDP_MEM_COMPLEX_FLOAT &&
            flux_type == SDP_MEM_DOUBLE &&
            dir_type == SDP_MEM_FLOAT)
    {
        image_to_flmn<complex<float>, double, float>(image, theta,
                shear_u, shear_v, image_taper_1d, flux, lmn, status
        );
    }
    else if (image_type == SDP_MEM_COMPLEX_FLOAT &&
            flux_type == SDP_MEM_DOUBLE &&
            dir_type == SDP_MEM_DOUBLE)
    {
        image_to_flmn<complex<float>, double, double>(image, theta,
                shear_u, shear_v, image_taper_1d, flux, lmn, status
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types");
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


void sdp_gridder_residual(
        const sdp_Mem* a,
        const sdp_Mem* b,
        sdp_Mem* out,
        sdp_Error* status
)
{
    const sdp_Mem* p_a = a;
    const sdp_Mem* p_b = b;
    sdp_Mem* temp_a = 0;
    sdp_Mem* temp_b = 0;
    if (*status) return;
    const sdp_MemType t_a = sdp_mem_type(a);
    const sdp_MemType t_b = sdp_mem_type(b);
    if (sdp_mem_num_dims(a) != 2 || sdp_mem_num_dims(b) != 2 ||
            sdp_mem_num_dims(out) != 2)
    {
        SDP_LOG_ERROR("All arrays must be 2D");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    if (sdp_mem_shape_dim(a, 0) != sdp_mem_shape_dim(b, 0) ||
            sdp_mem_shape_dim(a, 1) != sdp_mem_shape_dim(b, 1) ||
            sdp_mem_shape_dim(a, 0) != sdp_mem_shape_dim(out, 0) ||
            sdp_mem_shape_dim(a, 1) != sdp_mem_shape_dim(out, 1))
    {
        SDP_LOG_ERROR("All arrays must have the same shape");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    if (t_a != sdp_mem_type(out))
    {
        SDP_LOG_ERROR("Arrays 'a' and 'out' must be of the same type");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }
    if (sdp_mem_location(out) != SDP_MEM_CPU)
    {
        SDP_LOG_ERROR("Output residual must be in CPU memory");
        *status = SDP_ERR_MEM_LOCATION;
        return;
    }
    if (sdp_mem_location(a) != SDP_MEM_CPU)
    {
        temp_a = sdp_mem_create_copy(a, SDP_MEM_CPU, status);
        p_a = temp_a;
    }
    if (sdp_mem_location(b) != SDP_MEM_CPU)
    {
        temp_b = sdp_mem_create_copy(b, SDP_MEM_CPU, status);
        p_b = temp_b;
    }
    if (t_a == SDP_MEM_COMPLEX_DOUBLE && t_b == SDP_MEM_COMPLEX_DOUBLE)
    {
        residual<complex<double>, complex<double> >(p_a, p_b, out, status);
    }
    else if (t_a == SDP_MEM_COMPLEX_FLOAT && t_b == SDP_MEM_COMPLEX_FLOAT)
    {
        residual<complex<float>, complex<float> >(p_a, p_b, out, status);
    }
    else if (t_a == SDP_MEM_COMPLEX_DOUBLE && t_b == SDP_MEM_COMPLEX_FLOAT)
    {
        residual<complex<double>, complex<float> >(p_a, p_b, out, status);
    }
    else if (t_a == SDP_MEM_DOUBLE && t_b == SDP_MEM_DOUBLE)
    {
        residual<double, double>(p_a, p_b, out, status);
    }
    else if (t_a == SDP_MEM_FLOAT && t_b == SDP_MEM_FLOAT)
    {
        residual<float, float>(p_a, p_b, out, status);
    }
    else if (t_a == SDP_MEM_DOUBLE && t_b == SDP_MEM_FLOAT)
    {
        residual<double, float>(p_a, p_b, out, status);
    }
    else
    {
        SDP_LOG_ERROR("Unsupported data types for residual calculation");
        *status = SDP_ERR_DATA_TYPE;
    }
    sdp_mem_free(temp_a);
    sdp_mem_free(temp_b);
}


double sdp_gridder_rms_diff(
        const sdp_Mem* a,
        const sdp_Mem* b,
        sdp_Error* status
)
{
    const sdp_Mem* p_a = a;
    const sdp_Mem* p_b = b;
    sdp_Mem* temp_a = 0;
    sdp_Mem* temp_b = 0;
    double out = INFINITY;
    if (*status) return INFINITY;
    const sdp_MemType t_a = sdp_mem_type(a);
    const sdp_MemType t_b = sdp_mem_type(b);
    if (sdp_mem_num_dims(a) != 2 || sdp_mem_num_dims(b) != 2)
    {
        SDP_LOG_ERROR("Input arrays must be 2D");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return INFINITY;
    }
    if (sdp_mem_shape_dim(a, 0) != sdp_mem_shape_dim(b, 0) ||
            sdp_mem_shape_dim(a, 1) != sdp_mem_shape_dim(b, 1))
    {
        SDP_LOG_ERROR("Input arrays must have the same shape");
        *status = SDP_ERR_INVALID_ARGUMENT;
        return INFINITY;
    }
    if (sdp_mem_location(a) != SDP_MEM_CPU)
    {
        temp_a = sdp_mem_create_copy(a, SDP_MEM_CPU, status);
        p_a = temp_a;
    }
    if (sdp_mem_location(b) != SDP_MEM_CPU)
    {
        temp_b = sdp_mem_create_copy(b, SDP_MEM_CPU, status);
        p_b = temp_b;
    }
    if (t_a == SDP_MEM_COMPLEX_DOUBLE && t_b == SDP_MEM_COMPLEX_DOUBLE)
    {
        out = rms_diff<complex<double>, complex<double> >(p_a, p_b, status);
    }
    else if (t_a == SDP_MEM_COMPLEX_FLOAT && t_b == SDP_MEM_COMPLEX_FLOAT)
    {
        out = rms_diff<complex<float>, complex<float> >(p_a, p_b, status);
    }
    else if (t_a == SDP_MEM_COMPLEX_DOUBLE && t_b == SDP_MEM_COMPLEX_FLOAT)
    {
        out = rms_diff<complex<double>, complex<float> >(p_a, p_b, status);
    }
    else if (t_a == SDP_MEM_DOUBLE && t_b == SDP_MEM_DOUBLE)
    {
        out = rms_diff<double, double>(p_a, p_b, status);
    }
    else if (t_a == SDP_MEM_FLOAT && t_b == SDP_MEM_FLOAT)
    {
        out = rms_diff<float, float>(p_a, p_b, status);
    }
    else if (t_a == SDP_MEM_DOUBLE && t_b == SDP_MEM_FLOAT)
    {
        out = rms_diff<double, float>(p_a, p_b, status);
    }
    else
    {
        SDP_LOG_ERROR("Unsupported data types for RMS difference calculation");
        *status = SDP_ERR_DATA_TYPE;
    }
    sdp_mem_free(temp_a);
    sdp_mem_free(temp_b);
    return out;
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
        SDP_LOG_ERROR("All arrays must be in the same memory space");
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
            SDP_LOG_ERROR("Unsupported image data type");
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
        sdp_MemViewGpu<complex<double>, 2> out_cd;
        sdp_MemViewGpu<complex<float>, 2> out_cf;
        sdp_MemViewGpu<const complex<double>, 2> in1_cd;
        sdp_MemViewGpu<const complex<float>, 2> in1_cf;
        sdp_MemViewGpu<const double, 2> in1_d;
        sdp_MemViewGpu<const float, 2> in1_f;
        sdp_MemViewGpu<const complex<double>, 2> in2_;
        sdp_mem_check_and_view(in2, &in2_, status);
        const char* kernel_name = 0;
        const void* arg[] = {NULL, NULL, (const void*) &in2_, &exponent};
        if (sdp_mem_type(out) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in1) == SDP_MEM_DOUBLE &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cd, status);
            sdp_mem_check_and_view(in1, &in1_d, status);
            kernel_name = "sdp_gridder_scale_inv_array"
                    "<complex<double>, double, complex<double> >";
            arg[0] = (const void*) &out_cd;
            arg[1] = (const void*) &in1_d;
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in1) == SDP_MEM_FLOAT &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cf, status);
            sdp_mem_check_and_view(in1, &in1_f, status);
            kernel_name = "sdp_gridder_scale_inv_array"
                    "<complex<float>, float, complex<double> >";
            arg[0] = (const void*) &out_cf;
            arg[1] = (const void*) &in1_f;
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in1) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cd, status);
            sdp_mem_check_and_view(in1, &in1_cd, status);
            kernel_name = "sdp_gridder_scale_inv_array"
                    "<complex<double>, complex<double>, complex<double> >";
            arg[0] = (const void*) &out_cd;
            arg[1] = (const void*) &in1_cd;
        }
        else if (sdp_mem_type(out) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in1) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(in2) == SDP_MEM_COMPLEX_DOUBLE)
        {
            sdp_mem_check_and_view(out, &out_cf, status);
            sdp_mem_check_and_view(in1, &in1_cf, status);
            kernel_name = "sdp_gridder_scale_inv_array"
                    "<complex<float>, complex<float>, complex<double> >";
            arg[0] = (const void*) &out_cf;
            arg[1] = (const void*) &in1_cf;
        }
        else
        {
            SDP_LOG_ERROR("Unsupported image data type");
            *status = SDP_ERR_DATA_TYPE;
        }
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}


void sdp_gridder_shift_subgrids(sdp_Mem* subgrids, sdp_Error* status)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(subgrids);
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE)
        {
            shift_subgrids<complex<double> >(subgrids, status);
        }
        else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT)
        {
            shift_subgrids<complex<float> >(subgrids, status);
        }
        else
        {
            SDP_LOG_ERROR("Unsupported sub-grid data type");
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else
    {
        // Call the kernel.
        uint64_t num_threads[] = {32, 8, 1}, num_blocks[] = {1, 1, 1};
        const int64_t sub_size_u = sdp_mem_shape_dim(subgrids, 1);
        const int64_t sub_size_v = sdp_mem_shape_dim(subgrids, 2);
        num_blocks[0] = (sub_size_u + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (sub_size_v + num_threads[1] - 1) / num_threads[1];
        sdp_MemViewGpu<complex<double>, 3> sub_dbl;
        sdp_MemViewGpu<complex<float>, 3> sub_flt;
        const char* kernel_name = 0;
        int is_dbl = 0;
        if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(subgrids, &sub_dbl, status);
            kernel_name = "sdp_gridder_shift_subgrids<complex<double> >";
        }
        else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(subgrids, &sub_flt, status);
            kernel_name = "sdp_gridder_shift_subgrids<complex<float> >";
        }
        else
        {
            SDP_LOG_ERROR("Unsupported sub-grid data type");
            *status = SDP_ERR_DATA_TYPE;
        }
        const void* arg[] = {
            is_dbl ? (const void*)&sub_dbl : (const void*)&sub_flt
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}


void sdp_gridder_subgrid_add(
        sdp_Mem* grid,
        int offset_u,
        int offset_v,
        const sdp_Mem* subgrid,
        double factor,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(grid);
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_type(grid) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(subgrid) == SDP_MEM_COMPLEX_DOUBLE)
        {
            subgrid_add<complex<double>, complex<double>, double>(
                    grid, offset_u, offset_v, subgrid, factor, status
            );
        }
        else if (sdp_mem_type(grid) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(subgrid) == SDP_MEM_COMPLEX_FLOAT)
        {
            subgrid_add<complex<float>, complex<float>, float>(
                    grid, offset_u, offset_v, subgrid, (float) factor, status
            );
        }
        else if (sdp_mem_type(grid) == SDP_MEM_DOUBLE &&
                sdp_mem_type(subgrid) == SDP_MEM_DOUBLE)
        {
            subgrid_add<double, double, double>(
                    grid, offset_u, offset_v, subgrid, factor, status
            );
        }
        else if (sdp_mem_type(grid) == SDP_MEM_FLOAT &&
                sdp_mem_type(subgrid) == SDP_MEM_FLOAT)
        {
            subgrid_add<float, float, float>(
                    grid, offset_u, offset_v, subgrid, (float) factor, status
            );
        }
        else
        {
            SDP_LOG_ERROR("Unsupported grid or sub-grid data type");
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else
    {
        // Call the kernel.
        uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
        const int64_t sub_size_u = sdp_mem_shape_dim(subgrid, 0);
        const int64_t sub_size_v = sdp_mem_shape_dim(subgrid, 1);
        num_blocks[0] = (sub_size_u + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (sub_size_v + num_threads[1] - 1) / num_threads[1];
        sdp_MemViewGpu<const complex<double>, 2> grid_dbl, sub_dbl;
        sdp_MemViewGpu<const complex<float>, 2> grid_flt, sub_flt;
        const char* kernel_name = 0;
        int is_dbl = 0;
        if (sdp_mem_type(grid) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(subgrid) == SDP_MEM_COMPLEX_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(grid, &grid_dbl, status);
            sdp_mem_check_and_view(subgrid, &sub_dbl, status);
            kernel_name = "sdp_gridder_subgrid_add<"
                    "complex<double>, complex<double>, double>";
        }
        else if (sdp_mem_type(grid) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(subgrid) == SDP_MEM_COMPLEX_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(grid, &grid_flt, status);
            sdp_mem_check_and_view(subgrid, &sub_flt, status);
            kernel_name = "sdp_gridder_subgrid_add<"
                    "complex<float>, complex<float>, double>";
        }
        else
        {
            SDP_LOG_ERROR("Unsupported grid or sub-grid data type");
            *status = SDP_ERR_DATA_TYPE;
        }
        const void* arg[] = {
            is_dbl ? (const void*)&grid_dbl : (const void*)&grid_flt,
            (const void*)&offset_u,
            (const void*)&offset_v,
            is_dbl ? (const void*)&sub_dbl : (const void*)&sub_flt,
            (const void*)&factor
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}


void sdp_gridder_subgrid_cut_out(
        const sdp_Mem* grid,
        int offset_u,
        int offset_v,
        sdp_Mem* subgrid,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(grid);
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_type(grid) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(subgrid) == SDP_MEM_COMPLEX_DOUBLE)
        {
            subgrid_cut_out<complex<double>, complex<double> >(
                    grid, offset_u, offset_v, subgrid, status
            );
        }
        else if (sdp_mem_type(grid) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(subgrid) == SDP_MEM_COMPLEX_FLOAT)
        {
            subgrid_cut_out<complex<float>, complex<float> >(
                    grid, offset_u, offset_v, subgrid, status
            );
        }
        else if (sdp_mem_type(grid) == SDP_MEM_DOUBLE &&
                sdp_mem_type(subgrid) == SDP_MEM_DOUBLE)
        {
            subgrid_cut_out<double, double>(
                    grid, offset_u, offset_v, subgrid, status
            );
        }
        else if (sdp_mem_type(grid) == SDP_MEM_FLOAT &&
                sdp_mem_type(subgrid) == SDP_MEM_FLOAT)
        {
            subgrid_cut_out<float, float>(
                    grid, offset_u, offset_v, subgrid, status
            );
        }
        else
        {
            SDP_LOG_ERROR("Unsupported grid or sub-grid data type");
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else
    {
        // Call the kernel.
        uint64_t num_threads[] = {16, 16, 1}, num_blocks[] = {1, 1, 1};
        const int64_t sub_size_u = sdp_mem_shape_dim(subgrid, 0);
        const int64_t sub_size_v = sdp_mem_shape_dim(subgrid, 1);
        num_blocks[0] = (sub_size_u + num_threads[0] - 1) / num_threads[0];
        num_blocks[1] = (sub_size_v + num_threads[1] - 1) / num_threads[1];
        sdp_MemViewGpu<const complex<double>, 2> grid_dbl, sub_dbl;
        sdp_MemViewGpu<const complex<float>, 2> grid_flt, sub_flt;
        const char* kernel_name = 0;
        int is_dbl = 0;
        if (sdp_mem_type(grid) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(subgrid) == SDP_MEM_COMPLEX_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(grid, &grid_dbl, status);
            sdp_mem_check_and_view(subgrid, &sub_dbl, status);
            kernel_name = "sdp_gridder_subgrid_cut_out<"
                    "complex<double>, complex<double> >";
        }
        else if (sdp_mem_type(grid) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(subgrid) == SDP_MEM_COMPLEX_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(grid, &grid_flt, status);
            sdp_mem_check_and_view(subgrid, &sub_flt, status);
            kernel_name = "sdp_gridder_subgrid_cut_out<"
                    "complex<float>, complex<float> >";
        }
        else
        {
            SDP_LOG_ERROR("Unsupported grid or sub-grid data type");
            *status = SDP_ERR_DATA_TYPE;
        }
        const void* arg[] = {
            is_dbl ? (const void*)&grid_dbl : (const void*)&grid_flt,
            (const void*)&offset_u,
            (const void*)&offset_v,
            is_dbl ? (const void*)&sub_dbl : (const void*)&sub_flt
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}


void sdp_gridder_sum_diff(
        const sdp_Mem* a,
        const sdp_Mem* b,
        int64_t* result,
        int64_t start_row,
        int64_t end_row,
        sdp_Error* status
)
{
    if (*status) return;
    if (start_row < 0 || end_row < 0)
    {
        start_row = 0;
        end_row = sdp_mem_shape_dim(a, 0);
    }
    const sdp_MemLocation loc = sdp_mem_location(a);
    *result = 0;
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_type(a) == SDP_MEM_INT && sdp_mem_type(b) == SDP_MEM_INT)
        {
            sum_diff(a, b, result, start_row, end_row, status);
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
    }
    else
    {
        // Allocate space for the result.
        const int64_t shape[] = {1};
        sdp_Mem* result_gpu = sdp_mem_create(
                SDP_MEM_INT, SDP_MEM_GPU, 1, shape, status
        );
        sdp_mem_clear_contents(result_gpu, status);

        // Call the kernel.
        uint64_t num_threads[] = {512, 1, 1}, num_blocks[] = {8, 1, 1};
        sdp_MemViewGpu<const int, 1> a_int, b_int;
        sdp_mem_check_and_view(a, &a_int, status);
        sdp_mem_check_and_view(b, &b_int, status);
        const char* kernel_name = 0;
        if (sdp_mem_type(a) == SDP_MEM_INT && sdp_mem_type(b) == SDP_MEM_INT)
        {
            kernel_name = "sdp_gridder_sum_diff<int, 512>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
        const void* arg[] = {
            (const void*)&a_int,
            (const void*)&b_int,
            (const void*)&start_row,
            (const void*)&end_row,
            sdp_mem_gpu_buffer(result_gpu, status)
        };
        uint64_t shared_mem_bytes = num_threads[0] * sizeof(int);
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, shared_mem_bytes, 0, arg, status
        );

        sdp_Mem* result_cpu = sdp_mem_create_copy(
                result_gpu, SDP_MEM_CPU, status
        );
        *result = *((int*) sdp_mem_data(result_cpu));
        sdp_mem_free(result_cpu);
        sdp_mem_free(result_gpu);
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
            SDP_LOG_ERROR("Unsupported (u,v,w) data type");
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
            SDP_LOG_ERROR("Unsupported (u,v,w) data type");
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
