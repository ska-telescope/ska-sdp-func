/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_direct.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using std::complex;

#define C_0 299792458.0

struct sdp_GridderDirect
{
    int image_size;
    int subgrid_size;
    double theta;
    int support;
    sdp_Mem* pswf;
    sdp_Mem* pswf_sg;
};

// Begin anonymous namespace for file-local functions.
namespace {

// Local function to count non-zero pixels in image.
template<typename FLUX_TYPE>
int64_t count_nonzero_pixels(const sdp_Mem* image, sdp_Error* status)
{
    int64_t num_sources = 0;
    const int64_t image_size = sdp_mem_shape_dim(image, 0);
    sdp_MemViewCpu<const FLUX_TYPE, 2> image_;
    sdp_mem_check_and_view(image, &image_, status);
    if (*status) return 0;
    for (int64_t il = 0; il < image_size; ++il)
    {
        for (int64_t im = 0; im < image_size; ++im)
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
        sdp_GridderDirect* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* flux,
        const sdp_Mem* lmn,
        int subgrid_offset_u,
        int subgrid_offset_v,
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
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    sdp_mem_check_and_view(uvw, &uvw_, status);
    sdp_mem_check_and_view(flux, &flux_, status);
    sdp_mem_check_and_view(lmn, &lmn_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    if (*status) return;

    // Scale subgrid offset values.
    const double du = (double) subgrid_offset_u / plan->theta;
    const double dv = (double) subgrid_offset_v / plan->theta;

    // Loop over uvw values.
    #pragma omp parallel for
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        // Skip if there's no visibility to degrid.
        if (start_chs_(i) >= end_chs_(i)) continue;

        // Loop over channels.
        for (int c = 0; c < num_chan; ++c)
        {
            const double inv_wave = (freq0_hz + dfreq_hz * c) / C_0;

            // Scale and shift uvws.
            const double u = uvw_(i, 0) * inv_wave - du;
            const double v = uvw_(i, 1) * inv_wave - dv;
            const double w = uvw_(i, 2) * inv_wave;

            // Loop over sources.
            complex<VIS_TYPE> local_vis = 0;
            for (int s = 0; s < num_sources; ++s)
            {
                const double phase = -2.0 * M_PI *
                        (lmn_(s, 0) * u + lmn_(s, 1) * v + lmn_(s, 2) * w);
                const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                local_vis += flux_(s) * phasor;
            }

            // Store local visibility.
            vis_(i, c) = local_vis;
        }
    }
}


// Local function to generate subimage by direct FT.
template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
void idft(
        sdp_GridderDirect* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* vis,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        const sdp_Mem* lmn,
        int subgrid_offset_u,
        int subgrid_offset_v,
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
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    sdp_mem_check_and_view(uvw, &uvw_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    sdp_mem_check_and_view(lmn, &lmn_, status);
    sdp_mem_check_and_view(image, &image_, status);
    const double* pswf = (const double*) sdp_mem_data_const(plan->pswf_sg);
    if (*status) return;

    // Scale subgrid offset values.
    const double du = (double) subgrid_offset_u / plan->theta;
    const double dv = (double) subgrid_offset_v / plan->theta;

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
                if (start_chs_(i) >= end_chs_(i)) continue;

                // Loop over channels.
                for (int c = 0; c < num_chan; ++c)
                {
                    const double inv_wave = (freq0_hz + dfreq_hz * c) / C_0;

                    // Scale and shift uvws.
                    const double u = uvw_(i, 0) * inv_wave - du;
                    const double v = uvw_(i, 1) * inv_wave - dv;
                    const double w = uvw_(i, 2) * inv_wave;

                    const double phase = 2.0 * M_PI *
                            (lmn_(s, 0) * u + lmn_(s, 1) * v + lmn_(s, 2) * w);
                    const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                    local_pix += vis_(i, c) * phasor;
                }
            }

            // Store local pixel value, appropriately tapered by the PSWF.
            // (We can't taper the whole image, as the input may be nonzero.)
            image_(il, im) += local_pix * (FLUX_TYPE) (pswf[il] * pswf[im]);
        }
    }
}


// Local function to convert image pixel indices to direction cosines (l,m,n).
template<typename T>
void idx_to_dir(int il, int im, int size, double theta, T& l, T& m, T& n)
{
    l = (il - size / 2) * theta / size;
    m = (im - size / 2) * theta / size;
    n = sqrt(1.0 - l * l - m * m) - 1.0;
}


// Local function to convert nonzero image pixels to fluxes and coordinates.
template<typename DIR_TYPE, typename FLUX_TYPE>
void image_to_flmn(
        const sdp_Mem* image,
        double theta,
        const sdp_Mem* pswf,
        sdp_Mem* flux,
        sdp_Mem* lmn,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<FLUX_TYPE, 1> flux_;
    sdp_MemViewCpu<const FLUX_TYPE, 2> image_;
    sdp_MemViewCpu<DIR_TYPE, 2> lmn_;
    sdp_MemViewCpu<const double, 1> pswf_;
    sdp_mem_check_and_view(flux, &flux_, status);
    sdp_mem_check_and_view(image, &image_, status);
    sdp_mem_check_and_view(lmn, &lmn_, status);
    sdp_mem_check_and_view(pswf, &pswf_, status);

    // Store pixel data.
    if (*status) return;
    int64_t k = 0;
    const int64_t image_size = sdp_mem_shape_dim(image, 0);
    for (int64_t il = 0; il < image_size; ++il)
    {
        for (int64_t im = 0; im < image_size; ++im)
        {
            const FLUX_TYPE pix_val = image_(il, im);
            if (pix_val != (FLUX_TYPE) 0)
            {
                flux_(k) = pix_val * (FLUX_TYPE) (pswf_(il) * pswf_(im));
                idx_to_dir(il, im, image_size, theta,
                        lmn_(k, 0), lmn_(k, 1), lmn_(k, 2)
                );
                k++;
            }
        }
    }
}


// Local function to convert all image pixel positions to coordinates.
template<typename T>
void image_to_lmn(
        const sdp_Mem* image,
        double theta,
        sdp_Mem* lmn,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_MemViewCpu<T, 2> lmn_;
    sdp_mem_check_and_view(lmn, &lmn_, status);

    // Store pixel data.
    if (*status) return;
    const int64_t image_size = sdp_mem_shape_dim(image, 0);
    for (int64_t il = 0, k = 0; il < image_size; ++il)
    {
        for (int64_t im = 0; im < image_size; ++im, ++k)
        {
            idx_to_dir(il, im, image_size, theta,
                    lmn_(k, 0), lmn_(k, 1), lmn_(k, 2)
            );
        }
    }
}


// Local function to apply grid correction.
template<typename T>
void grid_corr(
        const sdp_GridderDirect* plan,
        sdp_Mem* facet,
        const double* pswf_l,
        const double* pswf_m,
        sdp_Error* status
)
{
    if (*status) return;

    // Apply portion of shifted PSWF to facet.
    sdp_MemViewCpu<T, 2> facet_;
    sdp_mem_check_and_view(facet, &facet_, status);
    const int64_t num_l = sdp_mem_shape_dim(facet, 0);
    const int64_t num_m = sdp_mem_shape_dim(facet, 1);
    for (int64_t il = 0, k = 0; il < num_l; ++il)
    {
        const int64_t pl = il + (plan->image_size / 2 - num_l / 2);
        for (int64_t im = 0; im < num_m; ++im, ++k)
        {
            const int64_t pm = im + (plan->image_size / 2 - num_m / 2);
            facet_(il, im) /= (T) (pswf_l[pl] * pswf_m[pm]);
        }
    }
}

} // End anonymous namespace for file-local functions.


sdp_GridderDirect* sdp_gridder_direct_create(
        int image_size,
        int subgrid_size,
        double theta,
        int support,
        sdp_Error* status
)
{
    sdp_GridderDirect* plan = (sdp_GridderDirect*)calloc(
            1, sizeof(sdp_GridderDirect)
    );
    plan->image_size = image_size;
    plan->subgrid_size = subgrid_size;
    plan->theta = theta;
    plan->support = support;
    const int64_t pswf_shape[] = {image_size};
    const int64_t pswf_sg_shape[] = {subgrid_size};
    plan->pswf = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, pswf_shape, status
    );
    plan->pswf_sg = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, pswf_sg_shape, status
    );
    sdp_generate_pswf(0, support * (M_PI / 2), plan->pswf, status);
    sdp_generate_pswf(0, support * (M_PI / 2), plan->pswf_sg, status);
    ((double*)sdp_mem_data(plan->pswf))[0] = 1e-15;
    ((double*)sdp_mem_data(plan->pswf_sg))[0] = 1e-15;
    return plan;
}


void sdp_gridder_direct_degrid(
        sdp_GridderDirect* plan,
        const sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        double freq0_hz,
        double dfreq_hz,
        const sdp_Mem* uvw,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        sdp_Mem* vis,
        sdp_Error* status
)
{
    if (*status) return;

    // Count nonzero pixels in image.
    int64_t num_src = 0;
    const sdp_MemType flux_type = sdp_mem_type(subgrid_image);
    switch (flux_type)
    {
    case SDP_MEM_COMPLEX_DOUBLE:
        num_src = count_nonzero_pixels<complex<double> >(subgrid_image, status);
        break;
    case SDP_MEM_COMPLEX_FLOAT:
        num_src = count_nonzero_pixels<complex<float> >(subgrid_image, status);
        break;
    case SDP_MEM_DOUBLE:
        num_src = count_nonzero_pixels<double>(subgrid_image, status);
        break;
    case SDP_MEM_FLOAT:
        num_src = count_nonzero_pixels<float>(subgrid_image, status);
        break;
    default:
        *status = SDP_ERR_DATA_TYPE;
        return;
    }
    const sdp_MemType dir_type = sdp_mem_type(uvw);
    const sdp_MemLocation loc = sdp_mem_location(vis);
    const int64_t lmn_shape[] = {num_src, 3};
    sdp_Mem* flux = sdp_mem_create(flux_type, loc, 1, &num_src, status);
    sdp_Mem* lmn = sdp_mem_create(dir_type, loc, 2, lmn_shape, status);

    // Convert image into positions and flux values for all nonzero pixels,
    // and use DFT for degridding.
    if (flux_type == SDP_MEM_COMPLEX_DOUBLE && dir_type == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        image_to_flmn<double, complex<double> >(
                subgrid_image, plan->theta, plan->pswf_sg, flux, lmn, status
        );
        dft<double, complex<double>, double, double>(
                plan, uvw, start_chs, end_chs, flux, lmn,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                vis, status
        );
    }
    else if (flux_type == SDP_MEM_COMPLEX_FLOAT && dir_type == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        image_to_flmn<float, complex<float> >(
                subgrid_image, plan->theta, plan->pswf_sg, flux, lmn, status
        );
        dft<float, complex<float>, float, float>(
                plan, uvw, start_chs, end_chs, flux, lmn,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                vis, status
        );
    }
    else if (flux_type == SDP_MEM_DOUBLE && dir_type == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        image_to_flmn<double, double>(
                subgrid_image, plan->theta, plan->pswf_sg, flux, lmn, status
        );
        dft<double, double, double, double>(
                plan, uvw, start_chs, end_chs, flux, lmn,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                vis, status
        );
    }
    else if (flux_type == SDP_MEM_FLOAT && dir_type == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        image_to_flmn<float, float>(
                subgrid_image, plan->theta, plan->pswf_sg, flux, lmn, status
        );
        dft<float, float, float, float>(
                plan, uvw, start_chs, end_chs, flux, lmn,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                vis, status
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }

    // Free scratch memory.
    sdp_mem_free(flux);
    sdp_mem_free(lmn);
}


void sdp_gridder_direct_degrid_correct(
        sdp_GridderDirect* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(facet) < 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }

    // Shift PSWF by facet offsets.
    sdp_Mem* mem_pswf_l = sdp_mem_create_copy(plan->pswf, SDP_MEM_CPU, status);
    sdp_Mem* mem_pswf_m = sdp_mem_create_copy(plan->pswf, SDP_MEM_CPU, status);
    if (*status) return;
    const double* pswf = (const double*) sdp_mem_data_const(plan->pswf);
    double* pswf_l = (double*) sdp_mem_data(mem_pswf_l);
    double* pswf_m = (double*) sdp_mem_data(mem_pswf_m);
    const int64_t pswf_size = sdp_mem_shape_dim(plan->pswf, 0);
    for (int64_t i = 0; i < pswf_size; ++i)
    {
        int64_t il = i + facet_offset_l;
        int64_t im = i + facet_offset_m;
        if (il >= pswf_size) il -= pswf_size;
        if (im >= pswf_size) im -= pswf_size;
        pswf_l[i] = pswf[il];
        pswf_m[i] = pswf[im];
    }

    // Apply grid correction for the appropriate data type.
    switch (sdp_mem_type(facet))
    {
    case SDP_MEM_DOUBLE:
        grid_corr<double>(plan, facet, pswf_l, pswf_m, status);
        break;
    case SDP_MEM_COMPLEX_DOUBLE:
        grid_corr<complex<double> >(plan, facet, pswf_l, pswf_m, status);
        break;
    case SDP_MEM_FLOAT:
        grid_corr<float>(plan, facet, pswf_l, pswf_m, status);
        break;
    case SDP_MEM_COMPLEX_FLOAT:
        grid_corr<complex<float> >(plan, facet, pswf_l, pswf_m, status);
        break;
    default:
        *status = SDP_ERR_DATA_TYPE;
        break;
    }
    sdp_mem_free(mem_pswf_l);
    sdp_mem_free(mem_pswf_m);
}


void sdp_gridder_direct_grid(
        sdp_GridderDirect* plan,
        const sdp_Mem* vis,
        const sdp_Mem* uvw,
        const sdp_Mem* start_chs,
        const sdp_Mem* end_chs,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* subgrid_image,
        int subgrid_offset_u,
        int subgrid_offset_v,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(subgrid_image) < 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }

    // Allocate space for coordinates of all pixels in the image.
    int64_t num_src = sdp_mem_num_elements(subgrid_image);
    const sdp_MemType flux_type = sdp_mem_type(subgrid_image);
    const sdp_MemType dir_type = sdp_mem_type(uvw);
    const sdp_MemLocation loc = sdp_mem_location(vis);
    const int64_t lmn_shape[] = {num_src, 3};
    sdp_Mem* lmn = sdp_mem_create(dir_type, loc, 2, lmn_shape, status);

    // Convert image into positions, and use DFT for gridding.
    if (flux_type == SDP_MEM_COMPLEX_DOUBLE && dir_type == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        image_to_lmn<double>(subgrid_image, plan->theta, lmn, status);
        idft<double, complex<double>, double, double>(
                plan, uvw, vis, start_chs, end_chs, lmn,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                subgrid_image, status
        );
    }
    else if (flux_type == SDP_MEM_COMPLEX_FLOAT && dir_type == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        image_to_lmn<float>(subgrid_image, plan->theta, lmn, status);
        idft<float, complex<float>, float, float>(
                plan, uvw, vis, start_chs, end_chs, lmn,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                subgrid_image, status
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }

    // Free scratch memory.
    sdp_mem_free(lmn);
}


void sdp_gridder_direct_grid_correct(
        sdp_GridderDirect* plan,
        sdp_Mem* facet,
        int facet_offset_l,
        int facet_offset_m,
        sdp_Error* status
)
{
    // Grid correction and degrid correction are the same in the notebook.
    sdp_gridder_direct_degrid_correct(
            plan, facet, facet_offset_l, facet_offset_m, status
    );
}


void sdp_gridder_direct_free(sdp_GridderDirect* plan)
{
    sdp_mem_free(plan->pswf);
    sdp_mem_free(plan->pswf_sg);
    free(plan);
}
