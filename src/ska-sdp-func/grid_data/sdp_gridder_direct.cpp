/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/grid_data/sdp_gridder_direct.h"

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

// Local function for prediction of visibilities via direct FT.
template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
void dft(
        sdp_GridderDirect* plan,
        const sdp_Mem* mem_uvw,
        const sdp_Mem* mem_start_chs,
        const sdp_Mem* mem_end_chs,
        const sdp_Mem* mem_flux,
        const sdp_Mem* mem_l,
        const sdp_Mem* mem_m,
        const sdp_Mem* mem_n,
        int subgrid_offset_u,
        int subgrid_offset_v,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* mem_vis,
        sdp_Error* status
)
{
    if (*status) return;

    // Get data pointers.
    const int64_t num_uvw = sdp_mem_shape_dim(mem_uvw, 0);
    const int num_chan = (int) sdp_mem_shape_dim(mem_vis, 1);
    const int num_sources = (int) sdp_mem_shape_dim(mem_flux, 0);
    const int* start_chs = (const int*) sdp_mem_data_const(mem_start_chs);
    const int* end_chs = (const int*) sdp_mem_data_const(mem_end_chs);
    const UVW_TYPE* uvw = (const UVW_TYPE*) sdp_mem_data_const(mem_uvw);
    const FLUX_TYPE* flux = (const FLUX_TYPE*) sdp_mem_data_const(mem_flux);
    const DIR_TYPE* l = (const DIR_TYPE*) sdp_mem_data_const(mem_l);
    const DIR_TYPE* m = (const DIR_TYPE*) sdp_mem_data_const(mem_m);
    const DIR_TYPE* n = (const DIR_TYPE*) sdp_mem_data_const(mem_n);
    complex<VIS_TYPE>* vis = (complex<VIS_TYPE>*) sdp_mem_data(mem_vis);

    // Scale subgrid offset values.
    const UVW_TYPE du = (UVW_TYPE) subgrid_offset_u / plan->theta;
    const UVW_TYPE dv = (UVW_TYPE) subgrid_offset_v / plan->theta;

    // Loop over uvw values.
    #pragma omp parallel for
    for (int64_t i = 0; i < num_uvw; ++i)
    {
        // Skip if there's no visibility to degrid.
        if (start_chs[i] >= end_chs[i])
            continue;

        // Loop over channels.
        for (int c = 0; c < num_chan; ++c)
        {
            const UVW_TYPE inv_wave = (freq0_hz + dfreq_hz * c) / C_0;

            // Scale and shift uvws.
            const UVW_TYPE u = uvw[3 * i + 0] * inv_wave - du;
            const UVW_TYPE v = uvw[3 * i + 1] * inv_wave - dv;
            const UVW_TYPE w = uvw[3 * i + 2] * inv_wave;

            // Loop over sources.
            complex<VIS_TYPE> local_vis = 0;
            for (int s = 0; s < num_sources; ++s)
            {
                const VIS_TYPE phase = -2.0 * M_PI *
                        (l[s] * u + m[s] * v + n[s] * w);
                const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                local_vis += flux[s] * phasor;
            }

            // Store local visibility.
            vis[i * num_chan + c] = local_vis;
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
        const sdp_Mem* mem_uvw,
        const sdp_Mem* mem_vis,
        const sdp_Mem* mem_start_chs,
        const sdp_Mem* mem_end_chs,
        const sdp_Mem* mem_l,
        const sdp_Mem* mem_m,
        const sdp_Mem* mem_n,
        int subgrid_offset_u,
        int subgrid_offset_v,
        double freq0_hz,
        double dfreq_hz,
        sdp_Mem* mem_image,
        sdp_Error* status
)
{
    if (*status) return;

    // Get data pointers.
    const int64_t num_uvw = sdp_mem_shape_dim(mem_uvw, 0);
    const int64_t image_size = sdp_mem_shape_dim(mem_image, 0);
    const int num_chan = (int) sdp_mem_shape_dim(mem_vis, 1);
    const int* start_chs = (const int*) sdp_mem_data_const(mem_start_chs);
    const int* end_chs = (const int*) sdp_mem_data_const(mem_end_chs);
    const UVW_TYPE* uvw = (const UVW_TYPE*) sdp_mem_data_const(mem_uvw);
    const DIR_TYPE* l = (const DIR_TYPE*) sdp_mem_data_const(mem_l);
    const DIR_TYPE* m = (const DIR_TYPE*) sdp_mem_data_const(mem_m);
    const DIR_TYPE* n = (const DIR_TYPE*) sdp_mem_data_const(mem_n);
    const complex<VIS_TYPE>* vis =
            (const complex<VIS_TYPE>*) sdp_mem_data_const(mem_vis);
    FLUX_TYPE* image = (FLUX_TYPE*) sdp_mem_data(mem_image);
    const double* pswf = (const double*) sdp_mem_data_const(plan->pswf_sg);

    // Scale subgrid offset values.
    const UVW_TYPE du = (UVW_TYPE) subgrid_offset_u / plan->theta;
    const UVW_TYPE dv = (UVW_TYPE) subgrid_offset_v / plan->theta;

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
                if (start_chs[i] >= end_chs[i])
                    continue;

                // Loop over channels.
                for (int c = 0; c < num_chan; ++c)
                {
                    const UVW_TYPE inv_wave = (freq0_hz + dfreq_hz * c) / C_0;

                    // Scale and shift uvws.
                    const UVW_TYPE u = uvw[3 * i + 0] * inv_wave - du;
                    const UVW_TYPE v = uvw[3 * i + 1] * inv_wave - dv;
                    const UVW_TYPE w = uvw[3 * i + 2] * inv_wave;

                    const VIS_TYPE phase = 2.0 * M_PI *
                            (l[s] * u + m[s] * v + n[s] * w);
                    const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                    local_pix += vis[i * num_chan + c] * phasor;
                }
            }

            // Store local pixel value, appropriately tapered by the PSWF.
            // (We can't taper the whole image, as the input may be nonzero.)
            image[s] += local_pix * (FLUX_TYPE) (pswf[il] * pswf[im]);
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
template<typename T>
void image_to_flmn(
        const sdp_Mem* mem_image,
        double theta,
        const sdp_Mem* mem_pswf,
        sdp_Mem** mem_flux,
        sdp_Mem** mem_l,
        sdp_Mem** mem_m,
        sdp_Mem** mem_n,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(mem_image) < 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }

    // Count non-zero pixels in image.
    int64_t num_sources = 0;
    const int64_t num_pix = sdp_mem_num_elements(mem_image);
    const T* pix = (const T*) sdp_mem_data_const(mem_image);
    for (int64_t i = 0; i < num_pix; ++i)
        if (pix[i] != 0) num_sources++;

    // Allocate space for pixel data (flux values and direction cosines).
    const sdp_MemType type = sdp_mem_type(mem_image);
    const int64_t data_shape[] = {num_sources};
    *mem_flux = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);
    *mem_l = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);
    *mem_m = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);
    *mem_n = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);

    // Store pixel data.
    if (*status) return;
    const int64_t image_size = sdp_mem_shape_dim(mem_image, 0);
    const double* pswf = (const double*) sdp_mem_data_const(mem_pswf);
    T* flux = (T*) sdp_mem_data(*mem_flux);
    T* l = (T*) sdp_mem_data(*mem_l);
    T* m = (T*) sdp_mem_data(*mem_m);
    T* n = (T*) sdp_mem_data(*mem_n);
    int64_t k = 0;
    for (int64_t il = 0; il < image_size; ++il)
    {
        for (int64_t im = 0; im < image_size; ++im)
        {
            const T pix_val = pix[il * image_size + im];
            if (pix_val != 0.0)
            {
                flux[k] = pix_val * pswf[il] * pswf[im];
                idx_to_dir(il, im, image_size, theta, l[k], m[k], n[k]);
                k++;
            }
        }
    }
}


// Local function to convert all image pixel positions to coordinates.
template<typename T>
void image_to_lmn(
        const sdp_Mem* mem_image,
        double theta,
        sdp_Mem** mem_l,
        sdp_Mem** mem_m,
        sdp_Mem** mem_n,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(mem_image) < 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }

    // Allocate space for pixel data (direction cosines only).
    const sdp_MemType type = sdp_mem_type(mem_image);
    const int64_t data_shape[] = {sdp_mem_num_elements(mem_image)};
    *mem_l = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);
    *mem_m = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);
    *mem_n = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);

    // Store pixel data.
    if (*status) return;
    const int64_t image_size = sdp_mem_shape_dim(mem_image, 0);
    T* l = (T*) sdp_mem_data(*mem_l);
    T* m = (T*) sdp_mem_data(*mem_m);
    T* n = (T*) sdp_mem_data(*mem_n);
    for (int64_t il = 0, k = 0; il < image_size; ++il)
        for (int64_t im = 0; im < image_size; ++im, ++k)
            idx_to_dir(il, im, image_size, theta, l[k], m[k], n[k]);
}


// Local function to apply grid correction.
template<typename T>
void grid_corr(
        const sdp_GridderDirect* plan,
        sdp_Mem* mem_facet,
        const double* pswf_l,
        const double* pswf_m,
        sdp_Error* status
)
{
    if (*status) return;

    // Apply portion of shifted PSWF to facet.
    T* facet = (T*) sdp_mem_data(mem_facet);
    const int64_t num_l = sdp_mem_shape_dim(mem_facet, 0);
    const int64_t num_m = sdp_mem_shape_dim(mem_facet, 1);
    for (int64_t il = 0, k = 0; il < num_l; ++il)
    {
        const int64_t pl = il + (plan->image_size / 2 - num_l / 2);
        for (int64_t im = 0; im < num_m; ++im, ++k)
        {
            const int64_t pm = im + (plan->image_size / 2 - num_m / 2);
            facet[k] /= (T) (pswf_l[pl] * pswf_m[pm]);
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

    // Arrays for nonzero pixel values in image.
    sdp_Mem* flux = 0, * l = 0, * m = 0, * n = 0;

    // Convert image into positions and flux values for all nonzero pixels,
    // and use DFT for degridding.
    if (sdp_mem_type(subgrid_image) == SDP_MEM_DOUBLE &&
            sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        image_to_flmn<double>(subgrid_image, plan->theta, plan->pswf_sg,
                &flux, &l, &m, &n, status
        );
        dft<double, double, double, double>(
                plan, uvw, start_chs, end_chs, flux, l, m, n,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                vis, status
        );
    }
    else if (sdp_mem_type(subgrid_image) == SDP_MEM_FLOAT &&
            sdp_mem_type(uvw) == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        image_to_flmn<float>(subgrid_image, plan->theta, plan->pswf_sg,
                &flux, &l, &m, &n, status
        );
        dft<float, float, float, float>(
                plan, uvw, start_chs, end_chs, flux, l, m, n,
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
    sdp_mem_free(l);
    sdp_mem_free(m);
    sdp_mem_free(n);
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

    // Arrays for all pixels in image.
    sdp_Mem* l = 0, * m = 0, * n = 0;

    // Convert image into positions, and use DFT for gridding.
    if (sdp_mem_type(subgrid_image) == SDP_MEM_COMPLEX_DOUBLE &&
            sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        image_to_lmn<double>(subgrid_image, plan->theta, &l, &m, &n, status);
        idft<double, complex<double>, double, double>(
                plan, uvw, vis, start_chs, end_chs, l, m, n,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                subgrid_image, status
        );
    }
    else if (sdp_mem_type(subgrid_image) == SDP_MEM_COMPLEX_FLOAT &&
            sdp_mem_type(uvw) == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
        image_to_lmn<float>(subgrid_image, plan->theta, &l, &m, &n, status);
        idft<float, complex<float>, float, float>(
                plan, uvw, vis, start_chs, end_chs, l, m, n,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                subgrid_image, status
        );
    }
    else
    {
        *status = SDP_ERR_DATA_TYPE;
    }

    // Free scratch memory.
    sdp_mem_free(l);
    sdp_mem_free(m);
    sdp_mem_free(n);
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
