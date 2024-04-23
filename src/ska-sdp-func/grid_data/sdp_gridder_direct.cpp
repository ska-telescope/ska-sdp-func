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


template<typename T>
static void get_pixels(
        const sdp_Mem* image,
        double theta,
        const sdp_Mem* taper_func_1d,
        sdp_Mem* mem_flux,
        sdp_Mem* mem_l,
        sdp_Mem* mem_m,
        sdp_Mem* mem_n
)
{
    const int64_t image_size = sdp_mem_shape_dim(image, 0);
    const T* pix = (const T*)sdp_mem_data_const(image);
    const double* taper = (const double*)sdp_mem_data_const(taper_func_1d);
    T* flux = (T*)sdp_mem_data(mem_flux);
    T* l = (T*)sdp_mem_data(mem_l);
    T* m = (T*)sdp_mem_data(mem_m);
    T* n = (T*)sdp_mem_data(mem_n);
    int64_t k = 0;
    for (int64_t il = 0; il < image_size; ++il)
    {
        for (int64_t im = 0; im < image_size; ++im)
        {
            const T pix_val = pix[il * image_size + im];
            if (pix_val != 0.0)
            {
                flux[k] = pix_val * taper[il] * taper[im];
                l[k] = (il - image_size / 2) * theta / image_size;
                m[k] = (im - image_size / 2) * theta / image_size;
                n[k] = sqrt(1.0 - l[k] * l[k] - m[k] * m[k]) - 1.0;
                k++;
            }
        }
    }
}


template<
        typename DIR_TYPE,
        typename FLUX_TYPE,
        typename UVW_TYPE,
        typename VIS_TYPE
>
static void dft(
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
    const int num_chan = sdp_mem_shape_dim(mem_vis, 1);
    const int num_sources = sdp_mem_shape_dim(mem_flux, 0);
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
                const double phase = -2.0 * M_PI *
                        (l[s] * u + m[s] * v + n[s] * w);
                const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                local_vis += flux[s] * phasor;
            }

            // Store local visibility.
            vis[i * num_chan + c] = local_vis;
        }
    }
}


static void image_to_flmn(
        const sdp_Mem* image,
        double theta,
        const sdp_Mem* taper_func_1d,
        sdp_Mem** mem_flux,
        sdp_Mem** mem_l,
        sdp_Mem** mem_m,
        sdp_Mem** mem_n,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(image) < 2)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        return;
    }

    // Count non-zero pixels in image.
    int64_t num_sources = 0;
    const int64_t num_pix = sdp_mem_num_elements(image);
    const sdp_MemType type = sdp_mem_type(image);
    if (type == SDP_MEM_DOUBLE)
    {
        const double* pix = (const double*)sdp_mem_data_const(image);
        for (int64_t i = 0; i < num_pix; ++i)
            if (pix[i] != 0) num_sources++;
    }
    else if (type == SDP_MEM_FLOAT)
    {
        const float* pix = (const float*)sdp_mem_data_const(image);
        for (int64_t i = 0; i < num_pix; ++i)
            if (pix[i] != 0) num_sources++;
    }

    // Allocate space for pixel data (flux values and direction cosines).
    const int64_t data_shape[] = {num_sources};
    *mem_flux = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);
    *mem_l = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);
    *mem_m = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);
    *mem_n = sdp_mem_create(type, SDP_MEM_CPU, 1, data_shape, status);

    // Store pixel data.
    if (type == SDP_MEM_DOUBLE)
    {
        get_pixels<double>(image, theta, taper_func_1d,
                *mem_flux, *mem_l, *mem_m, *mem_n
        );
    }
    else if (type == SDP_MEM_FLOAT)
    {
        get_pixels<float>(image, theta, taper_func_1d,
                *mem_flux, *mem_l, *mem_m, *mem_n
        );
    }
}


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

    // Convert image into positions.
    sdp_Mem* flux = 0, * l = 0, * m = 0, * n = 0;
    image_to_flmn(subgrid_image, plan->theta, plan->pswf_sg,
            &flux, &l, &m, &n, status
    );
    if (*status || sdp_mem_num_elements(flux) == 0) return;

    // Use DFT for degridding.
    if (sdp_mem_type(flux) == SDP_MEM_DOUBLE &&
            sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE)
    {
        dft<double, double, double, double>(
                plan, uvw, start_chs, end_chs, flux, l, m, n,
                subgrid_offset_u, subgrid_offset_v, freq0_hz, dfreq_hz,
                vis, status
        );
    }
    else if (sdp_mem_type(flux) == SDP_MEM_FLOAT &&
            sdp_mem_type(uvw) == SDP_MEM_FLOAT &&
            sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT)
    {
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


void sdp_gridder_direct_free(sdp_GridderDirect* plan)
{
    sdp_mem_free(plan->pswf);
    sdp_mem_free(plan->pswf_sg);
    free(plan);
}
