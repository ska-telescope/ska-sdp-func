/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>
#if defined(AVX512) || defined(AVX2)
#include <x86intrin.h>
#endif // AVX512 || AVX2

#include "ska-sdp-func/fourier_transforms/sdp_fft.h"
#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/grid_data/sdp_gridder_grid_correct.h"
#include "ska-sdp-func/grid_data/sdp_gridder_utils.h"
#include "ska-sdp-func/grid_data/sdp_gridder_wtower_uvw.h"
#include "ska-sdp-func/math/sdp_math_macros.h"
#include "ska-sdp-func/sdp_func_global.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"


using std::complex;

struct sdp_GridderWtowerUVW {
        int image_size;
        int subgrid_size;
        double theta;
        double w_step;
        double shear_u;
        double shear_v;
        int support;
        int oversampling;
        int w_support;
        int w_oversampling;
        int num_w_planes[2];
        sdp_Mem *uv_kernel;
        sdp_Mem *uv_kernel_gpu;
        sdp_Mem *w_kernel;
        sdp_Mem *w_kernel_gpu;
        sdp_Mem *w_pattern;
        sdp_Mem *w_pattern_gpu;
};

// Begin anonymous namespace for file-local functions.
namespace {

// There is no direct equivalent AVX2 instruction to _mm512_reduce_pd, 
// so we need to define one for ourself.
inline double _mm256_reduce_add_pd(__m256d vec) {
	// Shuffle the high and low halves of the vector
	// and move high 128-bit to low
	__m256d shuf = _mm256_permute2f128_pd(vec, vec, 1);
	// Add high and low halves
	__m256d sum = _mm256_add_pd(vec, shuf);

	// Shuffle within the low 128 bits,
	// and swap the adjacent pairs
	shuf = _mm256_permute_pd(sum, 0b0101);
	// Add adjacent pairs
	sum = _mm256_add_pd(sum, shuf);

	// Extract the scalar sum, each element is the sum
	return _mm_cvtsd_f64(_mm256_castpd256_pd128(sum));
}

inline float _mm256_reduce_add_ps(__m256d vec) {
	// Shuffle the high and low halves of the vector
	// and move high 128-bit to low
	__m256d shuf = _mm256_permute2f128_ps(vec, vec, 1);
	// Add high and low halves
	__m256d sum = _mm256_add_ps(vec, shuf);

	// Shuffle within the low 128 bits,
	// and swap the adjacent pairs
	shuf = _mm256_permute_ps(sum, 0b0101);
	// Add adjacent pairs
	sum = _mm256_add_ps(sum, shuf);

	// Extract the scalar sum, each element is the sum
	return _mm_cvtsd_f64(_mm256_castpd256_ps128(sum));
}

// Local function to do the degridding.
template <typename SUBGRID_TYPE, typename UVW_TYPE, typename VIS_TYPE>
void degrid(const sdp_GridderWtowerUVW *plan, const sdp_Mem *subgrids,
            int w_plane, int subgrid_offset_u, int subgrid_offset_v,
            int subgrid_offset_w, double freq0_hz, double dfreq_hz,
            int64_t start_row, int64_t end_row, const sdp_Mem *uvws,
            const sdp_MemViewCpu<const int, 1> &start_chs,
            const sdp_MemViewCpu<const int, 1> &end_chs, sdp_Mem *vis,
            sdp_Error *status) {
    if (*status) return;

    sdp_MemViewCpu<const SUBGRID_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<VIS_TYPE, 2> vis_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(vis, &vis_, status);
    const double *RESTRICT uv_kernel = (const double *)sdp_mem_data_const(plan->uv_kernel);
    const double *RESTRICT w_kernel = (const double *)sdp_mem_data_const(plan->w_kernel);

    const int half_subgrid = plan->subgrid_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;
    const double w_step = plan->w_step;
    const double theta_ov = theta * oversample;
    const double w_step_ov = 1.0 / w_step * w_oversample;
    const int half_sg_size_ov = (half_subgrid - support / 2 + 1) * oversample;

// Loop over rows. Each row contains visibilities for all channels.
#pragma omp parallel for schedule(dynamic, 500)
    for (int64_t i_row = start_row; i_row < end_row; ++i_row) {
        // Skip if there's no visibility to degrid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch) continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws_(i_row, 0), uvws_(i_row, 1), uvws_(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
        const double max_w = (w_plane + subgrid_offset_w) * w_step;
        sdp_gridder_clamp_channels_inline(uvw[2], freq0_hz, dfreq_hz, &start_ch, &end_ch, min_w, max_w);
        if (start_ch >= end_ch) continue;

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / theta;
        uvw0[1] -= subgrid_offset_v / theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane - 1) * w_step);

        // Bounds check.
        const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
        const double u_max = ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
        const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
        const double v_max = ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));
        if (u_min < -half_subgrid || u_max >= half_subgrid ||
            v_min < -half_subgrid || v_max >= half_subgrid) {
            continue;
        }

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++) {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region centred approximately
            // on visibility in oversampled coordinates (this is subgrid-local,
            // so should be safe for overflows).
            const int iu0_ov = int(round(u * theta_ov)) + half_sg_size_ov;
            const int iv0_ov = int(round(v * theta_ov)) + half_sg_size_ov;
            const int iw0_ov = int(round(w * w_step_ov));
            const int iu0 = iu0_ov / oversample;
            const int iv0 = iv0_ov / oversample;

            // Determine which kernel to use.
            const int u_off = (iu0_ov % oversample) * support;
            const int v_off = (iv0_ov % oversample) * support;
            const int w_off = (iw0_ov % w_oversample) * w_support;

            #if defined(AVX512)
            __m512d local_vis_real = _mm512_setzero_pd();
            __m512d local_vis_imag = _mm512_setzero_pd();
            
            for (int iw = 0; iw < w_support; ++iw) {
                #ifdef PREFETCH
                if(iw + 1 < w_support) {
                    _mm_prefetch(&w_kernel[w_off + iw + 1], _MM_HINT_T0);
                }
                #endif // PREFETCH
                
                const __m512d w_kernel_val = _mm512_set1_pd(w_kernel[w_off + iw]);
                __m512d local_vis_u_real = _mm512_setzero_pd();
                __m512d local_vis_u_imag = _mm512_setzero_pd();

                for (int iu = 0; iu < support; ++iu) {
                    #ifdef PREFETCH
                    if(iu + 1 < support) {
                        _mm_prefetch(&uv_kernel[u_off + iu + 1], _MM_HINT_T0);
                    }
                    #endif // PREFETCH
                    
                    const int ix_u = iu0 + iu;
                    const __m512d u_kernel_val = _mm512_set1_pd(uv_kernel[u_off + iu]);
                    __m512d local_vis_v_real = _mm512_setzero_pd();
                    __m512d local_vis_v_imag = _mm512_setzero_pd();

                    // Process 8 v elements at once
                    for (int iv = 0; iv < support; iv += 8) {
                        #ifdef PREFETCH
                        if(iv + 8 < support) {
                            _mm_prefetch(&uv_kernel[v_off + iv + 8], _MM_HINT_T0);
                            _mm_prefetch(&subgrids_(iw, ix_u, iv0 + iv + 8), _MM_HINT_T0);
                        }
                        #endif // PREFETCH
                        
                        const int ix_v = iv0 + iv;
                        __m512d v_kernel_vals = _mm512_load_pd(&uv_kernel[v_off + iv]);
                        __m512d subgrid_real = _mm512_setr_pd(
                            subgrids_(iw, ix_u, ix_v).real(),
                            subgrids_(iw, ix_u, ix_v + 1).real(),
                            subgrids_(iw, ix_u, ix_v + 2).real(),
                            subgrids_(iw, ix_u, ix_v + 3).real(),
                            subgrids_(iw, ix_u, ix_v + 4).real(),
                            subgrids_(iw, ix_u, ix_v + 5).real(),
                            subgrids_(iw, ix_u, ix_v + 6).real(),
                            subgrids_(iw, ix_u, ix_v + 7).real()
                        );
                        __m512d subgrid_imag = _mm512_setr_pd(
                            subgrids_(iw, ix_u, ix_v).imag(),
                            subgrids_(iw, ix_u, ix_v + 1).imag(),
                            subgrids_(iw, ix_u, ix_v + 2).imag(),
                            subgrids_(iw, ix_u, ix_v + 3).imag(),
                            subgrids_(iw, ix_u, ix_v + 4).imag(),
                            subgrids_(iw, ix_u, ix_v + 5).imag(),
                            subgrids_(iw, ix_u, ix_v + 6).imag(),
                            subgrids_(iw, ix_u, ix_v + 7).imag()
                        );

                        local_vis_v_real = _mm512_fmadd_pd(v_kernel_vals, subgrid_real, local_vis_v_real);
                        local_vis_v_imag = _mm512_fmadd_pd(v_kernel_vals, subgrid_imag, local_vis_v_imag);
                    }

                    // Reduce (horizontal add) the v dim results
                    double vis_v_real = _mm512_reduce_add_pd(local_vis_v_real);
                    double vis_v_imag = _mm512_reduce_add_pd(local_vis_v_imag);

                    local_vis_u_real = _mm512_fmadd_pd(u_kernel_val, _mm512_set1_pd(vis_v_real), local_vis_u_real);
                    local_vis_u_imag = _mm512_fmadd_pd(u_kernel_val, _mm512_set1_pd(vis_v_imag), local_vis_u_imag);
                }

                local_vis_real = _mm512_fmadd_pd(w_kernel_val, local_vis_u_real, local_vis_real);
                local_vis_imag = _mm512_fmadd_pd(w_kernel_val, local_vis_u_imag, local_vis_imag);
            }

            double local_vis_real_reduced = _mm512_reduce_add_pd(local_vis_real);
            double local_vis_imag_reduced = _mm512_reduce_add_pd(local_vis_imag);
            vis_(i_row, c) += VIS_TYPE(local_vis_real_reduced, local_vis_imag_reduced);

            #elif defined(AVX2)
            __m256d local_vis_real = _mm256_setzero_pd();
            __m256d local_vis_imag = _mm256_setzero_pd();
            
            for (int iw = 0; iw < w_support; ++iw) {
                #ifdef PREFETCH
                if(iw + 1 < w_support) {
                    _mm_prefetch(&w_kernel[w_off + iw + 1], _MM_HINT_T0);
                }
                #endif // PREFETCH

                const __m256d w_kernel_val = _mm256_set1_pd(w_kernel[w_off + iw]);
                __m256d local_vis_u_real = _mm256_setzero_pd();
                __m256d local_vis_u_imag = _mm256_setzero_pd();

                for (int iu = 0; iu < support; ++iu) {
                    #ifdef PREFETCH
                    if(iu + 1 < support) {
                        _mm_prefetch(&uv_kernel[u_off + iu + 1], _MM_HINT_T0);
                    }
                    #endif // PREFETCH

                    const int ix_u = iu0 + iu;
                    const __m256d u_kernel_val = _mm256_set1_pd(uv_kernel[u_off + iu]);
                    __m256d local_vis_v_real = _mm256_setzero_pd();
                    __m256d local_vis_v_imag = _mm256_setzero_pd();

                    // Process 4 v elements at once
                    for (int iv = 0; iv < support; iv += 4) {
                        #ifdef PREFETCH
                        if(iv + 4 < support) {
                            _mm_prefetch(&uv_kernel[v_off + iv + 4], _MM_HINT_T0);
                            _mm_prefetch(&subgrids_(iw, ix_u, iv0 + iv + 4), _MM_HINT_T0);
                        }
                        #endif // PREFETCH

                        const int ix_v = iv0 + iv;
                        __m256d v_kernel_vals = _mm256_load_pd(&uv_kernel[v_off + iv]);
                        __m256d subgrid_real = _mm256_setr_pd(
                            subgrids_(iw, ix_u, ix_v).real(),
                            subgrids_(iw, ix_u, ix_v + 1).real(),
                            subgrids_(iw, ix_u, ix_v + 2).real(),
                            subgrids_(iw, ix_u, ix_v + 3).real()
                        );
                        __m256d subgrid_imag = _mm256_setr_pd(
                            subgrids_(iw, ix_u, ix_v).imag(),
                            subgrids_(iw, ix_u, ix_v + 1).imag(),
                            subgrids_(iw, ix_u, ix_v + 2).imag(),
                            subgrids_(iw, ix_u, ix_v + 3).imag()
                        );

                        local_vis_v_real = _mm256_fmadd_pd(v_kernel_vals, subgrid_real, local_vis_v_real);
                        local_vis_v_imag = _mm256_fmadd_pd(v_kernel_vals, subgrid_imag, local_vis_v_imag);
                    }

                    // Reduce (horizontal add) the v dim results
                    double vis_v_real = _mm256_reduce_add_pd(local_vis_v_real);
                    double vis_v_imag = _mm256_reduce_add_pd(local_vis_v_imag);

                    local_vis_u_real = _mm256_fmadd_pd(u_kernel_val, _mm256_set1_pd(vis_v_real), local_vis_u_real);
                    local_vis_u_imag = _mm256_fmadd_pd(u_kernel_val, _mm256_set1_pd(vis_v_imag), local_vis_u_imag);
                }

                local_vis_real = _mm256_fmadd_pd(w_kernel_val, local_vis_u_real, local_vis_real);
                local_vis_imag = _mm256_fmadd_pd(w_kernel_val, local_vis_u_imag, local_vis_imag);
            }

            double local_vis_real_reduced = _mm256_reduce_add_pd(local_vis_real);
            double local_vis_imag_reduced = _mm256_reduce_add_pd(local_vis_imag);
            vis_(i_row, c) += VIS_TYPE(local_vis_real_reduced, local_vis_imag_reduced);

            #else // AVX512 || AVX2
            // Degrid visibility.
            SUBGRID_TYPE local_vis = (SUBGRID_TYPE)0;
            for (int iw = 0; iw < w_support; ++iw) {
                SUBGRID_TYPE local_vis_u = (SUBGRID_TYPE)0;
#pragma GCC ivdep
#pragma GCC unroll(8)
                for (int iu = 0; iu < support; ++iu) {
                    const int ix_u = iu0 + iu;
                    SUBGRID_TYPE local_vis_v = (SUBGRID_TYPE)0;
#pragma GCC ivdep
#pragma GCC unroll(8)
                    for (int iv = 0; iv < support; ++iv) {
                        const int ix_v = iv0 + iv;
                        local_vis_v += ((SUBGRID_TYPE)uv_kernel[v_off + iv] * subgrids_(iw, ix_u, ix_v));
                    }
                    local_vis_u += ((SUBGRID_TYPE)uv_kernel[u_off + iu] * local_vis_v);
                }
                local_vis += (SUBGRID_TYPE)w_kernel[w_off + iw] * local_vis_u;
            }
            vis_(i_row, c) += (VIS_TYPE)local_vis;
            #endif // AVX512 || AVX2
        }
    }
}

// Local function to call the CPU degridding kernel.
void degrid_cpu(const sdp_GridderWtowerUVW *plan, const sdp_Mem *subgrids,
                int w_plane, int subgrid_offset_u, int subgrid_offset_v,
                int subgrid_offset_w, double freq0_hz, double dfreq_hz,
                int64_t start_row, int64_t end_row, const sdp_Mem *uvws,
                const sdp_Mem *start_chs, const sdp_Mem *end_chs, sdp_Mem *vis,
                sdp_Error *status) {
    if (*status) return;

    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE &&
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {

        degrid<complex<double>, double, complex<double>>(plan, subgrids, w_plane, 
                                                         subgrid_offset_u, subgrid_offset_v, subgrid_offset_w, 
                                                         freq0_hz, dfreq_hz, start_row, end_row, uvws, 
                                                         start_chs_, end_chs_, vis, status);
    } 
    else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {

        degrid<complex<float>, double, complex<float>>(plan, subgrids, w_plane, 
                                                       subgrid_offset_u, subgrid_offset_v, subgrid_offset_w, 
                                                       freq0_hz, dfreq_hz, start_row, end_row, uvws, 
                                                       start_chs_, end_chs_, vis, status);

    } 
    else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {

        degrid<complex<float>, float, complex<float>>(plan, subgrids, w_plane, 
                                                      subgrid_offset_u, subgrid_offset_v, subgrid_offset_w, 
                                                      freq0_hz, dfreq_hz, start_row, end_row, uvws, 
                                                      start_chs_, end_chs_, vis, status);
    } 
    else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types: " "subgrids has type %s; uvws has type %s; vis has type %s",
                      sdp_mem_type_name(sdp_mem_type(subgrids)),
                      sdp_mem_type_name(sdp_mem_type(uvws)),
                      sdp_mem_type_name(sdp_mem_type(vis)));
    }
}

// Local function to call the GPU degridding kernel.
void degrid_gpu(const sdp_GridderWtowerUVW *plan, const sdp_Mem *subgrids,
                int w_plane, int subgrid_offset_u, int subgrid_offset_v,
                int subgrid_offset_w, double freq0_hz, double dfreq_hz,
                int64_t start_row, int64_t end_row, const sdp_Mem *uvws,
                const sdp_Mem *start_chs, const sdp_Mem *end_chs, sdp_Mem *vis,
                sdp_Error *status) {
    if (*status) return;

    uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
    const char *kernel_name = 0;
    int is_dbl_uvw = 0;
    int is_dbl_vis = 0;
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    sdp_MemViewGpu<const double, 2> uvws_dbl;
    sdp_MemViewGpu<const float, 2> uvws_flt;
    sdp_MemViewGpu<complex<double>, 2> vis_dbl;
    sdp_MemViewGpu<complex<float>, 2> vis_flt;
    sdp_MemViewGpu<const int, 1> start_chs_, end_chs_;

    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);
    if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE &&
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {
        is_dbl_uvw = 1;
        is_dbl_vis = 1;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_dbl, status);

        kernel_name = "sdp_gridder_wtower_degrid" "<complex<double>, double, complex<double> >";
    } 
    else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        is_dbl_uvw = 1;
        is_dbl_vis = 0;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);

        kernel_name = "sdp_gridder_wtower_degrid" "<complex<float>, double, complex<float> >";
    } 
    else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        is_dbl_uvw = 0;
        is_dbl_vis = 0;
        sdp_mem_check_and_view(uvws, &uvws_flt, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);
        kernel_name = "sdp_gridder_wtower_degrid"
                      "<complex<float>, float, complex<float> >";
    } 
    else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types: "
                      "subgrids has type %s; uvws has type %s; vis has type %s",
                      sdp_mem_type_name(sdp_mem_type(subgrids)),
                      sdp_mem_type_name(sdp_mem_type(uvws)),
                      sdp_mem_type_name(sdp_mem_type(vis)));
    }
    const void *arg[] = {
        sdp_mem_gpu_buffer_const(subgrids, status),
        &w_plane,
        &subgrid_offset_u,
        &subgrid_offset_v,
        &subgrid_offset_w,
        (const void *)&freq0_hz,
        (const void *)&dfreq_hz,
        (const void *)&start_row,
        (const void *)&end_row,
        is_dbl_uvw ? (const void *)&uvws_dbl : (const void *)&uvws_flt,
        (const void *)&start_chs_,
        (const void *)&end_chs_,
        sdp_mem_gpu_buffer_const(plan->uv_kernel_gpu, status),
        sdp_mem_gpu_buffer_const(plan->w_kernel_gpu, status),
        &plan->subgrid_size,
        &plan->support,
        &plan->w_support,
        &plan->oversampling,
        &plan->w_oversampling,
        (const void *)&plan->theta,
        (const void *)&plan->w_step,
        is_dbl_vis ? (const void *)&vis_dbl : (const void *)&vis_flt};
        num_blocks[0] = (num_rows + num_threads[0] - 1) / num_threads[0];

    sdp_launch_cuda_kernel(kernel_name, num_blocks, num_threads, 0, 0, arg, status);
}

// Local function to do the gridding.
template <typename SUBGRID_TYPE, typename UVW_TYPE, typename VIS_TYPE>
void grid(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids, int w_plane, 
          int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w, 
          double freq0_hz, double dfreq_hz, int64_t start_row, int64_t end_row, 
          const sdp_Mem *uvws, const sdp_MemViewCpu<const int, 1> &start_chs, 
          const sdp_MemViewCpu<const int, 1> &end_chs, const sdp_Mem *vis, sdp_Error *status) {

    if (*status) return;

    sdp_MemViewCpu<SUBGRID_TYPE, 3> subgrids_;
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<const VIS_TYPE, 2> vis_;
    sdp_mem_check_and_view(subgrids, &subgrids_, status);
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(vis, &vis_, status);

    const double *RESTRICT uv_kernel = (const double *)sdp_mem_data_const(plan->uv_kernel);
    const double *RESTRICT w_kernel = (const double *)sdp_mem_data_const(plan->w_kernel);

    const int half_subgrid = plan->subgrid_size / 2;
    const int oversample = plan->oversampling;
    const int w_oversample = plan->w_oversampling;
    const int support = plan->support;
    const int w_support = plan->w_support;
    const double theta = plan->theta;
    const double w_step = plan->w_step;
    const double theta_ov = theta * oversample;
    const double w_step_ov = 1.0 / w_step * w_oversample;
    const int half_sg_size_ov = (half_subgrid - support / 2 + 1) * oversample;

    // Loop over rows. Each row contains visibilities for all channels.
    for (int64_t i_row = start_row; i_row < end_row; ++i_row) {
        // Skip if there's no visibility to grid.
        int64_t start_ch = start_chs(i_row), end_ch = end_chs(i_row);
        if (start_ch >= end_ch) continue;

        // Select only visibilities on this w-plane.
        const UVW_TYPE uvw[] = {uvws_(i_row, 0), uvws_(i_row, 1), uvws_(i_row, 2)};
        const double min_w = (w_plane + subgrid_offset_w - 1) * w_step;
        const double max_w = (w_plane + subgrid_offset_w) * w_step;

        sdp_gridder_clamp_channels_inline(uvw[2], freq0_hz, dfreq_hz, &start_ch, &end_ch, min_w, max_w);
        if (start_ch >= end_ch) continue;

        // Scale + shift UVWs.
        const double s_uvw0 = freq0_hz / C_0, s_duvw = dfreq_hz / C_0;
        double uvw0[] = {uvw[0] * s_uvw0, uvw[1] * s_uvw0, uvw[2] * s_uvw0};
        double duvw[] = {uvw[0] * s_duvw, uvw[1] * s_duvw, uvw[2] * s_duvw};
        uvw0[0] -= subgrid_offset_u / theta;
        uvw0[1] -= subgrid_offset_v / theta;
        uvw0[2] -= ((subgrid_offset_w + w_plane - 1) * w_step);

        // Bounds check.
        const double u_min = floor(theta * (uvw0[0] + start_ch * duvw[0]));
        const double u_max = ceil(theta * (uvw0[0] + (end_ch - 1) * duvw[0]));
        const double v_min = floor(theta * (uvw0[1] + start_ch * duvw[1]));
        const double v_max = ceil(theta * (uvw0[1] + (end_ch - 1) * duvw[1]));

        if (u_min < -half_subgrid || u_max >= half_subgrid || 
            v_min < -half_subgrid || v_max >= half_subgrid) {
            continue;
        }

        // Loop over selected channels.
        for (int64_t c = start_ch; c < end_ch; c++) {
            const double u = uvw0[0] + c * duvw[0];
            const double v = uvw0[1] + c * duvw[1];
            const double w = uvw0[2] + c * duvw[2];

            // Determine top-left corner of grid region centred approximately
            // on visibility in oversampled coordinates (this is subgrid-local,
            // so should be safe for overflows).
            const int iu0_ov = int(round(u * theta_ov)) + half_sg_size_ov;
            const int iv0_ov = int(round(v * theta_ov)) + half_sg_size_ov;
            const int iw0_ov = int(round(w * w_step_ov));
            const int iu0 = iu0_ov / oversample;
            const int iv0 = iv0_ov / oversample;

            // Determine which kernel to use.
            // Comment from Peter:
            // For future reference - at least on CPU the memory latency for
            // accessing the kernel is the main bottleneck of (de)gridding.
            // This can be mitigated quite well by pre-fetching the next kernel
            // value before starting to (de)grid the current one.
            const int u_off = (iu0_ov % oversample) * support;
            const int v_off = (iv0_ov % oversample) * support;
            const int w_off = (iw0_ov % w_oversample) * w_support;

            // Grid visibility.
            const SUBGRID_TYPE local_vis = (SUBGRID_TYPE)vis_(i_row, c);

            #if defined(AVX512)
            for (int iw = 0; iw < w_support; ++iw) {
                #ifdef PREFETCH
                if(iw + 1 < w_support) {
                    _mm_prefetch(&w_kernel[w_off + iw + 1], _MM_HINT_T0);
                }
                #endif // PREFETCH

                const __m512d w_kernel_val = _mm512_set1_pd(w_kernel[w_off + iw]);
                const __m512d local_vis_w_real = _mm512_set1_pd(local_vis.real() * w_kernel[w_off + iw]);
                const __m512d local_vis_w_imag = _mm512_set1_pd(local_vis.imag() * w_kernel[w_off + iw]);

                const SUBGRID_TYPE local_vis_w = ((SUBGRID_TYPE)w_kernel[w_off + iw] * local_vis);

                for (int iu = 0; iu < support; ++iu) {
                    #ifdef PREFETCH
                    if(iu + 1 < support) {
                        _mm_prefetch(&uv_kernel[u_off + iu + 1], _MM_HINT_T0);
                    }
                    #endif // PREFETCH
                    
                    const int ix_u = iu0 + iu;
                    const __m512d u_kernel_val = _mm512_set1_pd(uv_kernel[u_off + iu]);
                    const __m512d local_vis_u_real = _mm512_mul_pd(u_kernel_val, local_vis_w_real);
                    const __m512d local_vis_u_imag = _mm512_mul_pd(u_kernel_val, local_vis_w_imag);

                    // Process 8 v elements at once using AVX-512
                    for (int iv = 0; iv < support; iv += 8) {
                        #ifdef PREFETCH
                        if(iv + 8 < support) {
                            _mm_prefetch(&uv_kernel[v_off + iv + 8], _MM_HINT_T0);
                            _mm_prefetch(&subgrids_(iw, ix_u, iv0 + iv + 8), _MM_HINT_T0);
                        }
                        #endif // PREFETCH
                        
                        const int ix_v = iv0 + iv;
                        __m512d v_kernel_vals = _mm512_load_pd(&uv_kernel[v_off + iv]);
                        __m512d subgrid_real = _mm512_setr_pd(
                            subgrids_(iw, ix_u, ix_v).real(),
                            subgrids_(iw, ix_u, ix_v + 1).real(),
                            subgrids_(iw, ix_u, ix_v + 2).real(),
                            subgrids_(iw, ix_u, ix_v + 3).real(),
                            subgrids_(iw, ix_u, ix_v + 4).real(),
                            subgrids_(iw, ix_u, ix_v + 5).real(),
                            subgrids_(iw, ix_u, ix_v + 6).real(),
                            subgrids_(iw, ix_u, ix_v + 7).real()
                        );
                        __m512d subgrid_imag = _mm512_setr_pd(
                            subgrids_(iw, ix_u, ix_v).imag(),
                            subgrids_(iw, ix_u, ix_v + 1).imag(),
                            subgrids_(iw, ix_u, ix_v + 2).imag(),
                            subgrids_(iw, ix_u, ix_v + 3).imag(),
                            subgrids_(iw, ix_u, ix_v + 4).imag(),
                            subgrids_(iw, ix_u, ix_v + 5).imag(),
                            subgrids_(iw, ix_u, ix_v + 6).imag(),
                            subgrids_(iw, ix_u, ix_v + 7).imag()
                        );

                        __m512d update_real = _mm512_mul_pd(v_kernel_vals, local_vis_u_real);
                        __m512d update_imag = _mm512_mul_pd(v_kernel_vals, local_vis_u_imag);

                        subgrid_real = _mm512_add_pd(subgrid_real, update_real);
                        subgrid_imag = _mm512_add_pd(subgrid_imag, update_imag);

                        for (int k = 0; k < 8; ++k) {
                            subgrids_(iw, ix_u, ix_v + k) = std::complex<double>(((double*)&subgrid_real)[k], ((double*)&subgrid_imag)[k]);
                        }
                    }
                
                    // Handle remaining elements, note that if the support size is less than avx vector size, this will be wrong!
                    const SUBGRID_TYPE local_vis_u = ((SUBGRID_TYPE)uv_kernel[u_off + iu] * local_vis_w);
                    for (int iv = (support/8)*8; iv < support; ++iv) {
                        const int ix_v = iv0 + iv;
                        subgrids_(iw, ix_u, ix_v) += ((SUBGRID_TYPE)uv_kernel[v_off + iv] * local_vis_u);
                    }
                }
            }

            #elif defined(AVX2)
            for (int iw = 0; iw < w_support; ++iw) {
                #ifdef PREFETCH
                if(iw + 1 < w_support) {
                    _mm_prefetch(&w_kernel[w_off + iw + 1], _MM_HINT_T0);
                }
                #endif // PREFETCH

                const __m256d w_kernel_val = _mm256_set1_pd(w_kernel[w_off + iw]);
                const __m256d local_vis_w_real = _mm256_set1_pd(local_vis.real() * w_kernel[w_off + iw]);
                const __m256d local_vis_w_imag = _mm256_set1_pd(local_vis.imag() * w_kernel[w_off + iw]);

                const SUBGRID_TYPE local_vis_w = ((SUBGRID_TYPE)w_kernel[w_off + iw] * local_vis);

                for (int iu = 0; iu < support; ++iu) {
                    #ifdef PREFETCH
                    if(iu + 1 < support) {
                        _mm_prefetch(&uv_kernel[u_off + iu + 1], _MM_HINT_T0);
                    }
                    #endif // PREFETCH

                    const int ix_u = iu0 + iu;
                    const __m256d u_kernel_val = _mm256_set1_pd(uv_kernel[u_off + iu]);
                    const __m256d local_vis_u_real = _mm256_mul_pd(u_kernel_val, local_vis_w_real);
                    const __m256d local_vis_u_imag = _mm256_mul_pd(u_kernel_val, local_vis_w_imag);

                    // Process 4 elements at a time with AVX2
                    for (int iv = 0; iv < support; iv += 4) {
                        #ifdef PREFETCH
                        if(iv + 4 < support) {
                            _mm_prefetch(&uv_kernel[v_off + iv + 4], _MM_HINT_T0);
                            _mm_prefetch(&subgrids_(iw, ix_u, iv0 + iv + 4), _MM_HINT_T0);
                        }
                        #endif // PREFETCH
                        
                        const int ix_v = iv0 + iv;
                        __m256d v_kernel_vals = _mm256_load_pd(&uv_kernel[v_off + iv]);
                        __m256d subgrid_real = _mm256_setr_pd(
                            subgrids_(iw, ix_u, ix_v).real(),
                            subgrids_(iw, ix_u, ix_v + 1).real(),
                            subgrids_(iw, ix_u, ix_v + 2).real(),
                            subgrids_(iw, ix_u, ix_v + 3).real()
                        );
                        __m256d subgrid_imag = _mm256_setr_pd(
                            subgrids_(iw, ix_u, ix_v).imag(),
                            subgrids_(iw, ix_u, ix_v + 1).imag(),
                            subgrids_(iw, ix_u, ix_v + 2).imag(),
                            subgrids_(iw, ix_u, ix_v + 3).imag()
                        );

                        __m256d update_real = _mm256_mul_pd(v_kernel_vals, local_vis_u_real);
                        __m256d update_imag = _mm256_mul_pd(v_kernel_vals, local_vis_u_imag);

                        subgrid_real = _mm256_add_pd(subgrid_real, update_real);
                        subgrid_imag = _mm256_add_pd(subgrid_imag, update_imag);

                        for (int k = 0; k < 4; ++k) {
                            subgrids_(iw, ix_u, ix_v + k) = std::complex<double>(((double*)&subgrid_real)[k], ((double*)&subgrid_imag)[k]);
                        }
                    }
                    
                    // Handle remaining elements, note that if the support size is less than avx vector size, this will be wrong!
                    const SUBGRID_TYPE local_vis_u = ((SUBGRID_TYPE)uv_kernel[u_off + iu] * local_vis_w);
                    for (int iv = (support/4)*4; iv < support; ++iv) {
                        const int ix_v = iv0 + iv;
                        subgrids_(iw, ix_u, ix_v) += ((SUBGRID_TYPE)uv_kernel[v_off + iv] * local_vis_u);
                    }
                }
            }
            #else // AVX512 || AVX2
            for (int iw = 0; iw < w_support; ++iw) {
                const SUBGRID_TYPE local_vis_w = ((SUBGRID_TYPE)w_kernel[w_off + iw] * local_vis);
#pragma GCC ivdep
#pragma GCC unroll(8)
                for (int iu = 0; iu < support; ++iu) {
                    const SUBGRID_TYPE local_vis_u = ((SUBGRID_TYPE)uv_kernel[u_off + iu] * local_vis_w);
                    const int ix_u = iu0 + iu;
#pragma GCC ivdep
#pragma GCC unroll(8)
                    for (int iv = 0; iv < support; ++iv) {
                        const int ix_v = iv0 + iv;
                        subgrids_(iw, ix_u, ix_v) += ((SUBGRID_TYPE)uv_kernel[v_off + iv] * local_vis_u);
                    }
                }
            }
            #endif // AVX512 || AVX2
        }
    }
}

// Local function to call the CPU gridding kernel.
void grid_cpu(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids, int w_plane,
              int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
              double freq0_hz, double dfreq_hz, int64_t start_row, int64_t end_row, 
              const sdp_Mem *uvws, const sdp_Mem *start_chs, const sdp_Mem *end_chs, 
              const sdp_Mem *vis, sdp_Error *status) {
    if (*status) return;

    sdp_MemViewCpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);

    if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE &&
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {

        grid<complex<double>, double, complex<double>>(plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w, 
                                                       freq0_hz, dfreq_hz, start_row, end_row, uvws, start_chs_, end_chs_, vis, status);
    } 
    else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {

        grid<complex<float>, double, complex<float>>(plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w, 
                                                     freq0_hz, dfreq_hz, start_row, end_row, uvws, start_chs_, end_chs_, vis, status);
    } 
    else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT &&
               sdp_mem_type(uvws) == SDP_MEM_FLOAT &&
               sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {

        grid<complex<float>, float, complex<float>>(plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w, 
                                                    freq0_hz, dfreq_hz, start_row, end_row, uvws, start_chs_, end_chs_, vis, status);
    } 
    else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types: "
                      "subgrids has type %s; uvws has type %s; vis has type %s",
                      sdp_mem_type_name(sdp_mem_type(subgrids)),
                      sdp_mem_type_name(sdp_mem_type(uvws)),
                      sdp_mem_type_name(sdp_mem_type(vis)));
    }
}

// Local function to call the GPU gridding kernel.
void grid_gpu(const sdp_GridderWtowerUVW *plan, sdp_Mem *subgrids, int w_plane,
              int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
              double freq0_hz, double dfreq_hz, int64_t start_row,
              int64_t end_row, const sdp_Mem *uvws, const sdp_Mem *start_chs,
              const sdp_Mem *end_chs, const sdp_Mem *vis, sdp_Error *status) {
    if (*status) return;

    uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
    const char *kernel_name = 0;
    int is_dbl_uvw = 0;
    int is_dbl_vis = 0;
    const int64_t num_rows = sdp_mem_shape_dim(vis, 0);
    sdp_MemViewGpu<const double, 2> uvws_dbl;
    sdp_MemViewGpu<const float, 2> uvws_flt;
    sdp_MemViewGpu<const complex<double>, 2> vis_dbl;
    sdp_MemViewGpu<const complex<float>, 2> vis_flt;
    sdp_MemViewGpu<const int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(start_chs, &start_chs_, status);
    sdp_mem_check_and_view(end_chs, &end_chs_, status);

    if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_DOUBLE &&
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE &&
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_DOUBLE) {
        is_dbl_uvw = 1;
        is_dbl_vis = 1;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_dbl, status);

        kernel_name = "sdp_gridder_wtower_grid" "<double, double, complex<double> >";
    } 
    else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT && 
        sdp_mem_type(uvws) == SDP_MEM_DOUBLE && 
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        is_dbl_uvw = 1;
        is_dbl_vis = 0;
        sdp_mem_check_and_view(uvws, &uvws_dbl, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);

        kernel_name = "sdp_gridder_wtower_grid" "<float, double, complex<float> >";
    } 
    else if (sdp_mem_type(subgrids) == SDP_MEM_COMPLEX_FLOAT && 
        sdp_mem_type(uvws) == SDP_MEM_FLOAT && 
        sdp_mem_type(vis) == SDP_MEM_COMPLEX_FLOAT) {
        is_dbl_uvw = 0;
        is_dbl_vis = 0;
        sdp_mem_check_and_view(uvws, &uvws_flt, status);
        sdp_mem_check_and_view(vis, &vis_flt, status);

        kernel_name = "sdp_gridder_wtower_grid" "<float, float, complex<float> >";
    } 
    else {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Unsupported data types: "
                      "subgrids has type %s; uvws has type %s; vis has type %s",
                      sdp_mem_type_name(sdp_mem_type(subgrids)),
                      sdp_mem_type_name(sdp_mem_type(uvws)),
                      sdp_mem_type_name(sdp_mem_type(vis)));
    }
    const void *arg[] = {
        sdp_mem_gpu_buffer(subgrids, status),
        &w_plane,
        &subgrid_offset_u,
        &subgrid_offset_v,
        &subgrid_offset_w,
        (const void *)&freq0_hz,
        (const void *)&dfreq_hz,
        (const void *)&start_row,
        (const void *)&end_row,
        is_dbl_uvw ? (const void *)&uvws_dbl : (const void *)&uvws_flt,
        (const void *)&start_chs_,
        (const void *)&end_chs_,
        sdp_mem_gpu_buffer_const(plan->uv_kernel_gpu, status),
        sdp_mem_gpu_buffer_const(plan->w_kernel_gpu, status),
        &plan->subgrid_size,
        &plan->support,
        &plan->w_support,
        &plan->oversampling,
        &plan->w_oversampling,
        (const void *)&plan->theta,
        (const void *)&plan->w_step,
        is_dbl_vis ? (const void *)&vis_dbl : (const void *)&vis_flt};
    num_blocks[0] = (num_rows + num_threads[0] - 1) / num_threads[0];

    sdp_launch_cuda_kernel(kernel_name, num_blocks, num_threads, 0, 0, arg, status);
}

} // namespace

sdp_GridderWtowerUVW *
sdp_gridder_wtower_uvw_create(int image_size, int subgrid_size, double theta,
                              double w_step, double shear_u, double shear_v,
                              int support, int oversampling, int w_support,
                              int w_oversampling, sdp_Error *status) {
    if (subgrid_size % 2 != 0) {
        // If subgrid_size isn't even, the current FFT shift won't be correct.
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Subgrid size must be even (value given was %d).", subgrid_size);
        return NULL;
    }
    sdp_GridderWtowerUVW *plan = (sdp_GridderWtowerUVW *)calloc(1, sizeof(sdp_GridderWtowerUVW));
    plan->image_size = image_size;
    plan->subgrid_size = subgrid_size;
    plan->theta = theta;
    plan->w_step = w_step;
    plan->shear_u = shear_u;
    plan->shear_v = shear_v;
    plan->support = support;
    plan->oversampling = oversampling;
    plan->w_support = w_support;
    plan->w_oversampling = w_oversampling;

    // Generate oversampled convolution kernel (uv_kernel).
    const int64_t uv_conv_shape[] = {plan->oversampling + 1, plan->support};
    plan->uv_kernel = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, uv_conv_shape, status);
    sdp_gridder_make_pswf_kernel(plan->support, plan->uv_kernel, status);

    // Generate oversampled w convolution kernel (w_kernel).
    const int64_t w_conv_shape[] = {plan->w_oversampling + 1, plan->w_support};
    plan->w_kernel = sdp_mem_create(SDP_MEM_DOUBLE, SDP_MEM_CPU, 2, w_conv_shape, status);
    sdp_gridder_make_pswf_kernel(plan->w_support, plan->w_kernel, status);

    // Generate w_pattern.
    // This is the iDFT of a sole visibility at (0, 0, w) - our plan is roughly
    // to convolve in uvw space by a delta function to move the grid in w.
    const int64_t w_pattern_shape[] = {subgrid_size, subgrid_size};
    plan->w_pattern = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 2, w_pattern_shape, status);
    sdp_gridder_make_w_pattern(subgrid_size, theta, shear_u, shear_v, w_step, plan->w_pattern, status);

    return plan;
}

void sdp_gridder_wtower_uvw_grid(sdp_GridderWtowerUVW *plan, const sdp_Mem *vis, const sdp_Mem *uvws, 
                                 const sdp_Mem *start_chs, const sdp_Mem *end_chs, double freq0_hz, double dfreq_hz, 
                                 sdp_Mem *subgrid_image, int subgrid_offset_u, int subgrid_offset_v,
                                 int subgrid_offset_w, int64_t start_row, int64_t end_row, sdp_Error *status) {
    if (*status) return;

    if (dfreq_hz == 0.0) dfreq_hz = 10; // Prevent possible divide-by-zero.
    
    const sdp_MemLocation loc = sdp_mem_location(vis);
    if (sdp_mem_location(subgrid_image) != loc || sdp_mem_location(uvws) != loc  || 
        sdp_mem_location(start_chs) != loc || sdp_mem_location(end_chs) != loc) {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        return;
    }

    if (start_row < 0 || end_row < 0) {
        start_row = 0;
        end_row = sdp_mem_shape_dim(uvws, 0);
    }

    // Copy internal arrays to GPU memory as required.
    sdp_Mem *w_pattern_ptr = plan->w_pattern;
    if (loc == SDP_MEM_GPU) {
        if (!plan->w_pattern_gpu) {
            plan->w_kernel_gpu = sdp_mem_create_copy(plan->w_kernel, loc, status);
            plan->uv_kernel_gpu = sdp_mem_create_copy(plan->uv_kernel, loc, status);
            plan->w_pattern_gpu = sdp_mem_create_copy(plan->w_pattern, loc, status);
        }
        w_pattern_ptr = plan->w_pattern_gpu;
    }

    // Determine w-range.
    double c_min[] = {0, 0, 0}, c_max[] = {0, 0, 0};
    sdp_gridder_uvw_bounds_all(uvws, freq0_hz, dfreq_hz, start_chs, end_chs, c_min, c_max, status);

    // Get subgrid at first w-plane.
    const double eta = 1e-5;
    const int first_w_plane = (int)floor(c_min[2] / plan->w_step - eta) - subgrid_offset_w;
    const int last_w_plane = (int)ceil(c_max[2] / plan->w_step + eta) - subgrid_offset_w + 1;

    // Create subgrid image and subgrids to accumulate on.
    const int64_t num_elements_sg = plan->subgrid_size * plan->subgrid_size;
    const int64_t subgrid_shape[] = {plan->subgrid_size, plan->subgrid_size};
    const int64_t subgrids_shape[] = {plan->w_support, plan->subgrid_size, plan->subgrid_size};

    sdp_Mem *w_subgrid_image = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, loc, 2, subgrid_shape, status);
    sdp_Mem *subgrids = sdp_mem_create(sdp_mem_type(vis), loc, 3, subgrids_shape, status);
    sdp_mem_set_value(subgrids, 0, status);
    sdp_mem_set_value(w_subgrid_image, 0, status);

    // Create the iFFT plan and scratch buffer.
    sdp_Mem *fft_buffer = sdp_mem_create(sdp_mem_type(subgrids), loc, 2, subgrid_shape, status);
    sdp_Fft *fft = sdp_fft_create(fft_buffer, fft_buffer, 2, false, status);

    // Loop over w-planes.
    const int num_w_planes = 1 + last_w_plane - first_w_plane;
    for (int w_plane = first_w_plane; w_plane <= last_w_plane; ++w_plane) {
        if (*status) break;

        // Move to next w-plane.
        if (w_plane != first_w_plane) {
            // Accumulate zero-th subgrid, shift, clear upper subgrid.
            // w_subgrid_image /= plan->w_pattern
            sdp_gridder_scale_inv_array(w_subgrid_image, w_subgrid_image, w_pattern_ptr, 1, status);

            // w_subgrid_image += ifft(subgrids[0])
            sdp_mem_copy_contents(fft_buffer, subgrids, 0, 0, num_elements_sg, status);
            sdp_fft_phase(fft_buffer, status);
            sdp_fft_exec(fft, fft_buffer, fft_buffer, status);
            sdp_fft_phase(fft_buffer, status);
            sdp_gridder_accumulate_scaled_arrays(w_subgrid_image, fft_buffer, 0, 0.0, status);

            // subgrids[:-1] = subgrids[1:]
            sdp_gridder_shift_subgrids(subgrids, status);

            // subgrids[-1] = 0
            sdp_mem_clear_portion(subgrids, num_elements_sg * (plan->w_support - 1), num_elements_sg, status);
        }

        if (loc == SDP_MEM_CPU) {
            grid_cpu(plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w, 
                     freq0_hz, dfreq_hz, start_row, end_row, uvws, start_chs, end_chs, vis, status);
        } 
        else if (loc == SDP_MEM_GPU) {
            grid_gpu(plan, subgrids, w_plane, subgrid_offset_u, subgrid_offset_v, subgrid_offset_w, 
                     freq0_hz, dfreq_hz, start_row, end_row, uvws, start_chs, end_chs, vis, status);
        }
    }

    // Accumulate remaining data from subgrids.
    for (int i = 0; i < plan->w_support; ++i) {
        // Perform w_subgrid_image /= plan->w_pattern
        sdp_gridder_scale_inv_array(w_subgrid_image, w_subgrid_image, w_pattern_ptr, 1, status);

        // Perform w_subgrid_image += ifft(subgrids[i])
        sdp_mem_copy_contents(fft_buffer, subgrids, 0, num_elements_sg * i, num_elements_sg, status);
        sdp_fft_phase(fft_buffer, status);
        sdp_fft_exec(fft, fft_buffer, fft_buffer, status);
        sdp_fft_phase(fft_buffer, status);
        sdp_gridder_accumulate_scaled_arrays(w_subgrid_image, fft_buffer, 0, 0.0, status);
    }

    // Return updated subgrid image.
    // Perform subgrid_image += (
    //     w_subgrid_image
    //     * plan->w_pattern ** (last_w_plane + plan->w_support // 2 - 1)
    //     * plan->subgrid_size**2
    // )
    // We don't need to multiply by subgrid_size**2,
    // because the iFFT output is already scaled by this.
    int exponent = last_w_plane + plan->w_support / 2 - 1;
    sdp_gridder_accumulate_scaled_arrays(subgrid_image, w_subgrid_image, w_pattern_ptr, exponent, status);

// Update w-plane counter.
#pragma omp atomic
    plan->num_w_planes[1] += num_w_planes;

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(fft_buffer);
    sdp_fft_free(fft);
}

void sdp_gridder_wtower_uvw_degrid(
    sdp_GridderWtowerUVW *plan, const sdp_Mem *subgrid_image,
    int subgrid_offset_u, int subgrid_offset_v, int subgrid_offset_w,
    double freq0_hz, double dfreq_hz, const sdp_Mem *uvws,
    const sdp_Mem *start_chs, const sdp_Mem *end_chs, sdp_Mem *vis,
    int64_t start_row, int64_t end_row, sdp_Error *status) {
    if (*status) return;

    if (dfreq_hz == 0.0) dfreq_hz = 10; // Prevent possible divide-by-zero.
    const sdp_MemLocation loc = sdp_mem_location(vis);
    if (sdp_mem_location(subgrid_image) != loc || 
        sdp_mem_location(uvws) != loc || 
        sdp_mem_location(start_chs) != loc ||
        sdp_mem_location(end_chs) != loc) {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("All arrays must be in the same memory space");
        return;
    }
    if (start_row < 0 || end_row < 0) {
        start_row = 0;
        end_row = sdp_mem_shape_dim(uvws, 0);
    }

    // Copy internal arrays to GPU memory as required.
    sdp_Mem *w_pattern_ptr = plan->w_pattern;
    if (loc == SDP_MEM_GPU) {
        if (!plan->w_pattern_gpu) {
            plan->w_kernel_gpu = sdp_mem_create_copy(plan->w_kernel, loc, status);
            plan->uv_kernel_gpu = sdp_mem_create_copy(plan->uv_kernel, loc, status);
            plan->w_pattern_gpu = sdp_mem_create_copy(plan->w_pattern, loc, status);
        }
        w_pattern_ptr = plan->w_pattern_gpu;
    }

    // Determine w-range.
    double c_min[] = {0, 0, 0}, c_max[] = {0, 0, 0};
    sdp_gridder_uvw_bounds_all(uvws, freq0_hz, dfreq_hz, start_chs, end_chs, c_min, c_max, status);

    // Get subgrid at first w-plane.
    const double eta = 1e-5;
    const int first_w_plane = (int)floor(c_min[2] / plan->w_step - eta) - subgrid_offset_w;
    const int last_w_plane = (int)ceil(c_max[2] / plan->w_step + eta) - subgrid_offset_w + 1;

    // First w-plane we need to generate is (support / 2) below the first one
    // with visibilities.
    // Comment from Peter: That is actually a bit of a simplifying
    // assumption I made, and might well overshoot.
    // TODO Might need to check this properly.
    const int64_t subgrid_shape[] = {plan->subgrid_size, plan->subgrid_size};
    sdp_Mem *w_subgrid_image = sdp_mem_create(sdp_mem_type(vis), loc, 2, subgrid_shape, status);

    // Perform w_subgrid_image = subgrid_image /
    //             plan->w_pattern ** (first_w_plane - plan->w_support // 2)
    const int exponent = first_w_plane - plan->w_support / 2;
    sdp_gridder_scale_inv_array(w_subgrid_image, subgrid_image, w_pattern_ptr, exponent, status);

    // Create the FFT plan.
    sdp_Fft *fft = sdp_fft_create(w_subgrid_image, w_subgrid_image, 2, true, status);

    // Create sub-grid stack, with size (w_support, subgrid_size, subgrid_size).
    const int64_t num_elements_sg = plan->subgrid_size * plan->subgrid_size;
    const int64_t subgrids_shape[] = {plan->w_support, plan->subgrid_size, plan->subgrid_size};
    sdp_Mem *subgrids = sdp_mem_create(sdp_mem_type(vis), loc, 3, subgrids_shape, status);

    // Get a pointer to the last subgrid in the stack.
    const int64_t slice_offsets[] = {plan->w_support - 1, 0, 0};
    sdp_Mem *last_subgrid_ptr = sdp_mem_create_wrapper_for_slice(subgrids, slice_offsets, 2, subgrid_shape, status);

    // Fill sub-grid stack.
    for (int i = 0; i < plan->w_support; ++i) {
        // Perform subgrids[i] = fft(w_subgrid_image)
        // Copy w_subgrid_image to current sub-grid, and then do FFT in-place.
        const int64_t slice_offsets[] = {i, 0, 0};
        sdp_Mem *current_subgrid_ptr = sdp_mem_create_wrapper_for_slice(subgrids, slice_offsets, 2, subgrid_shape, status);
        sdp_mem_copy_contents(current_subgrid_ptr, w_subgrid_image, 0, 0, num_elements_sg, status);
        sdp_fft_phase(current_subgrid_ptr, status);
        sdp_fft_exec(fft, current_subgrid_ptr, current_subgrid_ptr, status);
        sdp_fft_phase(current_subgrid_ptr, status);
        sdp_mem_free(current_subgrid_ptr);

        // Perform w_subgrid_image /= plan->w_pattern
        sdp_gridder_scale_inv_array(w_subgrid_image, w_subgrid_image, w_pattern_ptr, 1, status);
    }

    // Loop over w-planes.
    const int num_w_planes = 1 + last_w_plane - first_w_plane;
    for (int w_plane = first_w_plane; w_plane <= last_w_plane; ++w_plane) {
        if (*status) break;

        // Move to next w-plane.
        if (w_plane != first_w_plane) {
            // Shift subgrids, add new w-plane.
            // subgrids[:-1] = subgrids[1:]
            sdp_gridder_shift_subgrids(subgrids, status);

            // subgrids[-1] = fft(w_subgrid_image)
            // Copy w_subgrid_image to last subgrid, and then do FFT in-place.
            sdp_mem_copy_contents(last_subgrid_ptr, w_subgrid_image, 0, 0, num_elements_sg, status);
            sdp_fft_phase(last_subgrid_ptr, status);
            sdp_fft_exec(fft, last_subgrid_ptr, last_subgrid_ptr, status);
            sdp_fft_phase(last_subgrid_ptr, status);

            // w_subgrid_image /= plan->w_pattern
            sdp_gridder_scale_inv_array(w_subgrid_image, w_subgrid_image, w_pattern_ptr, 1, status);
        }

        if (loc == SDP_MEM_CPU) {
            degrid_cpu(plan, subgrids, w_plane, subgrid_offset_u,
                       subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                       start_row, end_row, uvws, start_chs, end_chs, vis,
                       status);
        } 
        else if (loc == SDP_MEM_GPU) {
            degrid_gpu(plan, subgrids, w_plane, subgrid_offset_u,
                       subgrid_offset_v, subgrid_offset_w, freq0_hz, dfreq_hz,
                       start_row, end_row, uvws, start_chs, end_chs, vis,
                       status);
        }
    }

// Update w-plane counter.
#pragma omp atomic
    plan->num_w_planes[0] += num_w_planes;

    sdp_mem_free(w_subgrid_image);
    sdp_mem_free(subgrids);
    sdp_mem_free(last_subgrid_ptr);
    sdp_fft_free(fft);
}

void sdp_gridder_wtower_uvw_degrid_correct(sdp_GridderWtowerUVW *plan,
                                           sdp_Mem *facet, int facet_offset_l,
                                           int facet_offset_m, int w_offset,
                                           sdp_Error *status) {
    sdp_gridder_grid_correct_pswf(plan->image_size, plan->theta, plan->w_step,
                                  plan->shear_u, plan->shear_v, plan->support,
                                  plan->w_support, facet, facet_offset_l,
                                  facet_offset_m, status);
    if (sdp_mem_is_complex(facet)) {
        sdp_gridder_grid_correct_w_stack(plan->image_size, plan->theta, plan->w_step, 
                                         plan->shear_u, plan->shear_v, facet, facet_offset_l, 
                                         facet_offset_m, w_offset, false, status);
    }
}

void sdp_gridder_wtower_uvw_grid_correct(sdp_GridderWtowerUVW *plan,
                                         sdp_Mem *facet, int facet_offset_l,
                                         int facet_offset_m, int w_offset,
                                         sdp_Error *status) {
    sdp_gridder_grid_correct_pswf(plan->image_size, plan->theta, plan->w_step,
                                  plan->shear_u, plan->shear_v, plan->support,
                                  plan->w_support, facet, facet_offset_l,
                                  facet_offset_m, status);

    if (sdp_mem_is_complex(facet)) {
        sdp_gridder_grid_correct_w_stack(plan->image_size, plan->theta, plan->w_step, 
                                         plan->shear_u, plan->shear_v, facet, facet_offset_l, 
                                         facet_offset_m, w_offset, true, status);
    }
}

void sdp_gridder_wtower_uvw_free(sdp_GridderWtowerUVW *plan) {
    if (!plan) return;
    sdp_mem_free(plan->uv_kernel);
    sdp_mem_free(plan->uv_kernel_gpu);
    sdp_mem_free(plan->w_kernel);
    sdp_mem_free(plan->w_kernel_gpu);
    sdp_mem_free(plan->w_pattern);
    sdp_mem_free(plan->w_pattern_gpu);
    free(plan);
}

int sdp_gridder_wtower_uvw_num_w_planes(const sdp_GridderWtowerUVW *plan, int gridding) {
    return (gridding >= 0 && gridding < 2) ? plan->num_w_planes[gridding] : 0;
}

int sdp_gridder_wtower_uvw_image_size(const sdp_GridderWtowerUVW *plan) {
    return plan->image_size;
}

int sdp_gridder_wtower_uvw_oversampling(const sdp_GridderWtowerUVW *plan) {
    return plan->oversampling;
}

double sdp_gridder_wtower_uvw_shear_u(const sdp_GridderWtowerUVW *plan) {
    return plan->shear_u;
}

double sdp_gridder_wtower_uvw_shear_v(const sdp_GridderWtowerUVW *plan) {
    return plan->shear_v;
}

int sdp_gridder_wtower_uvw_subgrid_size(const sdp_GridderWtowerUVW *plan) {
    return plan->subgrid_size;
}

int sdp_gridder_wtower_uvw_support(const sdp_GridderWtowerUVW *plan) {
    return plan->support;
}

double sdp_gridder_wtower_uvw_theta(const sdp_GridderWtowerUVW *plan) {
    return plan->theta;
}

int sdp_gridder_wtower_uvw_w_oversampling(const sdp_GridderWtowerUVW *plan) {
    return plan->w_oversampling;
}

double sdp_gridder_wtower_uvw_w_step(const sdp_GridderWtowerUVW *plan) {
    return plan->w_step;
}

int sdp_gridder_wtower_uvw_w_support(const sdp_GridderWtowerUVW *plan) {
    return plan->w_support;
}
