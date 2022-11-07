/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>

#include "ska-sdp-func/phase_rotate/sdp_phase_rotate.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#define C_0 299792458.0
#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

using std::complex;


template<typename FP>
static void rotate_uvw(
        const int64_t num,
        const double* matrix,
        const FP* uvw_in,
        FP* uvw_out
)
{
    for (int64_t i_uvw = 0; i_uvw < num; ++i_uvw)
    {
        const int64_t i_uvw3 = i_uvw * 3;
        const FP uu = uvw_in[i_uvw3 + 0];
        const FP vv = uvw_in[i_uvw3 + 1];
        const FP ww = uvw_in[i_uvw3 + 2];
        uvw_out[i_uvw3 + 0] = matrix[0] * uu + matrix[1] * vv + matrix[2] * ww;
        uvw_out[i_uvw3 + 1] = matrix[3] * uu + matrix[4] * vv + matrix[5] * ww;
        uvw_out[i_uvw3 + 2] = matrix[6] * uu + matrix[7] * vv + matrix[8] * ww;
    }
}


template<typename COORD_TYPE, typename VIS_TYPE>
static void rotate_vis(
        const int64_t num_times,
        const int64_t num_baselines,
        const int64_t num_channels,
        const int64_t num_pols,
        const double channel_start_hz,
        const double channel_step_hz,
        const double delta_l,
        const double delta_m,
        const double delta_n,
        const COORD_TYPE* uvw,
        const complex<VIS_TYPE>* vis_in,
        complex<VIS_TYPE>* vis_out
)
{
    for (int64_t i_time = 0; i_time < num_times; ++i_time)
    {
        for (int64_t i_baseline = 0; i_baseline < num_baselines; ++i_baseline)
        {
            const int64_t i_uvw = INDEX_3D(
                    num_times, num_baselines, 3,
                    i_time, i_baseline, 0
            );
            const COORD_TYPE uu = uvw[i_uvw + 0];
            const COORD_TYPE vv = uvw[i_uvw + 1];
            const COORD_TYPE ww = uvw[i_uvw + 2];
            for (int64_t i_channel = 0; i_channel < num_channels; ++i_channel)
            {
                const double inv_wavelength =
                        (channel_start_hz + i_channel * channel_step_hz) / C_0;
                const double phase = 2.0 * M_PI * inv_wavelength *
                        (uu * delta_l + vv * delta_m + ww * delta_n);
                const complex<VIS_TYPE> phasor(cos(phase), sin(phase));
                for (int64_t i_pol = 0; i_pol < num_pols; ++i_pol)
                {
                    const int64_t i_vis = INDEX_4D(
                            num_times, num_baselines, num_channels, num_pols,
                            i_time, i_baseline, i_channel, i_pol
                    );
                    vis_out[i_vis] = vis_in[i_vis] * phasor;
                }
            }
        }
    }
}


void sdp_phase_rotate_uvw(
        const sdp_SkyCoord* phase_centre_orig,
        const sdp_SkyCoord* phase_centre_new,
        const sdp_Mem* uvw_in,
        sdp_Mem* uvw_out,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(uvw_in) != 3 || sdp_mem_num_dims(uvw_out) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("(u,v,w) arrays must be 3-dimensional.");
        return;
    }
    if (sdp_mem_shape_dim(uvw_in, 2) != 3 || sdp_mem_shape_dim(uvw_out, 2) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Last dimension of (u,v,w) arrays must be of length 3.");
        return;
    }
    if (sdp_mem_location(uvw_in) != sdp_mem_location(uvw_out))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Input and output data must be co-located.");
        return;
    }
    if (sdp_mem_is_read_only(uvw_out))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output array is read-only.");
        return;
    }
    const int64_t num_times      = sdp_mem_shape_dim(uvw_in, 0);
    const int64_t num_baselines  = sdp_mem_shape_dim(uvw_in, 1);
    const int64_t num_total = num_times * num_baselines;
    if (sdp_mem_shape_dim(uvw_out, 0) != num_times ||
            sdp_mem_shape_dim(uvw_out, 1) != num_baselines)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Input and output coordinates must be the same shape.");
        return;
    }

    // Rotate by -delta_ra around v, then delta_dec around u.
    const double orig_ra_rad = sdp_sky_coord_value(phase_centre_orig, 0);
    const double orig_dec_rad = sdp_sky_coord_value(phase_centre_orig, 1);
    const double new_ra_rad = sdp_sky_coord_value(phase_centre_new, 0);
    const double new_dec_rad = sdp_sky_coord_value(phase_centre_new, 1);
    const double d_a = -(new_ra_rad - orig_ra_rad);
    const double d_d = (new_dec_rad - orig_dec_rad);
    const double sin_d_a = sin(d_a);
    const double cos_d_a = cos(d_a);
    const double sin_d_d = sin(d_d);
    const double cos_d_d = cos(d_d);
    double mat[9];
    mat[0] =  cos_d_a;           mat[1] = 0.0;     mat[2] =  sin_d_a;
    mat[3] =  sin_d_a * sin_d_d; mat[4] = cos_d_d; mat[5] = -cos_d_a * sin_d_d;
    mat[6] = -sin_d_a * cos_d_d; mat[7] = sin_d_d; mat[8] =  cos_d_a * cos_d_d;

    // Switch on location and data types.
    if (sdp_mem_location(uvw_in) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(uvw_in) == SDP_MEM_DOUBLE &&
                sdp_mem_type(uvw_out) == SDP_MEM_DOUBLE)
        {
            rotate_uvw<double>(num_total, mat,
                    (const double*)sdp_mem_data_const(uvw_in),
                    (double*)sdp_mem_data(uvw_out)
            );
        }
        else if (sdp_mem_type(uvw_in) == SDP_MEM_FLOAT &&
                sdp_mem_type(uvw_out) == SDP_MEM_FLOAT)
        {
            rotate_uvw<float>(num_total, mat,
                    (const float*)sdp_mem_data_const(uvw_in),
                    (float*)sdp_mem_data(uvw_out)
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else if (sdp_mem_location(uvw_in) == SDP_MEM_GPU)
    {
        const uint64_t num_threads[] = {256, 1, 1};
        const uint64_t num_blocks[] = {
            (num_total + num_threads[0] - 1) / num_threads[0], 1, 1
        };
        const char* kernel_name = 0;
        if (sdp_mem_type(uvw_in) == SDP_MEM_DOUBLE &&
                sdp_mem_type(uvw_out) == SDP_MEM_DOUBLE)
        {
            kernel_name = "rotate_uvw<double3>";
        }
        else if (sdp_mem_type(uvw_in) == SDP_MEM_FLOAT &&
                sdp_mem_type(uvw_out) == SDP_MEM_FLOAT)
        {
            kernel_name = "rotate_uvw<float3>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
        const void* args[] = {
            (const void*)&num_total,
            (const void*)&mat[0],
            (const void*)&mat[1],
            (const void*)&mat[2],
            (const void*)&mat[3],
            (const void*)&mat[4],
            (const void*)&mat[5],
            (const void*)&mat[6],
            (const void*)&mat[7],
            (const void*)&mat[8],
            sdp_mem_gpu_buffer_const(uvw_in, status),
            sdp_mem_gpu_buffer(uvw_out, status)
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, args, status
        );
    }
}


void sdp_phase_rotate_vis(
        const sdp_SkyCoord* phase_centre_orig,
        const sdp_SkyCoord* phase_centre_new,
        const double channel_start_hz,
        const double channel_step_hz,
        const sdp_Mem* uvw,
        const sdp_Mem* vis_in,
        sdp_Mem* vis_out,
        sdp_Error* status
)
{
    if (*status) return;
    if (sdp_mem_num_dims(uvw) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("(u,v,w) array must be 3-dimensional.");
        return;
    }
    if (sdp_mem_num_dims(vis_in) != 4 || sdp_mem_num_dims(vis_out) != 4)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Visibility data arrays must be 4-dimensional.");
        return;
    }
    if (sdp_mem_shape_dim(uvw, 2) != 3)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Last dimension of (u,v,w) array must be of length 3.");
        return;
    }
    if (sdp_mem_location(uvw) != sdp_mem_location(vis_in) ||
            sdp_mem_location(uvw) != sdp_mem_location(vis_out))
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Input and output data must be co-located.");
        return;
    }
    if (sdp_mem_is_read_only(vis_out))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output array is read-only.");
        return;
    }
    const int64_t num_times      = sdp_mem_shape_dim(vis_in, 0);
    const int64_t num_baselines  = sdp_mem_shape_dim(vis_in, 1);
    const int64_t num_channels   = sdp_mem_shape_dim(vis_in, 2);
    const int64_t num_pols       = sdp_mem_shape_dim(vis_in, 3);
    if (sdp_mem_shape_dim(vis_out, 0) != num_times ||
            sdp_mem_shape_dim(vis_out, 1) != num_baselines ||
            sdp_mem_shape_dim(vis_out, 2) != num_channels ||
            sdp_mem_shape_dim(vis_out, 3) != num_pols)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Input and output visibilities must be the same shape.");
        return;
    }

    // Convert from spherical to tangent-plane to get delta (l, m, n).
    const double orig_ra_rad = sdp_sky_coord_value(phase_centre_orig, 0);
    const double orig_dec_rad = sdp_sky_coord_value(phase_centre_orig, 1);
    const double new_ra_rad = sdp_sky_coord_value(phase_centre_new, 0);
    const double new_dec_rad = sdp_sky_coord_value(phase_centre_new, 1);
    const double d_a = -(new_ra_rad - orig_ra_rad);
    const double sin_d_a = sin(d_a);
    const double cos_d_a = cos(d_a);
    const double sin_dec0 = sin(orig_dec_rad);
    const double cos_dec0 = cos(orig_dec_rad);
    const double sin_dec  = sin(new_dec_rad);
    const double cos_dec  = cos(new_dec_rad);
    const double l1 = cos_dec * -sin_d_a;
    const double m1 = cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_d_a;
    const double n1 = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_d_a;
    const double delta_l = 0.0 - l1;
    const double delta_m = 0.0 - m1;
    const double delta_n = 1.0 - n1;

    // Switch on location and data types.
    if (sdp_mem_location(uvw) == SDP_MEM_CPU)
    {
        if (sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis_in) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(vis_out) == SDP_MEM_COMPLEX_DOUBLE)
        {
            rotate_vis<double, double>(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    channel_start_hz,
                    channel_step_hz,
                    delta_l,
                    delta_m,
                    delta_n,
                    (const double*)sdp_mem_data_const(uvw),
                    (const complex<double>*)sdp_mem_data_const(vis_in),
                    (complex<double>*)sdp_mem_data(vis_out)
            );
        }
        else if (sdp_mem_type(uvw) == SDP_MEM_FLOAT &&
                sdp_mem_type(vis_in) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(vis_out) == SDP_MEM_COMPLEX_FLOAT)
        {
            rotate_vis<float, float>(
                    num_times,
                    num_baselines,
                    num_channels,
                    num_pols,
                    channel_start_hz,
                    channel_step_hz,
                    delta_l,
                    delta_m,
                    delta_n,
                    (const float*)sdp_mem_data_const(uvw),
                    (const complex<float>*)sdp_mem_data_const(vis_in),
                    (complex<float>*)sdp_mem_data(vis_out)
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
    }
    else if (sdp_mem_location(uvw) == SDP_MEM_GPU)
    {
        const uint64_t num_threads[] = {128, 2, 2};
        const uint64_t num_blocks[] = {
            (num_baselines + num_threads[0] - 1) / num_threads[0],
            (num_channels + num_threads[1] - 1) / num_threads[1],
            (num_times + num_threads[2] - 1) / num_threads[2]
        };
        const char* kernel_name = 0;
        if (sdp_mem_type(uvw) == SDP_MEM_DOUBLE &&
                sdp_mem_type(vis_in) == SDP_MEM_COMPLEX_DOUBLE &&
                sdp_mem_type(vis_out) == SDP_MEM_COMPLEX_DOUBLE)
        {
            kernel_name = "rotate_vis<double3, double2>";
        }
        else if (sdp_mem_type(uvw) == SDP_MEM_FLOAT &&
                sdp_mem_type(vis_in) == SDP_MEM_COMPLEX_FLOAT &&
                sdp_mem_type(vis_out) == SDP_MEM_COMPLEX_FLOAT)
        {
            kernel_name = "rotate_vis<float3, float2>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            SDP_LOG_ERROR("Unsupported data type(s)");
        }
        const void* args[] = {
            (const void*)&num_times,
            (const void*)&num_baselines,
            (const void*)&num_channels,
            (const void*)&num_pols,
            (const void*)&channel_start_hz,
            (const void*)&channel_step_hz,
            (const void*)&delta_l,
            (const void*)&delta_m,
            (const void*)&delta_n,
            sdp_mem_gpu_buffer_const(uvw, status),
            sdp_mem_gpu_buffer_const(vis_in, status),
            sdp_mem_gpu_buffer(vis_out, status)
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, args, status
        );
    }
}
