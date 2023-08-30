/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <complex>
#include <cstdlib>

#include "ska-sdp-func/station_beam/sdp_element_dipole.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

using std::complex;

#define C_0 299792458.0


template<typename FP>
static void sdp_dipole(
        FP kL,
        FP cos_kL,
        FP phi_rad,
        FP sin_theta,
        FP cos_theta,
        complex<FP>& e_theta,
        complex<FP>& e_phi
)
{
    const FP cos_phi = cos(phi_rad);
    const FP denominator =
            (FP)1 + cos_phi * cos_phi * (cos_theta * cos_theta - (FP)1);
    if (denominator == (FP)0)
    {
        e_theta = e_phi = complex<FP>(0, 0);
    }
    else
    {
        const FP temp = (cos(kL * cos_phi * sin_theta) - cos_kL) / denominator;
        e_theta = complex<FP>(-cos_phi * cos_theta * temp, 0);
        e_phi = complex<FP>(sin(phi_rad) * temp, 0);
    }
}


template<typename FP>
static void sdp_dipole_pattern(
        const int num_points,
        const FP* theta_rad,
        const FP* phi_rad,
        const FP kL,
        const FP cos_kL,
        const int stride,
        const int e_theta_offset,
        const int e_phi_offset,
        complex<FP>* e_theta,
        complex<FP>* e_phi
)
{
    for (int i = 0; i < num_points; ++i)
    {
        sdp_dipole(kL, cos_kL,
                phi_rad[i], (FP)sin(theta_rad[i]), (FP)cos(theta_rad[i]),
                e_theta[i * stride + e_theta_offset],
                e_phi[i * stride + e_phi_offset]
        );
    }
}


template<typename FP>
static void sdp_dipole_pattern_scalar(
        const int num_points,
        const FP* theta_rad,
        const FP* phi_rad,
        const FP kL,
        const FP cos_kL,
        const int stride,
        const int offset,
        complex<FP>* pattern
)
{
    for (int i = 0; i < num_points; ++i)
    {
        FP amp, phi_;
        const FP sin_theta = sin(theta_rad[i]);
        const FP cos_theta = cos(theta_rad[i]);
        phi_ = phi_rad[i];
        complex<FP> x_theta, x_phi, y_theta, y_phi;
        sdp_dipole(kL, cos_kL, phi_, sin_theta, cos_theta, x_theta, x_phi);
        phi_ += ((FP) M_PI / (FP)2);
        sdp_dipole(kL, cos_kL, phi_, sin_theta, cos_theta, y_theta, y_phi);
        amp = x_theta.real() * x_theta.real() + x_phi.real() * x_phi.real() +
                y_theta.real() * y_theta.real() + y_phi.real() * y_phi.real();
        amp /= (FP)2;
        amp = sqrt(amp);
        pattern[i * stride + offset] = complex<FP>(amp, 0);
    }
}


void sdp_element_beam_dipole(
        int num_points,
        const sdp_Mem* theta_rad,
        const sdp_Mem* phi_rad,
        double freq_hz,
        double dipole_length_m,
        int stride_element_beam,
        int index_offset_element_beam,
        sdp_Mem* element_beam,
        sdp_Error* status
)
{
    if (*status) return;
    const int location = sdp_mem_location(element_beam);
    const int element_beam_is_complex4 = sdp_mem_is_complex4(element_beam);
    const sdp_MemType type = sdp_mem_type(element_beam);
    const int E_theta_offset = index_offset_element_beam;
    const int E_phi_offset = index_offset_element_beam + 1;
    const double kL = dipole_length_m * (M_PI * freq_hz / C_0);
    const double cos_kL = cos(kL);
    const float kL_f = (float) kL;
    const float cos_kL_f = (float) cos_kL;
    if (sdp_mem_location(theta_rad) != location ||
            sdp_mem_location(phi_rad) != location)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Memory location mismatch");
        return;
    }
    if (!sdp_mem_is_complex(element_beam))
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Element beam array must be of complex type");
        return;
    }
    if (location == SDP_MEM_CPU)
    {
        if (element_beam_is_complex4)
        {
            if (type == SDP_MEM_COMPLEX_FLOAT)
            {
                sdp_dipole_pattern<float>(num_points,
                        (const float*)sdp_mem_data_const(theta_rad),
                        (const float*)sdp_mem_data_const(phi_rad),
                        kL_f, cos_kL_f,
                        stride_element_beam, E_theta_offset, E_phi_offset,
                        (complex<float>*)sdp_mem_data(element_beam),
                        (complex<float>*)sdp_mem_data(element_beam)
                );
            }
            else if (type == SDP_MEM_COMPLEX_DOUBLE)
            {
                sdp_dipole_pattern<double>(num_points,
                        (const double*)sdp_mem_data_const(theta_rad),
                        (const double*)sdp_mem_data_const(phi_rad),
                        kL, cos_kL,
                        stride_element_beam, E_theta_offset, E_phi_offset,
                        (complex<double>*)sdp_mem_data(element_beam),
                        (complex<double>*)sdp_mem_data(element_beam)
                );
            }
            else
            {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Unsupported data type");
            }
        }
        else // complex scalar output
        {
            if (type == SDP_MEM_COMPLEX_FLOAT)
            {
                sdp_dipole_pattern_scalar<float>(num_points,
                        (const float*)sdp_mem_data_const(theta_rad),
                        (const float*)sdp_mem_data_const(phi_rad),
                        kL_f, cos_kL_f,
                        stride_element_beam, index_offset_element_beam,
                        (complex<float>*)sdp_mem_data(element_beam)
                );
            }
            else if (type == SDP_MEM_COMPLEX_DOUBLE)
            {
                sdp_dipole_pattern_scalar<double>(num_points,
                        (const double*)sdp_mem_data_const(theta_rad),
                        (const double*)sdp_mem_data_const(phi_rad),
                        kL, cos_kL,
                        stride_element_beam, index_offset_element_beam,
                        (complex<double>*)sdp_mem_data(element_beam)
                );
            }
            else
            {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Unsupported data type");
            }
        }
    }
    else if (location == SDP_MEM_GPU)
    {
        const uint64_t num_threads[] = {256, 1, 1};
        const uint64_t num_blocks[] = {
            (num_points + num_threads[0] - 1) / num_threads[0], 1, 1
        };
        const char* kernel_name = 0;
        const int is_dbl = sdp_mem_is_double(element_beam);
        if (element_beam_is_complex4)
        {
            if (type == SDP_MEM_COMPLEX_FLOAT)
            {
                kernel_name = "sdp_dipole_pattern<float, float2>";
            }
            else if (type == SDP_MEM_COMPLEX_DOUBLE)
            {
                kernel_name = "sdp_dipole_pattern<double, double2>";
            }
            else
            {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Unsupported data type");
            }
            const void* args[] = {
                &num_points,
                sdp_mem_gpu_buffer_const(theta_rad, status),
                sdp_mem_gpu_buffer_const(phi_rad, status),
                is_dbl ? (const void*)&kL : (const void*)&kL_f,
                is_dbl ? (const void*)&cos_kL : (const void*)&cos_kL_f,
                &stride_element_beam,
                &E_theta_offset,
                &E_phi_offset,
                sdp_mem_gpu_buffer(element_beam, status),
                sdp_mem_gpu_buffer(element_beam, status)
            };
            sdp_launch_cuda_kernel(
                    kernel_name, num_blocks, num_threads, 0, 0, args, status
            );
        }
        else
        {
            if (type == SDP_MEM_COMPLEX_FLOAT)
            {
                kernel_name = "sdp_dipole_pattern_scalar<float, float2>";
            }
            else if (type == SDP_MEM_COMPLEX_DOUBLE)
            {
                kernel_name = "sdp_dipole_pattern_scalar<double, double2>";
            }
            else
            {
                *status = SDP_ERR_DATA_TYPE;
                SDP_LOG_ERROR("Unsupported data type");
            }
            const void* args[] = {
                &num_points,
                sdp_mem_gpu_buffer_const(theta_rad, status),
                sdp_mem_gpu_buffer_const(phi_rad, status),
                is_dbl ? (const void*)&kL : (const void*)&kL_f,
                is_dbl ? (const void*)&cos_kL : (const void*)&cos_kL_f,
                &stride_element_beam,
                &index_offset_element_beam,
                sdp_mem_gpu_buffer(element_beam, status)
            };
            sdp_launch_cuda_kernel(
                    kernel_name, num_blocks, num_threads, 0, 0, args, status
            );
        }
    }
}
