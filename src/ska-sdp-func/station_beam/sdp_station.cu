/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"

template<typename FP, typename FP2>
__global__ void sdp_station_beam_dft_scalar(
        const FP wavenumber,
        const int num_in,
        const FP2* const __restrict__ weights_in,
        const FP* const __restrict__ x_in,
        const FP* const __restrict__ y_in,
        const FP* const __restrict__ z_in,
        const int idx_offset_out,
        const int num_out,
        const FP* const __restrict__ x_out,
        const FP* const __restrict__ y_out,
        const FP* const __restrict__ z_out,
        const int* const __restrict__ data_idx,
        const FP2* const __restrict__ data,
        const int idx_offset_data,
        FP2* __restrict__ output,
        const FP norm_factor,
        const int max_in_chunk
)
{
    extern __shared__ __align__(64) unsigned char my_smem[];
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    FP2 out;
    out.x = out.y = (FP) 0;
    FP xo = (FP) 0, yo = (FP) 0, zo = (FP) 0;
    if (i_out < num_out)
    {
        xo = wavenumber * x_out[i_out + idx_offset_out];
        yo = wavenumber * y_out[i_out + idx_offset_out];
        zo = wavenumber * z_out[i_out + idx_offset_out];
    }
    FP2* smem = reinterpret_cast<FP2*>(my_smem);
    FP2* c_w = smem;
    FP2* c_xy = c_w + max_in_chunk;
    int* c_index = (int*)(c_xy + max_in_chunk);
    FP* c_z = (FP*)(c_index + max_in_chunk);
    for (int j = 0; j < num_in; j += max_in_chunk)
    {
        int chunk_size = num_in - j;
        if (chunk_size > max_in_chunk)
        {
            chunk_size = max_in_chunk;
        }
        for (int t = threadIdx.x; t < chunk_size; t += blockDim.x)
        {
            const int g = j + t;
            c_w[t] = weights_in[g];
            c_xy[t].x = x_in[g];
            c_xy[t].y = y_in[g];
            c_index[t] = data_idx ? data_idx[g] : g;
            c_z[t] = z_in[g];
        }
        __syncthreads();
        if (i_out < num_out)
        {
            for (int i = 0; i < chunk_size; ++i)
            {
                FP re, im, t = xo * c_xy[i].x + yo * c_xy[i].y + zo * c_z[i];
                sincos(t, &im, &re);
                t = re;
                const FP2 w = c_w[i];
                re *= w.x; re -= w.y * im;
                im *= w.x; im += w.y * t;
                if (data)
                {
                    const int i_in = c_index[i] * num_out + i_out;
                    const FP2 in = data[i_in];
                    out.x += in.x * re; out.x -= in.y * im;
                    out.y += in.y * re; out.y += in.x * im;
                }
                else
                {
                    out.x += re;;
                    out.y += im;
                }
            }
        }
        __syncthreads();
    }
    out.x *= norm_factor;
    out.y *= norm_factor;
    if (i_out < num_out)
    {
        output[i_out + idx_offset_data] = out;
    }
}

SDP_CUDA_KERNEL(sdp_station_beam_dft_scalar<float, float2>)
SDP_CUDA_KERNEL(sdp_station_beam_dft_scalar<double, double2>)
