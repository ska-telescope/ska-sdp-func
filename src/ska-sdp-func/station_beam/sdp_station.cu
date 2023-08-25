/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>

#include "ska-sdp-func/utility/sdp_device_wrapper.h"


template<typename FP, typename FP2, int NUM_POL>
__global__ void sdp_station_beam_dft(
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
        const int idx_offset_output,
        FP2* __restrict__ output,
        const FP norm_factor,
        const int eval_x,
        const int eval_y,
        const int max_in_chunk
)
{
    extern __shared__ __align__(64) unsigned char my_smem[];
    const int i_out = blockDim.x * blockIdx.x + threadIdx.x;
    FP2 out[NUM_POL];
    #pragma unroll
    for (int k = 0; k < NUM_POL; ++k)
    {
        out[k].x = out[k].y = (FP) 0;
    }
    FP xo = (FP) 0, yo = (FP) 0, zo = (FP) 0;
    if (i_out < num_out)
    {
        xo = wavenumber * x_out[i_out + idx_offset_out];
        yo = wavenumber * y_out[i_out + idx_offset_out];
        zo = z_out ? wavenumber * z_out[i_out + idx_offset_out] : (FP) 0;
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
                    const int i_in = NUM_POL * (c_index[i] * num_out + i_out);
                    if (NUM_POL == 1)
                    {
                        const FP2 in = data[i_in];
                        out[0].x += (in.x * re - in.y * im);
                        out[0].y += (in.y * re + in.x * im);
                    }
                    else if (NUM_POL == 4)
                    {
                        if (eval_x)
                        {
                            const FP2 xx = data[i_in + 0];
                            const FP2 xy = data[i_in + 1];
                            out[0].x += (xx.x * re - xx.y * im);
                            out[0].y += (xx.y * re + xx.x * im);
                            out[1].x += (xy.x * re - xy.y * im);
                            out[1].y += (xy.y * re + xy.x * im);
                        }
                        if (eval_y)
                        {
                            const FP2 yx = data[i_in + 2];
                            const FP2 yy = data[i_in + 3];
                            out[2].x += (yx.x * re - yx.y * im);
                            out[2].y += (yx.y * re + yx.x * im);
                            out[3].x += (yy.x * re - yy.y * im);
                            out[3].y += (yy.y * re + yy.x * im);
                        }
                    }
                }
                else
                {
                    if (NUM_POL == 1)
                    {
                        out[0].x += re;
                        out[0].y += im;
                    }
                    else if (NUM_POL == 4)
                    {
                        if (eval_x)
                        {
                            out[0].x += re;
                            out[0].y += im;
                            out[1].x += re;
                            out[1].y += im;
                        }
                        if (eval_y)
                        {
                            out[2].x += re;
                            out[2].y += im;
                            out[3].x += re;
                            out[3].y += im;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (i_out < num_out)
    {
        #pragma unroll
        for (int k = 0; k < NUM_POL; ++k)
        {
            out[k].x *= norm_factor;
            out[k].y *= norm_factor;
        }
        const int i_out_offset_scaled = NUM_POL * (i_out + idx_offset_output);
        if (NUM_POL == 1)
        {
            output[i_out_offset_scaled] = out[0];
        }
        else if (NUM_POL == 4)
        {
            if (eval_x)
            {
                output[i_out_offset_scaled + 0] = out[0];
                output[i_out_offset_scaled + 1] = out[1];
            }
            if (eval_y)
            {
                output[i_out_offset_scaled + 2] = out[2];
                output[i_out_offset_scaled + 3] = out[3];
            }
        }
    }
}

SDP_CUDA_KERNEL(sdp_station_beam_dft<float, float2, 1>)
SDP_CUDA_KERNEL(sdp_station_beam_dft<double, double2, 1>)
SDP_CUDA_KERNEL(sdp_station_beam_dft<float, float2, 4>)
SDP_CUDA_KERNEL(sdp_station_beam_dft<double, double2, 4>)
