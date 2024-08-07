/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

// Begin anonymous namespace for file-local functions.
namespace {

template<typename UVW_TYPE>
void clamp_channels_single(
        const sdp_Mem* uvws,
        const int dim,
        const double freq0_hz,
        const double dfreq_hz,
        sdp_Mem* start_ch,
        sdp_Mem* end_ch,
        const double min_u,
        const double max_u,
        sdp_Error* status
)
{
    if (*status) return;
    const double eta = 1e-2;
    const int range_includes_zero = (min_u <= 0 && max_u > 0);
    sdp_MemViewCpu<const UVW_TYPE, 2> uvws_;
    sdp_MemViewCpu<int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(start_ch, &start_chs_, status);
    sdp_mem_check_and_view(end_ch, &end_chs_, status);

    for (int64_t i = 0; i < uvws_.shape[0]; ++i)
    {
        const double u = uvws_(i, dim);
        if (fabs(u) > eta * C_0 / dfreq_hz)
        {
            const double u0 = u * freq0_hz / C_0;
            const double du = u * dfreq_hz / C_0;
            const int64_t mins = (int64_t) (ceil((min_u - u0) / du));
            const int64_t maxs = (int64_t) (ceil((max_u - u0) / du));
            const int is_positive = du > 0;
            const int start_ch_ = is_positive ? (int) mins : (int) maxs;
            const int end_ch_ = is_positive ? (int) maxs : (int) mins;
            start_chs_(i) = MAX(start_chs_(i), start_ch_);
            end_chs_(i) = MIN(end_chs_(i), end_ch_);
        }
        else if (!range_includes_zero)
        {
            start_chs_(i) = 0;
            end_chs_(i) = 0;
        }
        end_chs_(i) = MAX(end_chs_(i), start_chs_(i));
    }
}

} // End anonymous namespace for file-local functions.


void sdp_gridder_clamp_channels_single(
        const sdp_Mem* uvws,
        const int dim,
        const double freq0_hz,
        const double dfreq_hz,
        sdp_Mem* start_ch,
        sdp_Mem* end_ch,
        const double min_u,
        const double max_u,
        sdp_Error* status
)
{
    if (*status) return;
    const sdp_MemLocation loc = sdp_mem_location(uvws);
    if (loc == SDP_MEM_CPU)
    {
        if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE)
        {
            clamp_channels_single<double>(uvws, dim, freq0_hz, dfreq_hz,
                    start_ch, end_ch, min_u, max_u, status
            );
        }
        else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT)
        {
            clamp_channels_single<float>(uvws, dim, freq0_hz, dfreq_hz,
                    start_ch, end_ch, min_u, max_u, status
            );
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
            return;
        }
    }
    else if (loc == SDP_MEM_GPU)
    {
        // Call the kernel.
        uint64_t num_threads[] = {256, 1, 1}, num_blocks[] = {1, 1, 1};
        const int64_t num_elements = sdp_mem_shape_dim(uvws, 0);
        num_blocks[0] = (num_elements + num_threads[0] - 1) / num_threads[0];
        sdp_MemViewGpu<const double, 2> uvws_dbl;
        sdp_MemViewGpu<const float, 2> uvws_flt;
        sdp_MemViewGpu<int, 1> start_chs_, end_chs_;
        sdp_mem_check_and_view(start_ch, &start_chs_, status);
        sdp_mem_check_and_view(end_ch, &end_chs_, status);
        const char* kernel_name = 0;
        int is_dbl = 0;
        if (sdp_mem_type(uvws) == SDP_MEM_DOUBLE)
        {
            is_dbl = 1;
            sdp_mem_check_and_view(uvws, &uvws_dbl, status);
            kernel_name = "sdp_gridder_clamp_channels_single<double>";
        }
        else if (sdp_mem_type(uvws) == SDP_MEM_FLOAT)
        {
            is_dbl = 0;
            sdp_mem_check_and_view(uvws, &uvws_flt, status);
            kernel_name = "sdp_gridder_clamp_channels_single<float>";
        }
        else
        {
            *status = SDP_ERR_DATA_TYPE;
        }
        const void* arg[] = {
            is_dbl ? (const void*)&uvws_dbl : (const void*)&uvws_flt,
            (const void*)&dim,
            (const void*)&freq0_hz,
            (const void*)&dfreq_hz,
            (const void*)&start_chs_,
            (const void*)&end_chs_,
            (const void*)&min_u,
            (const void*)&max_u,
        };
        sdp_launch_cuda_kernel(kernel_name,
                num_blocks, num_threads, 0, 0, arg, status
        );
    }
}
