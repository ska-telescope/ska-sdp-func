/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/grid_data/sdp_gridder_clamp_channels.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"


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
    const double eta = 1e-2;
    const int range_includes_zero = (min_u <= 0 && max_u > 0);
    sdp_MemViewCpu<const double, 2> uvws_;
    sdp_MemViewCpu<int, 1> start_chs_, end_chs_;
    sdp_mem_check_and_view(uvws, &uvws_, status);
    sdp_mem_check_and_view(start_ch, &start_chs_, status);
    sdp_mem_check_and_view(end_ch, &end_chs_, status);

    for (int64_t i = 0; i < uvws_.shape[0]; ++i)
    {
        const double u = uvws_(i, dim);
        if (abs(u) > eta * C_0 / dfreq_hz)
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
