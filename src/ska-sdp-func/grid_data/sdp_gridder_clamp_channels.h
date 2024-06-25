/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_GRIDDER_CLAMP_CHANNELS_H_
#define SDP_GRIDDER_CLAMP_CHANNELS_H_

/**
 * @file sdp_gridder_clamp_channels.h
 */

#include "ska-sdp-func/math/sdp_math_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Clamp channels for a particular uvw position.
 *
 * Restricts a channel range such that all visibilities lie in
 * the given uvw bounding box.
 *
 * Adapted from
 * https://gitlab.com/ska-telescope/sdp/ska-sdp-exec-iotest/-/blob/proc-func-refactor/src/grid.c?ref_type=heads#L464
 *
 * @param uvw u or v or w position, in metres.
 * @param freq0_hz Frequency of first channel, in Hz.
 * @param dfreq_hz Channel width, in Hz.
 * @param start_ch Channel range to clamp (excluding end).
 * @param end_ch Channel range to clamp (excluding end).
 * @param min_uvw Minimum value for u or v or w (inclusive).
 * @param max_uvw Maximum value for u or v or w (exclusive).
 */
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
void sdp_gridder_clamp_channels_inline(
        const double u,
        const double freq0_hz,
        const double dfreq_hz,
        int64_t* start_ch,
        int64_t* end_ch,
        const double min_u,
        const double max_u
)
{
    // We have to be slightly careful about degenerate cases in the
    // division below - not only can we have divisions by zero,
    // but also channel numbers that go over the integer range.
    // So it is safer to round these coordinates to zero for the
    // purpose of the bounds check.
    const double eta = 1e-3;
    const double u0 = freq0_hz * u / C_0;
    const double du = dfreq_hz * u / C_0;

    // Note the symmetry below: we get precisely the same expression
    // for maximum and minimum, however start_ch is inclusive but
    // end_ch is exclusive. This means that two calls to
    // clamp_channels where any min_uvw is equal to any max_uvw will
    // never return overlapping channel ranges.
    if (u > eta)
    {
        const int64_t start_ch_ = int64_t(ceil((min_u - u0) / du));
        const int64_t end_ch_ = int64_t(ceil((max_u - u0) / du));
        *start_ch = MAX(*start_ch, start_ch_);
        *end_ch = MIN(*end_ch, end_ch_);
    }
    else if (u < -eta)
    {
        const int64_t start_ch_ = int64_t(ceil((max_u - u0) / du));
        const int64_t end_ch_ = int64_t(ceil((min_u - u0) / du));
        *start_ch = MAX(*start_ch, start_ch_);
        *end_ch = MIN(*end_ch, end_ch_);
    }
    else
    {
        // Assume u = 0, which makes this a binary decision:
        // Does the range include 0 or not? Also let's be careful
        // just in case somebody puts a subgrid boundary right at zero.
        if (min_u > 0 || max_u <= 0)
        {
            *start_ch = 0;
            *end_ch = 0;
        }
    }
    if (*end_ch <= *start_ch)
    {
        *start_ch = 0;
        *end_ch = 0;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* include guard */
