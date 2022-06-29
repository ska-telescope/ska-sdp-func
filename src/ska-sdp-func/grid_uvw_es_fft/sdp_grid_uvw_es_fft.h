/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_GRIDDER_H_
#define SKA_SDP_PROC_GRIDDER_H_

/**
 * @file sdp_gridder.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare structure handles for private implementation. */
struct sdp_Gridder;
typedef struct sdp_Gridder sdp_Gridder;

/**
 * @defgroup Gridder_func
 * @{
 */

/**
 * @brief Creates gridder.
 *
 * @param a Value of a.
 * @param b Value of b.
 * @param c Value of c.
 * @param status Error status.
 * @return sdp_Gridder* Handle to processing function.
 */
sdp_Gridder* sdp_gridder_create_plan(
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,  // in Hz
        const sdp_Mem* vis,
        const sdp_Mem* weight,
        const sdp_Mem* dirty_image,
        const double pixsize_x_rad, 
        const double pixsize_y_rad, 
        const double epsilon,
        const double min_abs_w, 
        const double max_abs_w, 
        const bool do_wstacking,
        sdp_Error* status
    );

/**
 * @brief Demonstrate a function utilising a plan.
 *
 * @param plan Handle to processing function.
 * @param output Output buffer.
 * @param status Error status.
 */
void sdp_gridder_ms2dirty(
        sdp_Gridder* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
        const sdp_Mem* vis,
        const sdp_Mem* weight,
              sdp_Mem *dirty_image,
        sdp_Error* status
    );

void sdp_gridder_dirty2ms(
        sdp_Gridder* plan,
        const sdp_Mem* uvw,
        const sdp_Mem* freq_hz,
              sdp_Mem* vis,
        const sdp_Mem* weight,
              sdp_Mem *dirty_image,   // even though this is an input, it is modified in place so can't be constant
        sdp_Error* status
    );

/**
 * @brief Releases handle to gridder.
 *
 * @param handle Handle to gridder.
 */
void sdp_gridder_free_plan(sdp_Gridder* handle);

/** @} */ /* End group Gridder_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
