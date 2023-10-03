/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_ELEMENT_DIPOLE_H_
#define SKA_SDP_PROC_FUNC_ELEMENT_DIPOLE_H_

/**
 * @file sdp_element_dipole.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup element_func
 * @{
 */

/**
 * @brief
 * Evaluates an element beam from a dipole.
 *
 * @details
 * This function evaluates an element beam from a dipole,
 * at a set of source positions.
 *
 * @param num_points Number of source coordinates to use.
 * @param theta_rad Point position (modified) theta values in rad.
 * @param phi_rad Point position (modified) phi values in rad.
 * @param freq_hz Observing frequency in Hz.
 * @param dipole_length_m Length of dipole in metres.
 * @param stride_element_beam Stride into output array (normally 1 or 4).
 * @param index_offset_element_beam Start offset into output array.
 * @param element_beam Output complex element beam array.
 * @param status Error status.
*/
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
);

/** @} */ /* End group element_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
