/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_ELEMENT_SPH_WAVE_HARP_H_
#define SKA_SDP_PROC_FUNC_ELEMENT_SPH_WAVE_HARP_H_

/**
 * @file sdp_element_spherical_wave_harp.h
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
 * Evaluates an element beam using spherical wave coefficients (HARP version).
 *
 * @details
 * This function evaluates an element using HARP-compatible
 * spherical wave coefficients, at a set of source positions.
 *
 * @param num_points Number of source coordinates to use.
 * @param theta_rad Point position (modified) theta values in rad.
 * @param phi_x_rad Point position (modified) phi values for X, in rad.
 * @param phi_y_rad Point position (modified) phi values for Y, in rad.
 * @param l_max Maximum order of spherical wave.
 * @param coeffs TE and TM mode coefficients for X and Y antennas.
 * @param index_offset_element_beam Start offset into output array.
 * @param element_beam Output complex element beam array.
 * @param status Error status.
*/
void sdp_element_beam_spherical_wave_harp(
        int num_points,
        const sdp_Mem* theta_rad,
        const sdp_Mem* phi_x_rad,
        const sdp_Mem* phi_y_rad,
        int l_max,
        const sdp_Mem* coeffs,
        int index_offset_element_beam,
        sdp_Mem* element_beam,
        sdp_Error* status
);

/** @} */ /* End group element_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
