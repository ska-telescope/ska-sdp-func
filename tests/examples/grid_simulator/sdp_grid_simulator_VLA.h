/*
 * sdp_grid_simulator_VLA.h
 *
 *  Created on: 5 Feb 2024
 *      Author: vlad
 */

#ifndef SDP_GRID_SIMULATOR_VLA_H_
#define SDP_GRID_SIMULATOR_VLA_H_

/**
 * @file sdp_function_name.h
 *       (Change this to match the name of the header file)
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sdp_Double2
{
    double x;
    double y;
} sdp_Double2;


// VLA configuration
const double vlas[][3] =
		{{-4.01284200e+02, -2.70639500e+02,  1.33450000e+00},
		       {-1.31799260e+03, -8.89027900e+02,  2.03360000e+00},
		       {-2.64299430e+03, -1.78274590e+03,  7.83280000e+00},
		       {-4.32994140e+03, -2.92062980e+03,  4.21700000e+00},
		       {-6.35001200e+03, -4.28312470e+03, -6.07790000e+00},
		       {-8.68248720e+03, -5.85645850e+03, -7.38610000e+00},
		       {-1.13114962e+04, -7.62938500e+03, -1.93219000e+01},
		       {-1.42243397e+04, -9.59402680e+03, -3.22199000e+01},
		       {-1.74101952e+04, -1.17426658e+04, -5.25716000e+01},
		       { 4.38695300e+02, -2.04497100e+02, -1.94900000e-01},
		       { 1.44099740e+03, -6.71852900e+02,  6.19900000e-01},
		       { 2.88945970e+03, -1.34723240e+03,  1.24453000e+01},
		       { 4.73362700e+03, -2.20712600e+03,  1.99349000e+01},
		       { 6.94206610e+03, -3.23684230e+03,  2.80543000e+01},
		       { 9.49192690e+03, -4.42550980e+03,  1.93104000e+01},
		       { 1.23660731e+04, -5.76530610e+03,  1.38351000e+01},
		       { 1.55504596e+04, -7.24969040e+03,  2.53408000e+01},
		       { 1.90902771e+04, -8.74844180e+03, -5.32768000e+01},
		       {-3.80377000e+01,  4.34713500e+02, -2.60000000e-02},
		       {-1.24977500e+02,  1.42815670e+03, -1.40120000e+00},
		       {-2.59368400e+02,  2.96335470e+03, -8.15000000e-02},
		       {-4.10658700e+02,  4.69150510e+03, -3.72200000e-01},
		       {-6.02292000e+02,  6.88014080e+03,  5.88500000e-01},
		       {-8.23556900e+02,  9.40751720e+03,  6.47000000e-02},
		       {-1.07292720e+03,  1.22558935e+04, -4.27410000e+00},
		       {-1.34924890e+03,  1.54117447e+04, -7.76930000e+00},
		       {-1.65146370e+03,  1.88634683e+04, -9.22480000e+00}};
const int nants_vla = 27;


/**
 * @brief Visibility simulator based on the VLA telescope layout.
 *
 * The function uses VLA antenna layout and a source list 
 * to simulate the visibilities given the hour angle range and
 * the declination. It also performs a simple gridding producing
 * a 2D grid as an output
 *
 * @param source_list Input source list, 2D SDP_MEM_DOUBLE, {Amp, l, m}.
 * @param ha_start Starting value of the hour angle in radians, double.
 * @param ha_step Step in HA, double.
 * @param ha_num Number of HA points, integer.
 * @param dec_rad Declination in radians, double.
 * @param uvw_scale Scaling factor to divide UVW values, double.
 * @param grid_sim Output grid array, 2D SDP_MEM_COMPLEX_DOUBLE.
 * @param status Error status.
 */
void sdp_grid_simulator_VLA(
        sdp_Mem* source_list,
		double ha_start,
		double ha_step,
		int ha_num,
		double dec_rad,
		double uvw_scale,
		sdp_Mem* grid_sim,
        sdp_Error* status);

#ifdef __cplusplus
}
#endif



#endif /* SDP_GRID_SIMULATOR_VLA_H_ */
