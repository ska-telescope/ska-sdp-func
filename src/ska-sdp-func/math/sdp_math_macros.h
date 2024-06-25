/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_MATH_MACROS_H_
#define SDP_MATH_MACROS_H_

#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

/* Speed of light, in m/s. */
#define C_0 299792458.0

/* Max and min macros (used in some CUDA kernels). */
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

#endif /* include guard */
