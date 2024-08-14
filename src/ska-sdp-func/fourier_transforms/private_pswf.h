/*
  The code in this file was converted from specfun.f from scipy, which
  is in turn based on the work by Shanjie Zhang and Jianming Jin in
  their book "Computation of Special Functions", 1996, John Wiley &
  Sons, Inc.
 */

#ifndef SKA_SDP_PROC_FUNC_PRIVATE_PSWF_H_
#define SKA_SDP_PROC_FUNC_PRIVATE_PSWF_H_

/**
 * @file private_pswf.h
 */

#include <math.h>
#include "ska-sdp-func/math/sdp_math_macros.h"

#define abs(x) ((x) >= 0 ? (x) : -(x))

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Helper function to evaluate base raised to integer exponent.
 *
 * @param base Base.
 * @param exp Integer exponent.
 * @return Base raised to integer exponent.
 */
SDP_INLINE
double sdp_pswf_cipow(double base, int exp)
{
    double result = 1;
    // This is never called with negative exponents. Avoids potential recursion.
    if (exp == 1) return base;
    while (exp)
    {
        if ((exp & 1) != 0) result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}


/**
 * @brief Evaluate PSWF at a specific point.
 *
 * Compute the prolate and oblate spheroidal angular functions of the first
 * kind and their derivatives.
 *
 * This function has been heavily specialised, as it it basically the
 * inner loop of PSWF generation.
 *
 * @param m Mode parameter, m = 0, 1, 2, ...
 * @param n Mode parameter, n = m, m + 1, ...
 * @param c Spheroidal parameter.
 * @param ck Expansion coefficients; CK(1), CK(2) ... correspond to c0, c2 ...
 * @param x Argument of angular function, |x| < 1.0
 * @return Angular function of the first kind.
 */
SDP_INLINE
double sdp_pswf_aswfa(int m, int n, double c, const double* ck, double x)
{
    const int nm = (int) ((n - m) / 2 + c) + 40;
    const int nm2 = nm / 2 - 2;
    const double x1 = 1.0 - x * x;
    const double a0 = (m == 0 && x1 == 0.0) ? 1.0 : pow(x1, m * 0.5);
    // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
    double su1 = ck[0];
    // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
    for (int k = 1; k <= nm2; ++k)
    {
        const double r_ = ck[k] * sdp_pswf_cipow(x1, k);
        su1 += r_;
        const double t_ = r_ / su1;
        if (k >= 10 && abs(t_) < 1e-14) break;
    }
    return ((n - m) % 2 == 0) ? (a0 * su1) : (a0 * x * su1);
}

#ifdef __cplusplus
}
#endif

#endif /* include guard */
