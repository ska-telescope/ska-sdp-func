/*
  The code in this file was converted from specfun.f from scipy, which
  is in turn based on the work by Shanjie Zhang and Jianming Jin in
  their book "Computation of Special Functions", 1996, John Wiley &
  Sons, Inc.

  This file has three parts:
  * Declarations to make f2c code work
  * f2c-converted routine aswfa, sdmn, sckb, segv
  * Routines for external usage
 */

#include <assert.h>
#include <complex>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/fourier_transforms/private_pswf.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"

using std::complex;


/*       ********************************** */
/* Subroutine */ static inline int pswf_sdmn(
        const int* m,
        const int* n,
        const double* c,
        const double* cv,
        const int* kd,
        double* df
)
{
    // NOLINTBEGIN(cppcoreguidelines-init-variables)

    /* System generated locals */
    int i_1, i_2;
    double d_1;

    /* Local variables */
    double a[200], d_[200], f, g[200];
    int i_, j, k;
    double f0, f1, f2;
    int k1;
    double r1, s0, r3, r4;
    int kb;
    double fl, cs;
    int ip, nm;
    double fs, sw, dk0, dk1, dk2, d2k, su1, su2;

    // NOLINTEND(cppcoreguidelines-init-variables)

/*       ===================================================== */
/*       Purpose: Compute the expansion coefficients of the */
/*                prolate and oblate spheroidal functions, dk */
/*       Input :  m  --- Mode parameter */
/*                n  --- Mode parameter */
/*                c  --- Spheroidal parameter */
/*                cv --- Characteristic value */
/*                KD --- Function code */
/*                       KD=1 for prolate; KD=-1 for oblate */
/*       Output:  DF(k) --- Expansion coefficients dk; */
/*                          DF(1), DF(2), ... correspond to */
/*                          d0, d2, ... for even n-m and d1, */
/*                          d3, ... for odd n-m */
/*       ===================================================== */

    /* Function Body */
    nm = (int) ((*n - *m) * .5f + *c) + 25;
    if (*c < 1e-10)
    {
        i_1 = nm;
        for (i_ = 1; i_ <= i_1; ++i_)
        {
/* L5: */
            df[i_ - 1] = 0.;
        }
        df[(*n - *m) / 2] = 1.;
        return 0;
    }
    cs = *c * *c * *kd;
    ip = 1;
    k = 0;
    if (*n - *m == (*n - *m) / 2 << 1)
    {
        ip = 0;
    }
    i_1 = nm + 2;
    for (i_ = 1; i_ <= i_1; ++i_)
    {
        if (ip == 0)
        {
            k = (i_ - 1) << 1;
        }
        if (ip == 1)
        {
            k = (i_ << 1) - 1;
        }
        dk0 = (double) (*m + k);
        dk1 = (double) (*m + k + 1);
        dk2 = (double) ((*m + k) << 1);
        d2k = (double) ((*m << 1) + k);
        a[i_ - 1] = (d2k + 2.f) * (d2k + 1.f) / ((dk2 + 3.f) * (dk2 + 5.f)) *
                cs;
        d_[i_ - 1] = dk0 * dk1 + (dk0 * 2.f * dk1 - *m * 2.f * *m - 1.f) / (
            (dk2 - 1.f) * (dk2 + 3.f)) * cs;
        g[i_ - 1] = k * (k - 1.f) / ((dk2 - 3.f) * (dk2 - 1.f)) * cs;
/* L10: */
    }
    fs = 1.;
    f1 = 0.;
    f0 = 1e-100;
    kb = 0;
    df[nm] = 0.;
    fl = 0.;
    for (k = nm; k >= 1; --k)
    {
        f = -((d_[k] - *cv) * f0 + a[k] * f1) / g[k];
        if (abs(f) > (d_1 = df[k], abs(d_1)))
        {
            df[k - 1] = f;
            f1 = f0;
            f0 = f;
            if (abs(f) > 1e100)
            {
                i_1 = nm;
                for (k1 = k; k1 <= i_1; ++k1)
                {
/* L12: */
                    df[k1 - 1] *= 1e-100;
                }
                f1 *= 1e-100;
                f0 *= 1e-100;
            }
        }
        else
        {
            kb = k;
            fl = df[k];
            f1 = 1e-100;
            // NOLINTBEGIN(clang-analyzer-core.UndefinedBinaryOperatorResult)
            f2 = -(d_[0] - *cv) / a[0] * f1;
            // NOLINTEND(clang-analyzer-core.UndefinedBinaryOperatorResult)
            df[0] = f1;
            if (kb == 1)
            {
                fs = f2;
            }
            else if (kb == 2)
            {
                df[1] = f2;
                fs = -((d_[1] - *cv) * f2 + g[1] * f1) / a[1];
            }
            else
            {
                df[1] = f2;
                i_1 = kb + 1;
                for (j = 3; j <= i_1; ++j)
                {
                    f = -((d_[j - 2] - *cv) * f2 + g[j - 2] * f1) / a[j - 2];
                    if (j <= kb)
                    {
                        df[j - 1] = f;
                    }
                    if (abs(f) > 1e100)
                    {
                        i_2 = j;
                        for (k1 = 1; k1 <= i_2; ++k1)
                        {
/* L15: */
                            df[k1 - 1] *= 1e-100;
                        }
                        f *= 1e-100;
                        f2 *= 1e-100;
                    }
                    f1 = f2;
/* L20: */
                    f2 = f;
                }
                fs = f;
            }
            goto L35;
        }
/* L30: */
    }
L35:
    r1 = 1.;
    i_1 = (*m + ip) << 1;
    for (j = *m + ip + 1; j <= i_1; ++j)
    {
/* L40: */
        r1 *= j;
    }
    // NOLINTBEGIN(clang-analyzer-core.UndefinedBinaryOperatorResult)
    su1 = df[0] * r1;
    // NOLINTEND(clang-analyzer-core.UndefinedBinaryOperatorResult)
    i_1 = kb;
    for (k = 2; k <= i_1; ++k)
    {
        r1 = -r1 * (k + *m + ip - 1.5) / (k - 1.);
/* L45: */
        su1 += r1 * df[k - 1];
    }
    su2 = 0.;
    sw = 0.;
    i_1 = nm;
    for (k = kb + 1; k <= i_1; ++k)
    {
        if (k != 1)
        {
            r1 = -r1 * (k + *m + ip - 1.5) / (k - 1.);
        }
        su2 += r1 * df[k - 1];
        if (abs(sw - su2) < abs(su2) * 1e-14)
        {
            goto L55;
        }
/* L50: */
        sw = su2;
    }
L55:
    r3 = 1.;
    i_1 = (*m + *n + ip) / 2;
    for (j = 1; j <= i_1; ++j)
    {
/* L60: */
        r3 *= j + (*n + *m + ip) * .5;
    }
    r4 = 1.;
    i_1 = (*n - *m - ip) / 2;
    for (j = 1; j <= i_1; ++j)
    {
/* L65: */
        r4 = r4 * -4. * j;
    }
    s0 = r3 / (fl * (su1 / fs) + su2) / r4;
    i_1 = kb;
    for (k = 1; k <= i_1; ++k)
    {
/* L70: */
        df[k - 1] = fl / fs * s0 * df[k - 1];
    }
    i_1 = nm;
    for (k = kb + 1; k <= i_1; ++k)
    {
/* L75: */
        df[k - 1] = s0 * df[k - 1];
    }
    return 0;
} /* sdmn_ */


/*       ********************************** */
/* Subroutine */ static inline int pswf_sckb(
        int m,
        int n,
        double c,
        const double* df,
        double* ck
)
{
    // NOLINTBEGIN(cppcoreguidelines-init-variables)
    /* System generated locals */
    int i_1, i_2;
    double d_1;

    /* Local variables */
    int i_, k;
    double r_, d1, d2, d3;
    int i1, i2;
    double r1;
    int ip, nm;
    double sw, fac, reg, sum;
    // NOLINTEND(cppcoreguidelines-init-variables)

/*       ====================================================== */
/*       Purpose: Compute the expansion coefficients of the */
/*                prolate and oblate spheroidal functions */
/*       Input :  m  --- Mode parameter */
/*                n  --- Mode parameter */
/*                c  --- Spheroidal parameter */
/*                DF(k) --- Expansion coefficients dk */
/*       Output:  CK(k) --- Expansion coefficients ck; */
/*                          CK(1), CK(2), ... correspond to */
/*                          c0, c2, ... */
/*       ====================================================== */

    /* Parameter adjustments */
    --ck;

    /* Function Body */
    if (c <= 1e-10)
    {
        c = 1e-10;
    }
    nm = (int) ((n - m) * .5f + c) + 25;
    ip = 1;
    if (n - m == (n - m) / 2 << 1)
    {
        ip = 0;
    }
    reg = 1.;
    if (m + nm > 80)
    {
        reg = 1e-200;
    }
    fac = -sdp_pswf_cipow(.5, m);
    sw = 0.;
    i_1 = nm - 1;
    for (k = 0; k <= i_1; ++k)
    {
        fac = -fac;
        i1 = (k << 1) + ip + 1;
        r_ = reg;
        i_2 = i1 + (m << 1) - 1;
        for (i_ = i1; i_ <= i_2; ++i_)
        {
/* L10: */
            r_ *= i_;
        }
        i2 = k + m + ip;
        i_2 = i2 + k - 1;
        for (i_ = i2; i_ <= i_2; ++i_)
        {
/* L15: */
            r_ *= i_ + .5;
        }
        sum = r_ * df[k];
        i_2 = nm;
        for (i_ = k + 1; i_ <= i_2; ++i_)
        {
            d1 = i_ * 2. + ip;
            d2 = m * 2. + d1;
            d3 = i_ + m + ip - .5;
            r_ = r_ * d2 * (d2 - 1.) * i_ * (d3 + k) / (d1 * (d1 - 1.) * (
                        i_ - k) * d3);
            // NOLINTBEGIN(clang-analyzer-core.UndefinedBinaryOperatorResult)
            sum += r_ * df[i_];
            // NOLINTEND(clang-analyzer-core.UndefinedBinaryOperatorResult)
            if ((d_1 = sw - sum, abs(d_1)) < abs(sum) * 1e-14)
            {
                goto L25;
            }
/* L20: */
            sw = sum;
        }
L25:
        r1 = reg;
        i_2 = m + k;
        for (i_ = 2; i_ <= i_2; ++i_)
        {
/* L30: */
            r1 *= i_;
        }
/* L35: */
        ck[k + 1] = fac * sum / r1;
    }
    return 0;
} /* sckb_ */


/*       ********************************** */
/* Subroutine */ static inline int pswf_segv(
        const int* m,
        const int* n,
        const double* c,
        const int* kd,
        double* cv,
        double* eg
)
{
    // NOLINTBEGIN(cppcoreguidelines-init-variables)
    /* System generated locals */
    int i_1, i_2;
    double d_1, d_2;

    /* Local variables */
    double a[300], b[100], d_[300], e[300], f[300], g[300], h_[
        100];
    int i_, j, k, l;
    double s, t;
    int k1;
    double t1, x1, cs, xa;
    int nm;
    double xb, dk0, dk1, dk2, d2k, cv0[100];
    int nm1, icm;
    // NOLINTEND(cppcoreguidelines-init-variables)

/*       ========================================================= */
/*       Purpose: Compute the characteristic values of spheroidal */
/*                wave functions */
/*       Input :  m  --- Mode parameter */
/*                n  --- Mode parameter */
/*                c  --- Spheroidal parameter */
/*                KD --- Function code */
/*                       KD=1 for Prolate; KD=-1 for Oblate */
/*       Output:  CV --- Characteristic value for given m, n and c */
/*                EG(L) --- Characteristic value for mode m and n' */
/*                          ( L = n' - m + 1 ) */
/*       ========================================================= */

    /* Parameter adjustments */
    --eg;

    /* Function Body */
    if (*c < 1e-10)
    {
        i_1 = *n - *m + 1;
        for (i_ = 1; i_ <= i_1; ++i_)
        {
/* L5: */
            eg[i_] = (i_ + *m) * (i_ + *m - 1.);
        }
        goto L70;
    }
    icm = (*n - *m + 2) / 2;
    nm = (int) ((*n - *m) * .5f + *c) + 10;
    cs = *c * *c * *kd;
    k = 0;
    for (l = 0; l <= 1; ++l)
    {
        i_1 = nm;
        for (i_ = 1; i_ <= i_1; ++i_)
        {
            if (l == 0)
            {
                k = (i_ - 1) << 1;
            }
            if (l == 1)
            {
                k = (i_ << 1) - 1;
            }
            dk0 = (double) (*m + k);
            dk1 = (double) (*m + k + 1);
            dk2 = (double) ((*m + k) << 1);
            d2k = (double) ((*m << 1) + k);
            a[i_ - 1] = (d2k + 2.f) * (d2k + 1.f) / ((dk2 + 3.f) * (dk2 +
                    5.f)) * cs;
            d_[i_ - 1] = dk0 * dk1 + (dk0 * 2.f * dk1 - *m * 2.f * *m - 1.f)
                    / ((dk2 - 1.f) * (dk2 + 3.f)) * cs;
/* L10: */
            g[i_ - 1] = k * (k - 1.f) / ((dk2 - 3.f) * (dk2 - 1.f)) * cs;
        }
        i_1 = nm;
        for (k = 2; k <= i_1; ++k)
        {
            e[k - 1] = sqrt(a[k - 2] * g[k - 1]);
/* L15: */
            f[k - 1] = e[k - 1] * e[k - 1];
        }
        f[0] = 0.;
        e[0] = 0.;
        xa = d_[nm - 1] + (d_1 = e[nm - 1], abs(d_1));
        xb = d_[nm - 1] - (d_1 = e[nm - 1], abs(d_1));
        nm1 = nm - 1;
        i_1 = nm1;
        for (i_ = 1; i_ <= i_1; ++i_)
        {
            // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
            t = (d_1 = e[i_ - 1], abs(d_1)) + (d_2 = e[i_], abs(d_2));
            // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
            t1 = d_[i_ - 1] + t;
            if (xa < t1)
            {
                xa = t1;
            }
            t1 = d_[i_ - 1] - t;
            if (t1 < xb)
            {
                xb = t1;
            }
/* L20: */
        }
        i_1 = icm;
        for (i_ = 1; i_ <= i_1; ++i_)
        {
            b[i_ - 1] = xa;
/* L25: */
            h_[i_ - 1] = xb;
        }
        i_1 = icm;
        for (k = 1; k <= i_1; ++k)
        {
            i_2 = icm;
            for (k1 = k; k1 <= i_2; ++k1)
            {
                if (b[k1 - 1] < b[k - 1])
                {
                    b[k - 1] = b[k1 - 1];
                    goto L35;
                }
/* L30: */
            }
L35:
            if (k != 1)
            {
                if (h_[k - 1] < h_[k - 2])
                {
                    h_[k - 1] = h_[k - 2];
                }
            }
L40:
            x1 = (b[k - 1] + h_[k - 1]) / 2.;
            cv0[k - 1] = x1;
            if ((d_1 = (b[k - 1] - h_[k - 1]) / x1, abs(d_1)) < 1e-14)
            {
                goto L50;
            }
            j = 0;
            s = 1.;
            i_2 = nm;
            for (i_ = 1; i_ <= i_2; ++i_)
            {
                if (s == 0.)
                {
                    s += 1e-30;
                }
                t = f[i_ - 1] / s;
                s = d_[i_ - 1] - t - x1;
                if (s < 0.)
                {
                    ++j;
                }
/* L45: */
            }
            if (j < k)
            {
                h_[k - 1] = x1;
            }
            else
            {
                b[k - 1] = x1;
                if (j >= icm)
                {
                    b[icm - 1] = x1;
                }
                else
                {
                    if (h_[j] < x1)
                    {
                        h_[j] = x1;
                    }
                    if (x1 < b[j - 1])
                    {
                        b[j - 1] = x1;
                    }
                }
            }
            goto L40;
L50:
            cv0[k - 1] = x1;
            if (l == 0)
            {
                eg[(k << 1) - 1] = cv0[k - 1];
            }
            if (l == 1)
            {
                eg[k * 2] = cv0[k - 1];
            }
/* L55: */
        }
/* L60: */
    }
L70:
    *cv = eg[*n - *m + 1];
    return 0;
} /* segv_ */


template<class T>
static inline void generate_pswf(
        int m,
        double c,
        double* ck,
        sdp_Mem* pswf,
        int end_correction,
        sdp_Error* status
)
{
    sdp_MemViewCpu<T, 1> pswf_;
    sdp_mem_check_and_view(pswf, &pswf_, status);
    if (*status) return;

    // Get value at 0
    const int size = (int) pswf_.shape[0];
    pswf_(0) = 0.0;
    pswf_(size / 2) = sdp_pswf_aswfa(m, m, c, ck, 0);

    // Get remaining values
    for (int i = 1; i < size / 2; i++)
    {
        // Get value (plus derivative)
        const double x = 2 * ((double)i) / size;
        const double s1f = sdp_pswf_aswfa(m, m, c, ck, x);
        pswf_(size / 2 + i) = s1f;
        pswf_(size / 2 - i) = s1f;
    }
    if (end_correction && size % 2 == 0)
    {
        pswf_(0) = 1e-15;
    }
}


struct sdp_Pswf
{
    int m;
    double c;
    sdp_Mem* pswf;
    sdp_Mem* pswf_gpu;
    sdp_Mem* coeff;
    sdp_Mem* coeff_gpu;
    double* coeff_p;
};


sdp_Pswf* sdp_pswf_create(int m, double c)
{
    sdp_Error status = SDP_SUCCESS;
    sdp_Pswf* plan = (sdp_Pswf*) calloc(1, sizeof(sdp_Pswf));
    const int64_t coeff_shape[] = {200};
    plan->coeff = sdp_mem_create(
            SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, coeff_shape, &status
    );
    plan->coeff_p = (double*) sdp_mem_data(plan->coeff);
    plan->m = m;
    plan->c = c;

    // Calculate characteristic values of spheroidal wave functions
    int n = m;
    int kd = 1; // prolate
    double cv = 0, eg[2];
    pswf_segv(&plan->m, &n, &plan->c, &kd, &cv, eg);

    // Calculate expansion coefficients
    double df[200];
    pswf_sdmn(&plan->m, &n, &plan->c, &cv, &kd, df);
    pswf_sckb(plan->m, n, plan->c, df, plan->coeff_p);
    return plan;
}


const sdp_Mem* sdp_pswf_coeff(
        sdp_Pswf* plan,
        sdp_MemLocation location,
        sdp_Error* status
)
{
    if (location == SDP_MEM_GPU)
    {
        if (!plan->coeff_gpu)
        {
            plan->coeff_gpu = sdp_mem_create_copy(
                    plan->coeff, location, status
            );
        }
        return plan->coeff_gpu;
    }
    return plan->coeff;
}


const sdp_Mem* sdp_pswf_values(
        sdp_Pswf* plan,
        sdp_MemLocation location,
        sdp_Error* status
)
{
    if (location == SDP_MEM_GPU)
    {
        if (!plan->pswf_gpu)
        {
            plan->pswf_gpu = sdp_mem_create_copy(plan->pswf, location, status);
        }
        return plan->pswf_gpu;
    }
    return plan->pswf;
}


double sdp_pswf_evaluate(const sdp_Pswf* plan, double x)
{
    const int m = plan->m, n = plan->m;
    const double abs_x = fabs(x);
    return abs_x < 1 ? sdp_pswf_aswfa(m, n, plan->c, plan->coeff_p, abs_x) : 0;
}


double sdp_pswf_par_c(const sdp_Pswf* plan)
{
    return plan->c;
}


double sdp_pswf_par_m(const sdp_Pswf* plan)
{
    return plan->m;
}


void sdp_pswf_free(sdp_Pswf* plan)
{
    sdp_mem_free(plan->pswf);
    sdp_mem_free(plan->pswf_gpu);
    sdp_mem_free(plan->coeff);
    sdp_mem_free(plan->coeff_gpu);
    free(plan);
}


void sdp_pswf_generate(
        sdp_Pswf* plan,
        sdp_Mem* out,
        int size,
        int end_correction,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_Mem* out_ptr = out;
    if (out && size == 0)
    {
        if (sdp_mem_num_dims(out) != 1)
        {
            *status = SDP_ERR_INVALID_ARGUMENT;
            return;
        }
        if (sdp_mem_location(out) != SDP_MEM_CPU)
        {
            *status = SDP_ERR_MEM_LOCATION;
            return;
        }
    }
    else if (!out && size > 0)
    {
        // Remove any cached versions first.
        sdp_mem_free(plan->pswf);
        sdp_mem_free(plan->pswf_gpu);
        plan->pswf_gpu = 0;
        const int64_t shape[] = {size};
        plan->pswf = sdp_mem_create(
                SDP_MEM_DOUBLE, SDP_MEM_CPU, 1, shape, status
        );
        out_ptr = plan->pswf;
    }
    else
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Must specify only one of 'out' or 'size'");
        return;
    }
    double* ck = (double*) sdp_mem_data(plan->coeff);
    switch (sdp_mem_type(out_ptr))
    {
    case SDP_MEM_DOUBLE:
        generate_pswf<double>(
                plan->m, plan->c, ck, out_ptr, end_correction, status
        );
        break;
    case SDP_MEM_FLOAT:
        generate_pswf<float>(
                plan->m, plan->c, ck, out_ptr, end_correction, status
        );
        break;
    case SDP_MEM_COMPLEX_FLOAT:
        generate_pswf<complex<float> >(
                plan->m, plan->c, ck, out_ptr, end_correction, status
        );
        break;
    case SDP_MEM_COMPLEX_DOUBLE:
        generate_pswf<complex<double> >(
                plan->m, plan->c, ck, out_ptr, end_correction, status
        );
        break;
    default:
        *status = SDP_ERR_DATA_TYPE;
    }
}


/*
 * Generate prolate spheroidal wave function (PSWF).
 */
void sdp_generate_pswf(
        int m,
        double c,
        sdp_Mem* out,
        sdp_Error* status
)
{
    if (*status) return;
    sdp_Pswf* pswf = sdp_pswf_create(m, c);
    const int size = 0, end_correction = 0;
    sdp_pswf_generate(pswf, out, size, end_correction, status);
    sdp_pswf_free(pswf);
}
