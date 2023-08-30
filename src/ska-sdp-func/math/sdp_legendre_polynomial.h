/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef __cplusplus


/* out0 is P_l^m(cos_theta),
 * out1 is P_l^m(cos_theta) / sin_theta,
 * out2 is d/d(cos_theta){P_l^m(cos_theta)} * sin_theta. */
template<typename FP>
void sdp_legendre2(
        int l,
        int m,
        FP cos_t,
        FP sin_t,
        FP& out0,
        FP& out1,
        FP& out2
)
{
    FP p0 = (FP) 1, p1 = (FP) 0;
    if (m > 0)
    {
        FP fact = (FP) 1;
        for (int i = 1; i <= m; ++i)
        {
            p0 *= (-fact) * sin_t;
            fact += (FP) 2;
        }
    }
    out0 = cos_t * (2 * m + 1) * p0;
    if (l == m)
    {
        p1 = out0;
        out0 = p0;
    }
    else
    {
        p1 = out0;
        for (int i = m + 2; i <= l + 1; ++i)
        {
            out0 = p1;
            p1 = ((2 * i - 1) * cos_t * out0 - (i + m - 1) * p0) / (i - m);
            p0 = out0;
        }
    }
    if (sin_t != (FP) 0)
    {
        // BOTH of these are divides.
        out1 = out0 / sin_t;
        out2 = (cos_t * out0 * (l + 1) - p1 * (l - m + 1)) / sin_t;
    }
    else
    {
        out1 = out2 = (FP) 0;
    }
}

#endif /* __cplusplus */
