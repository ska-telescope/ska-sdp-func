/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ska-sdp-func/fourier_transforms/sdp_pswf.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <complex>
#include <assert.h>
#include <string>

template<typename num_t> double val(num_t);


template<>
double val<float>(float x)
{
    return x;
}


template<>
double val<double>(double x)
{
    return x;
}


template<>
double val<std::complex<float> >(std::complex<float> x)
{
    return x.real();
}


template<>
double val<std::complex<double> >(std::complex<double> x)
{
    return x.real();
}


template<typename num_t>
void check(
        const char* test_name,
        sdp_MemType typ,
        const int m,
        const double c,
        const int64_t size,
        const double* expected,
        const double limit
)
{
    // Allocate memory
    sdp_Error status = SDP_SUCCESS;
    sdp_Mem* pswf_out = sdp_mem_create(
            typ, SDP_MEM_CPU, 1, &size, &status
    );
    assert(status == SDP_SUCCESS);

    // Generate PSWF
    sdp_generate_pswf(m, c, pswf_out, &status);
    assert(status == SDP_SUCCESS);
    const num_t* pswf = static_cast<const num_t*>(
        sdp_mem_data_const(pswf_out));
    for (int i = 0; i < size; i++)
    {
        double scale = expected[i];
        if (scale < 1e-15) scale = 1e-15;
        assert(fabs((val(pswf[i]) - expected[i]) / scale) < limit);
    }

    SDP_LOG_INFO("%s: Test passed", test_name);
}


void check_all(
        const char* test_name,
        const int m,
        const double c,
        const int64_t size,
        const double* expected
)
{
    std::string name = test_name;
    check<float>((name + " float").c_str(),
            SDP_MEM_FLOAT,
            m,
            c,
            size,
            expected,
            6e-8
    );
    check<double>((name + " double").c_str(),
            SDP_MEM_DOUBLE,
            m,
            c,
            size,
            expected,
            6e-13
    );
    check<std::complex<float> >((name + " complex float").c_str(),
            SDP_MEM_COMPLEX_FLOAT,
            m,
            c,
            size,
            expected,
            6e-8
    );
    check<std::complex<double> >((name + " complex double").c_str(),
            SDP_MEM_COMPLEX_DOUBLE,
            m,
            c,
            size,
            expected,
            6e-13
    );
}

const double ref_0_8_16[] = {
    0., 0.027739141134746, 0.091191127759381,
    0.209044784141754, 0.382566549867655, 0.5914940612484,
    0.795355148039012, 0.944953307469797, 0.999999999999999,
    0.944953307469797, 0.795355148039012, 0.5914940612484,
    0.382566549867655, 0.209044784141754, 0.091191127759381,
    0.027739141134746
};

const double ref_1_12_16[] = {
    0., 0.003294739717825, 0.02282344776114,
    0.085481700400665, 0.221815527395368, 0.439706432303182,
    0.699133301246467, 0.915341842238297, 0.999999999999999,
    0.915341842238297, 0.699133301246467, 0.439706432303182,
    0.221815527395368, 0.085481700400665, 0.02282344776114,
    0.003294739717825
};

const double ref_0_4_32[] = {
    0., 0.170790162306119, 0.229028505587468,
    0.293249007653473, 0.362375107129104, 0.435092139714442,
    0.509883171680941, 0.585074290491016, 0.658887707928077,
    0.729500645949425, 0.795107677241523, 0.853983996619878,
    0.904547016598184, 0.94541371587235, 0.975451322633679,
    0.99381917932372, 1., 0.99381917932372,
    0.975451322633679, 0.94541371587235, 0.904547016598184,
    0.853983996619878, 0.795107677241523, 0.729500645949425,
    0.658887707928077, 0.585074290491016, 0.509883171680941,
    0.435092139714442, 0.362375107129104, 0.293249007653473,
    0.229028505587468, 0.170790162306119
};


int main()
{
    check_all("PSWF(0, 8) size 16", 0, 8, 16, ref_0_8_16);
    check_all("PSWF(1, 12) size 16", 1, 12, 16, ref_1_12_16);
    check_all("PSWF(0, 4) size 32", 0, 4, 32, ref_0_4_32);
    return 0;
}
