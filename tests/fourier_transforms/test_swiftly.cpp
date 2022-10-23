
/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ska-sdp-func/fourier_transforms/sdp_swiftly.h"
#include "ska-sdp-func/fourier_transforms/sdp_mem_utils.h"
#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_mem.h"

#include <complex>
#include <assert.h>
#include <string>
#include <cstring>

int main()
{

    // Instantiate SwiFTly
    sdp_Error status = SDP_SUCCESS;
    const int64_t image_size = 1024;
    const int64_t xM_size = 256;
    const int64_t yN_size = 512;
    sdp_SwiFTly *swiftly = sdp_swiftly_create(
        image_size, yN_size, xM_size, 13.5625, &status);
    assert(!status);

    // Make facet data + output region
    const int64_t yB_size = 416;
    sdp_Mem *facet = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &yB_size, &status);
    sdp_Mem *prepared = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &yN_size, &status);
    sdp_CpuVec<std::complex<double>, 1> fct;
    sdp_mem_checked_get_vec(facet, &fct, &status);
    assert(!status);
    int64_t i;
    for (i = 0; i < fct.shape[0]; i++) {
        fct[i] = 1.0;
    }
    
    // Check facet preparation
    sdp_swiftly_prepare_facet(swiftly, facet, prepared, 0, &status);
    assert(!status);
    
    sdp_CpuVec<std::complex<double>, 1> out;
    sdp_mem_checked_get_vec(prepared, &out, &status);
    assert(!status);
    for (i = 0; i < out.shape[0]; i++) {
        printf("%g%+gj ", out[i].real(), out[i].imag());
    }
    
    sdp_swiftly_free(swiftly);
    return 0;
}
