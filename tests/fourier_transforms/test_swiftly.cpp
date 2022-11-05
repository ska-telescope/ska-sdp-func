/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ska-sdp-func/fourier_transforms/sdp_swiftly.h"
#include "ska-sdp-func/utility/sdp_mem_view.h"
#include "ska-sdp-func/utility/sdp_logging.h"

#include <complex>
#include <assert.h>
#include <string>
#include <cstring>
#include <vector>

void test_facet_to_subgrid_basic()
{
    // Basic 1D test: Source in the middle of the image should result
    // in subgrid filled with ones.

    // Instantiate SwiFTly
    sdp_Error status = SDP_SUCCESS;
    const int64_t image_size = 1024;
    const int64_t xM_size = 256;
    const int64_t yN_size = 512;
    const int64_t yB_size = 416;
    const int64_t xM_yN_size = (xM_size * yN_size) / image_size;
    const int64_t facet_off_step = image_size / xM_size;
    const int64_t sg_off_step = image_size / yN_size;
    sdp_SwiFTly *swiftly = sdp_swiftly_create(
        image_size, yN_size, xM_size, 13.5625, &status);
    assert(!status);

    // Make facet data + output region
    sdp_Mem *facet = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &yB_size, &status);
    sdp_Mem *prepared = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &yN_size, &status);
    sdp_Mem *contrib = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &xM_yN_size, &status);
    sdp_Mem *subgrid_image = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &xM_size, &status);
    sdp_MemViewCpu<std::complex<double>, 1> fct;
    sdp_mem_check_and_view(facet, &fct, &status);
    assert(!status);

    // Test different facet offsets
    int64_t facet_off;
    for (facet_off = -5*facet_off_step;
         facet_off <= 5*facet_off_step;
         facet_off += facet_off_step) {

        // Initialise facet with a single 1 in the overall image centre
        int64_t i;
        for (i = 0; i < yB_size; i++) {
            fct(i) = 0.0;
        }
        fct(yB_size / 2 - facet_off) = image_size;

        // Check different subgrid offsets
        int64_t sg_off;
        for (sg_off = 0;
             sg_off < image_size;
             sg_off += sg_off_step) {

            // Extract a single subgrid
            sdp_swiftly_prepare_facet(swiftly, facet, prepared, facet_off, &status);
            sdp_swiftly_extract_from_facet(swiftly, prepared, contrib, sg_off, &status);
            memset(sdp_mem_data(subgrid_image), 0, xM_size * sizeof(std::complex<double>));
            sdp_swiftly_add_to_subgrid(swiftly, contrib, subgrid_image, facet_off, &status);
            sdp_swiftly_finish_subgrid_inplace(swiftly, subgrid_image, sg_off, &status);
            sdp_MemViewCpu<std::complex<double>, 1> out;
            sdp_mem_check_and_view(subgrid_image, &out, &status);
            assert(!status);

            // Result should be precisely ones
            for (i = 0; i < out.shape[0]; i++) {
                assert(std::abs(out(i) - std::complex<double>(1)) < 1e-13);
            }
        }
    }

    sdp_mem_free(facet);
    sdp_mem_free(prepared);
    sdp_mem_free(contrib);
    sdp_mem_free(subgrid_image);
    sdp_swiftly_free(swiftly);

    SDP_LOG_INFO("test_facet_to_subgrid_basic: Test passed");
}

void test_facet_to_subgrid_dft()
{
    // General 1D test: Check that sources at arbitrary locations
    // produce result that matches direct Fourier transform evaluation
    // to a certain precision.

    // Instantiate SwiFTly
    sdp_Error status = SDP_SUCCESS;
    const int64_t image_size = 1024;
    const int64_t xM_size = 256;
    const int64_t yN_size = 512;
    const int64_t yB_size = 416;
    const int64_t xA_size = 228;
    const int64_t xM_yN_size = (xM_size * yN_size) / image_size;
    const int64_t facet_off_step = image_size / xM_size;
    const int64_t sg_off_step = image_size / yN_size;
    sdp_SwiFTly *swiftly = sdp_swiftly_create(
        image_size, yN_size, xM_size, 14.75, &status);
    assert(!status);

    // Make facet data + output region
    sdp_Mem *facet = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &yB_size, &status);
    sdp_Mem *prepared = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &yN_size, &status);
    sdp_Mem *contrib = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &xM_yN_size, &status);
    sdp_Mem *subgrid_image = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE, SDP_MEM_CPU, 1, &xM_size, &status);
    sdp_MemViewCpu<std::complex<double>, 1> fct;
    sdp_mem_check_and_view(facet, &fct, &status);
    assert(!status);

    std::vector<std::vector<int64_t> > source_pos_list = {
        { 0 }, { 1 }, { -3 }, { 5 }, { 20, 5, -4 },
        { -yB_size }, { yB_size }, { yB_size, 4, -yB_size }  // clamped below
    };

    // Test different source positions
    for (std::vector<int64_t> sources : source_pos_list) {

        // Test different facet offsets
        double ssum = 0; int64_t count = 0;
        int64_t facet_off;
        for (facet_off = -50*facet_off_step;
             facet_off <= 50*facet_off_step;
             facet_off += 10*facet_off_step) {

            // Initialise facet with a single 1 at given position
            int64_t i;
            for (i = 0; i < yB_size; i++) {
                fct(i) = 0.0;
            }

            // Determine (clamped) source position, add to facet
            std::vector<int64_t>::size_type isrc;
            for (isrc = 0; isrc < sources.size(); isrc++) {
                int64_t source_y = sources[isrc];
                int64_t min_y = facet_off - yB_size / 2;
                if (source_y < min_y)
                    source_y = min_y;
                if (source_y >= min_y + yB_size)
                    source_y = min_y + yB_size - 1;

                // Add + write back. We divide by number of sources
                // here to normalise the error a bit.
                fct(source_y + yB_size / 2 - facet_off) =
                    double(image_size) / sources.size();
                sources[isrc] = source_y;
            }

            // Check different subgrid offsets
            int64_t sg_off;
            for (sg_off = -sg_off_step * 10;
                 sg_off < sg_off_step * 10;
                 sg_off += sg_off_step) {

                // Extract a single subgrid
                sdp_swiftly_prepare_facet(swiftly, facet, prepared, facet_off, &status);
                sdp_swiftly_extract_from_facet(swiftly, prepared, contrib, sg_off, &status);
                memset(sdp_mem_data(subgrid_image), 0, xM_size * sizeof(std::complex<double>));
                sdp_swiftly_add_to_subgrid(swiftly, contrib, subgrid_image, facet_off, &status);
                sdp_swiftly_finish_subgrid_inplace(swiftly, subgrid_image, sg_off, &status);
                sdp_MemViewCpu<std::complex<double>, 1> out;
                sdp_mem_check_and_view(subgrid_image, &out, &status);
                assert(!status);

                // Result should match DFT to good precision
                for (i = 0; i < xA_size; i++) {
                    int64_t pos = i + sg_off - xA_size / 2;
                    int64_t iM = i - xA_size / 2 + xM_size / 2;

                    // Sum up DFT over sources
                    std::complex<double> expected = { 0, 0 };
                    for (isrc = 0; isrc < sources.size(); isrc++) {
                        int64_t source_y = sources[isrc];
                        double phase = (2 * M_PI / image_size) * source_y * pos;
                        expected += std::complex<double>(cos(phase), sin(phase));
                    }
                    expected /= sources.size(); // normalisation, see above

                    // Compare with actual
                    double diff = std::abs(out(iM) - expected);
                    ssum += diff * diff; count++;
                    assert(diff < 4e-6);
                }
            }
        }

        // RMSE should be significantly better than worst-case
        double rmse = sqrt(ssum / count);
        assert(rmse < 3e-7);
        SDP_LOG_INFO("test_facet_to_subgrid_dft: RMSE %g", rmse);
    }

    sdp_mem_free(facet);
    sdp_mem_free(prepared);
    sdp_mem_free(contrib);
    sdp_mem_free(subgrid_image);
    sdp_swiftly_free(swiftly);

    SDP_LOG_INFO("test_facet_to_subgrid_dft: Test passed");
}

int main()
{
    test_facet_to_subgrid_basic();
    test_facet_to_subgrid_dft();
    return 0;
}
