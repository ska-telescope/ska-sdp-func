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
    sdp_SwiFTly* swiftly = sdp_swiftly_create(
            image_size, yN_size, xM_size, 13.5625, &status
    );
    assert(!status);

    // Make facet data + output region
    int64_t facet_size[] = { 1, yB_size };
    sdp_Mem* facet = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            facet_size,
            &status
    );
    int64_t prepared_size[] = { 1, yN_size };
    sdp_Mem* prepared = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            prepared_size,
            &status
    );
    int64_t contrib_size[] = { 1, xM_yN_size };
    sdp_Mem* contrib = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            contrib_size,
            &status
    );
    int64_t subgrid_size[] = { 1, xM_size };
    sdp_Mem* subgrid_image = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            subgrid_size,
            &status
    );
    sdp_MemViewCpu<std::complex<double>, 2> fct;
    sdp_mem_check_and_view(facet, &fct, &status);
    assert(!status);

    // Test different facet offsets
    int64_t facet_off;
    for (facet_off = -5 * facet_off_step;
            facet_off <= 5 * facet_off_step;
            facet_off += facet_off_step)
    {
        // Initialise facet with a single 1 in the overall image centre
        int64_t i;
        for (i = 0; i < yB_size; i++)
        {
            fct(0, i) = 0.0;
        }
        fct(0, yB_size / 2 - facet_off) = image_size;

        // Check different subgrid offsets
        int64_t sg_off;
        for (sg_off = 0;
                sg_off < image_size;
                sg_off += sg_off_step)
        {
            // Extract a single subgrid
            sdp_swiftly_prepare_facet(swiftly,
                    facet,
                    prepared,
                    facet_off,
                    &status
            );
            sdp_swiftly_extract_from_facet(swiftly,
                    prepared,
                    contrib,
                    sg_off,
                    &status
            );
            memset(sdp_mem_data(subgrid_image), 0,
                    xM_size * sizeof(std::complex<double>)
            );
            sdp_swiftly_add_to_subgrid(swiftly,
                    contrib,
                    subgrid_image,
                    facet_off,
                    &status
            );
            sdp_swiftly_finish_subgrid_inplace(swiftly,
                    subgrid_image,
                    sg_off,
                    &status
            );
            sdp_MemViewCpu<std::complex<double>, 2> out;
            sdp_mem_check_and_view(subgrid_image, &out, &status);
            assert(!status);

            // Result should be precisely ones
            for (i = 0; i < out.shape[1]; i++)
            {
                assert(std::abs(out(0, i) - std::complex<double>(1)) < 1e-13);
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
    sdp_SwiFTly* swiftly = sdp_swiftly_create(
            image_size, yN_size, xM_size, 14.75, &status
    );
    assert(!status);

    // Make facet data + output region
    int64_t facet_size[] = { 1, yB_size };
    sdp_Mem* facet = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            facet_size,
            &status
    );
    int64_t prepared_size[] = { 1, yN_size };
    sdp_Mem* prepared = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            prepared_size,
            &status
    );
    int64_t contrib_size[] = { 1, xM_yN_size };
    sdp_Mem* contrib = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            contrib_size,
            &status
    );
    int64_t subgrid_size[] = { 1, xM_size };
    sdp_Mem* subgrid_image = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            subgrid_size,
            &status
    );
    sdp_MemViewCpu<std::complex<double>, 2> fct;
    sdp_mem_check_and_view(facet, &fct, &status);
    assert(!status);

    std::vector<std::vector<int64_t> > source_pos_list = {
        { 0 }, { 1 }, { -3 }, { 5 }, { 20, 5, -4 },
        { -yB_size }, { yB_size }, { yB_size, 4, -yB_size }  // clamped below
    };

    // Test different source positions
    for (std::vector<int64_t> sources : source_pos_list)
    {
        // Test different facet offsets
        double ssum = 0; int64_t count = 0;
        int64_t facet_off;
        for (facet_off = -50 * facet_off_step;
                facet_off <= 50 * facet_off_step;
                facet_off += 10 * facet_off_step)
        {
            // Initialise facet with zeroes
            int64_t i;
            for (i = 0; i < yB_size; i++)
            {
                fct(0, i) = 0.0;
            }

            // Determine (clamped) source position, add to facet
            std::vector<int64_t>::size_type isrc;
            std::vector<int64_t> new_sources(sources.size());
            for (isrc = 0; isrc < sources.size(); isrc++)
            {
                int64_t source_y = sources[isrc];
                int64_t min_y = facet_off - yB_size / 2;
                if (source_y < min_y)
                    source_y = min_y;
                if (source_y >= min_y + yB_size)
                    source_y = min_y + yB_size - 1;

                // Add + write back. We divide by number of sources
                // here to normalise the error a bit.
                fct(0, source_y + yB_size / 2 - facet_off) =
                        double(image_size) / sources.size();
                new_sources[isrc] = source_y;
            }

            // Check different subgrid offsets
            int64_t sg_off;
            for (sg_off = -sg_off_step * 10;
                    sg_off < sg_off_step * 10;
                    sg_off += sg_off_step)
            {
                // Extract a single subgrid
                sdp_swiftly_prepare_facet(swiftly,
                        facet,
                        prepared,
                        facet_off,
                        &status
                );
                sdp_swiftly_extract_from_facet(swiftly,
                        prepared,
                        contrib,
                        sg_off,
                        &status
                );
                memset(sdp_mem_data(subgrid_image), 0,
                        xM_size * sizeof(std::complex<double>)
                );
                sdp_swiftly_add_to_subgrid(swiftly,
                        contrib,
                        subgrid_image,
                        facet_off,
                        &status
                );
                sdp_swiftly_finish_subgrid_inplace(swiftly,
                        subgrid_image,
                        sg_off,
                        &status
                );
                sdp_MemViewCpu<std::complex<double>, 2> out;
                sdp_mem_check_and_view(subgrid_image, &out, &status);
                assert(!status);

                // Result should match DFT to good precision
                for (i = 0; i < xA_size; i++)
                {
                    int64_t pos = i + sg_off - xA_size / 2;
                    int64_t iM = i - xA_size / 2 + xM_size / 2;

                    // Sum up DFT over sources
                    std::complex<double> expected = { 0, 0 };
                    for (isrc = 0; isrc < sources.size(); isrc++)
                    {
                        int64_t source_y = new_sources[isrc];
                        double phase = (2 * M_PI / image_size) * source_y * pos;
                        expected +=
                                std::complex<double>(cos(phase), sin(phase));
                    }
                    expected /= sources.size(); // normalisation, see above

                    // Compare with actual
                    double diff = std::abs(out(0, iM) - expected);
                    ssum += diff * diff; count++;
                    if (diff >= 4e-6)
                    {
                        printf("%g%+g != %g%+g\n",
                                out(0, iM).real(), out(0, iM).imag(),
                                expected.real(), expected.imag()
                        );
                    }
                    assert(diff < 4e-6);
                }
            }
        }

        // RMSE should be significantly better than worst-case
        double rmse = sqrt(ssum / count);
        SDP_LOG_INFO("test_facet_to_subgrid_dft: RMSE %g (%d samples)",
                rmse,
                count
        );
        assert(rmse < 8e-7);
    }

    sdp_mem_free(facet);
    sdp_mem_free(prepared);
    sdp_mem_free(contrib);
    sdp_mem_free(subgrid_image);
    sdp_swiftly_free(swiftly);

    SDP_LOG_INFO("test_facet_to_subgrid_dft: Test passed");
}


static sdp_Mem* sdp_mem_transpose(
        sdp_Mem* mem,
        int32_t dim0,
        int32_t dim1,
        sdp_Error* status
)
{
    if (*status) return NULL;

    const int32_t ndims = sdp_mem_num_dims(mem);
    if (dim0 >= ndims || dim1 >= ndims)
    {
        SDP_LOG_ERROR("sdp_mem_transpose: Cannot transpose dimensions "
                "%d and %d, object only has %d dimensions!",
                dim0, dim1, ndims
        );
        *status = SDP_ERR_INVALID_ARGUMENT;
        return NULL;
    }

    // Create new shape + strides
    std::vector<int64_t> shape(ndims), stride(ndims);
    int64_t i;
    for (i = 0; i < sdp_mem_num_dims(mem); i++)
    {
        if (i == dim0)
        {
            shape[i] = sdp_mem_shape_dim(mem, dim1);
            stride[i] = sdp_mem_stride_bytes_dim(mem, dim1);
        }
        else if (i == dim1)
        {
            shape[i] = sdp_mem_shape_dim(mem, dim0);
            stride[i] = sdp_mem_stride_bytes_dim(mem, dim0);
        }
        else
        {
            shape[i] = sdp_mem_shape_dim(mem, i);
            stride[i] = sdp_mem_stride_bytes_dim(mem, i);
        }
    }

    return sdp_mem_create_wrapper(
            sdp_mem_data(mem),
            sdp_mem_type(mem),
            sdp_mem_location(mem),
            ndims,
            &shape[0],
            &stride[0],
            status
    );
}


void check_facet_to_subgrid_dft_2d(
        sdp_SwiFTly* swiftly,
        sdp_Mem* facet,
        const std::vector<std::pair<int64_t, int64_t> > &sources,
        int64_t facet_off0,
        int64_t facet_off1,
        int64_t sg_off0,
        int64_t sg_off1,
        double* ssum,
        int64_t* count
)
{
    sdp_Error status = SDP_SUCCESS;
    const int64_t image_size = sdp_swiftly_get_image_size(swiftly);
    const int64_t xM_size = sdp_swiftly_get_subgrid_size(swiftly);
    const int64_t yN_size = sdp_swiftly_get_facet_size(swiftly);
    const int64_t yB_size = 416;
    const int64_t xA_size = 228;
    const int64_t xM_yN_size = (xM_size * yN_size) / image_size;

    // Allocate work buffers
    sdp_Mem* facet_t = sdp_mem_transpose(facet, 0, 1, &status);
    int64_t prepared0_size[] = { yN_size, yB_size };
    sdp_Mem* prepared0 = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            prepared0_size,
            &status
    );
    sdp_Mem* prepared0_t = sdp_mem_transpose(prepared0, 0, 1, &status);
    int64_t contrib0_size[] = { xM_yN_size, yB_size };
    sdp_Mem* contrib0 = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            contrib0_size,
            &status
    );
    sdp_Mem* contrib0_t = sdp_mem_transpose(contrib0, 0, 1, &status);
    int64_t prepared1_size[] = { xM_yN_size, yN_size };
    sdp_Mem* prepared1 = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            prepared1_size,
            &status
    );
    int64_t contrib1_size[] = { xM_yN_size, xM_yN_size };
    sdp_Mem* contrib1 = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            contrib1_size,
            &status
    );
    sdp_Mem* contrib1_t = sdp_mem_transpose(contrib1, 0, 1, &status);
    int64_t subgrid0_size[] = { xM_size, xM_yN_size };
    sdp_Mem* subgrid0_image = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            subgrid0_size,
            &status
    );
    sdp_Mem* subgrid0_image_t =
            sdp_mem_transpose(subgrid0_image, 0, 1, &status);
    int64_t subgrid_size[] = { xM_size, xM_size };
    sdp_Mem* subgrid_image = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            subgrid_size,
            &status
    );
    sdp_Mem* subgrid_image_t = sdp_mem_transpose(subgrid_image, 0, 1, &status);
    sdp_Mem* subgrid_image_copy = sdp_mem_create_copy(subgrid_image,
            sdp_mem_location(subgrid_image),
            &status
    );

    // Extract a single subgrid
    sdp_swiftly_prepare_facet(swiftly, facet_t, prepared0_t, facet_off0,
            &status
    );
    sdp_swiftly_extract_from_facet(swiftly,
            prepared0_t,
            contrib0_t,
            sg_off0,
            &status
    );
    sdp_swiftly_prepare_facet(swiftly, contrib0, prepared1, facet_off1,
            &status
    );
    sdp_swiftly_extract_from_facet(swiftly,
            prepared1,
            contrib1,
            sg_off1,
            &status
    );

    // Add to subgrid (using both 1D and 2D methods)
    memset(sdp_mem_data(subgrid0_image), 0,
            xM_size * xM_yN_size * sizeof(std::complex<double>)
    );
    sdp_swiftly_add_to_subgrid(swiftly,
            contrib1_t,
            subgrid0_image_t,
            facet_off0,
            &status
    );
    memset(sdp_mem_data(subgrid_image), 0,
            xM_size * xM_size * sizeof(std::complex<double>)
    );
    sdp_swiftly_add_to_subgrid(swiftly,
            subgrid0_image,
            subgrid_image,
            facet_off1,
            &status
    );
    memset(sdp_mem_data(subgrid_image_copy), 0,
            xM_size * xM_size * sizeof(std::complex<double>)
    );
    sdp_swiftly_add_to_subgrid_2d(swiftly,
            contrib1,
            subgrid_image_copy,
            facet_off0,
            facet_off1,
            &status
    );

    // Check that both arrived at the same result
    int64_t i0, i1;
    sdp_MemViewCpu<std::complex<double>, 2> out, out_cp;
    sdp_mem_check_and_view(subgrid_image, &out, &status);
    sdp_mem_check_and_view(subgrid_image_copy, &out_cp, &status);
    for (i0 = 0; i0 < xM_size; i0++)
    {
        for (i1 = 0; i1 < xM_size; i1++)
        {
            double scale = abs(out(i0,i1));
            if (scale <= 1e-10) scale = 1e-10;
            assert (abs(out(i0,i1) - out_cp(i0,i1)) / scale < 1e-13);
        }
    }

    // Finish subgrid (using both 1D and 2D methods)
    sdp_mem_copy_contents(subgrid_image_copy, subgrid_image, 0, 0,
            sdp_mem_num_elements(subgrid_image), &status
    );
    sdp_swiftly_finish_subgrid_inplace(swiftly,
            subgrid_image_t,
            sg_off0,
            &status
    );
    sdp_swiftly_finish_subgrid_inplace(swiftly, subgrid_image, sg_off1,
            &status
    );
    sdp_swiftly_finish_subgrid_inplace_2d(swiftly,
            subgrid_image_copy,
            sg_off0,
            sg_off1,
            &status
    );
    assert(!status);

    // Again, check that both arrived at the same result
    for (i0 = 0; i0 < xM_size; i0++)
    {
        for (i1 = 0; i1 < xM_size; i1++)
        {
            assert(abs(out(i0,i1) - out_cp(i0,i1)) / abs(out(i0,i1)) < 1e-7);
        }
    }

    // Result should match DFT to good precision
    for (i0 = 0; i0 < xA_size; i0++)
    {
        for (i1 = 0; i1 < xA_size; i1++)
        {
            int64_t pos0 = i0 + sg_off0 - xA_size / 2;
            int64_t pos1 = i1 + sg_off1 - xA_size / 2;
            int64_t iM0 = i0 - xA_size / 2 + xM_size / 2;
            int64_t iM1 = i1 - xA_size / 2 + xM_size / 2;

            // Sum up DFT over sources
            std::complex<double> expected = { 0, 0 };
            std::vector<int64_t>::size_type isrc;
            for (isrc = 0; isrc < sources.size(); isrc++)
            {
                auto source = sources[isrc];
                double phase = (2 * M_PI / image_size) *
                        (source.first * pos0 + source.second * pos1);
                expected += std::complex<double>(cos(phase), sin(phase));
            }
            expected /= sources.size(); // normalisation, see above

            // Compare with actual
            double diff = std::abs(out(iM0, iM1) - expected);
            *ssum += diff * diff; (*count)++;
            assert(diff < 4e-6);
        }
    }

    // Free buffers
    sdp_mem_free(facet_t);
    sdp_mem_free(prepared0); sdp_mem_free(prepared0_t);
    sdp_mem_free(contrib0); sdp_mem_free(contrib0_t);
    sdp_mem_free(prepared1);
    sdp_mem_free(contrib1); sdp_mem_free(contrib1_t);
    sdp_mem_free(subgrid0_image); sdp_mem_free(subgrid0_image_t);
    sdp_mem_free(subgrid_image); sdp_mem_free(subgrid_image_t);
    sdp_mem_free(subgrid_image_copy);
}


void test_facet_to_subgrid_dft_2d()
{
    // General 2D test: Check that sources at arbitrary locations
    // produce result that matches direct Fourier transform evaluation
    // to a certain precision.

    // Instantiate SwiFTly
    sdp_Error status = SDP_SUCCESS;
    const int64_t image_size = 1024;
    const int64_t xM_size = 256;
    const int64_t yN_size = 512;
    const int64_t yB_size = 416;
    const int64_t facet_off_step = image_size / xM_size;
    const int64_t sg_off_step = image_size / yN_size;
    sdp_SwiFTly* swiftly = sdp_swiftly_create(
            image_size, yN_size, xM_size, 14.75, &status
    );
    assert(!status);

    // Make facet data + output regions for going through the
    // algorithm first on axis 0, then on axis 1. We create transposed
    // references to the buffers to pass to the routines later.
    int64_t facet_size[] = { yB_size, yB_size };
    sdp_Mem* facet = sdp_mem_create(SDP_MEM_COMPLEX_DOUBLE,
            SDP_MEM_CPU,
            2,
            facet_size,
            &status
    );
    sdp_MemViewCpu<std::complex<double>, 2> fct;
    sdp_mem_check_and_view(facet, &fct, &status);
    assert(!status);

    std::vector<std::vector<std::pair<int64_t, int64_t> > > source_pos_list = {
        { { 0, 0 } },
        { { 2, 1 } },
        { { -2, -3 } },
        { { 10, 5 } },
        { { 50, 20}, {2, 5}, {3, -4} },
        { { 0, -yB_size } },
        { { -yB_size, yB_size } },
        { { yB_size / 2, yB_size}, { 0, 4 }, {0, -yB_size } }  // clamped below
    };

    // Test different source positions
    for (auto sources : source_pos_list)
    {
        // Test different facet offsets
        double ssum = 0; int64_t count = 0;
        int64_t facet_off1;
        for (facet_off1 = -20 * facet_off_step;
                facet_off1 <= 20 * facet_off_step;
                facet_off1 += 10 * facet_off_step)
        {
            int64_t facet_off0 = facet_off1 * 2;

            // Initialise facet with zeroes
            int64_t i, j;
            for (i = 0; i < yB_size; i++)
            {
                for (j = 0; j < yB_size; j++)
                {
                    fct(i, j) = 0.0;
                }
            }

            // Determine (clamped) source position, add to facet
            std::vector<std::pair<int64_t, int64_t> >::size_type isrc;
            std::vector<std::pair<int64_t,
                    int64_t> > new_sources(sources.size());
            for (isrc = 0; isrc < sources.size(); isrc++)
            {
                int64_t source0 = sources[isrc].first;
                int64_t source1 = sources[isrc].second;
                int64_t min0 = facet_off0 - yB_size / 2;
                if (source0 < min0)
                    source0 = min0;
                if (source0 >= min0 + yB_size)
                    source0 = min0 + yB_size - 1;
                int64_t min1 = facet_off1 - yB_size / 2;
                if (source1 < min1)
                    source1 = min1;
                if (source1 >= min1 + yB_size)
                    source1 = min1 + yB_size - 1;

                // Add + write back. We divide by number of sources
                // here to normalise the error a bit.
                fct(source0 + yB_size / 2 - facet_off0,
                        source1 + yB_size / 2 - facet_off1
                ) =
                        double(image_size * image_size) / sources.size();
                new_sources[isrc] =
                        std::pair<int64_t, int64_t>(source0, source1);
            }

            // Check different subgrid offsets
            int64_t sg_off1;
            for (sg_off1 = -sg_off_step * 2;
                    sg_off1 <= sg_off_step * 2;
                    sg_off1 += sg_off_step)
            {
                int64_t sg_off0 = -sg_off1 * 3 + sg_off_step;

                // Check that things work out for those parameters,
                // capturing accuracy statistics.
                check_facet_to_subgrid_dft_2d(
                        swiftly, facet, new_sources,
                        facet_off0, facet_off1, sg_off0, sg_off1,
                        &ssum, &count
                );
            }
        }

        // RMSE should be significantly better than worst-case
        double rmse = sqrt(ssum / count);
        SDP_LOG_INFO("test_facet_to_subgrid_dft: RMSE %g (%d samples)",
                rmse,
                count
        );
        assert(rmse < 8e-7);
    }

    sdp_mem_free(facet);
    sdp_swiftly_free(swiftly);

    SDP_LOG_INFO("test_facet_to_subgrid_dft_2d: Test passed");
}


int main()
{
    test_facet_to_subgrid_basic();
    test_facet_to_subgrid_dft();
    test_facet_to_subgrid_dft_2d();
    return 0;
}
