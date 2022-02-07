/* See the LICENSE file at the top-level directory of this distribution. */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "func/vector/sdp_vector_add.h"
#include "utility/sdp_logging.h"
#include "utility/sdp_mem.h"

template<typename T>
void check_results(
        const char* test_name,
        const sdp_Mem* a,
        const sdp_Mem* b,
        const sdp_Mem* out,
        sdp_Error* status)
{
    if (*status)
    {
        SDP_LOG_ERROR("%s: Test failed (error signalled)", test_name);
        return;
    }
    const T* a_data = (const T*)sdp_mem_data_const(a);
    const T* b_data = (const T*)sdp_mem_data_const(b);
    const T* out_data = (const T*)sdp_mem_data_const(out);
    const int64_t num_elements = sdp_mem_num_elements(a);
    for (int64_t i = 0; i < num_elements; ++i)
    {
        const T expected = a_data[i] + b_data[i];
        assert(fabs(out_data[i] - expected) < 1e-5);
    }
    SDP_LOG_INFO("%s: Test passed", test_name);
}


int main()
{
    // Generate some test data.
    // Use C++ vectors as an example of externally-managed memory.
    const int64_t num_elements = 10;
    std::vector<double> vec_a(num_elements);
    std::vector<double> vec_b(num_elements);
    std::vector<double> vec_out(num_elements);
    for (int64_t i = 0; i < num_elements; ++i)
    {
        vec_a[i] = rand() / (double)RAND_MAX;
        vec_b[i] = rand() / (double)RAND_MAX;
    }

    // Wrap pointers to externally-managed memory.
    const sdp_MemType type = SDP_MEM_DOUBLE;
    sdp_Error status = SDP_SUCCESS;
    sdp_Mem* a = sdp_mem_create_wrapper(
            vec_a.data(), type, SDP_MEM_CPU, 1, &num_elements, 0, &status);
    sdp_Mem* b = sdp_mem_create_wrapper(
            vec_b.data(), type, SDP_MEM_CPU, 1, &num_elements, 0, &status);
    sdp_Mem* out1 = sdp_mem_create_wrapper(
            vec_out.data(), type, SDP_MEM_CPU, 1, &num_elements, 0, &status);

    // Call CPU version of processing function.
    sdp_vector_add(a, b, out1, &status);

    // Check results.
    check_results<double>("CPU vector add", a, b, out1, &status);
    sdp_mem_free(out1);

#ifdef SDP_HAVE_CUDA
    // Copy test data to GPU.
    sdp_Mem* a_gpu = sdp_mem_create_copy(a, SDP_MEM_GPU, &status);
    sdp_Mem* b_gpu = sdp_mem_create_copy(b, SDP_MEM_GPU, &status);
    sdp_Mem* out_gpu = sdp_mem_create(type, SDP_MEM_GPU,
            1, &num_elements, &status);
    sdp_mem_clear_contents(out_gpu, &status);

    // Call GPU version of processing function.
    sdp_vector_add(a_gpu, b_gpu, out_gpu, &status);
    sdp_mem_free(a_gpu);
    sdp_mem_free(b_gpu);

    // Copy GPU output back to host for checking.
    sdp_Mem* out2 = sdp_mem_create_copy(out_gpu, SDP_MEM_CPU, &status);
    sdp_mem_free(out_gpu);

    // Check results.
    check_results<double>("GPU vector add", a, b, out2, &status);
    sdp_mem_free(out2);
#endif

    sdp_mem_free(a);
    sdp_mem_free(b);
    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
