/* See the LICENSE file at the top-level directory of this distribution. */

#include "func/function_example_a/sdp_function_example_a.h"
#include "utility/sdp_device_wrapper.h"
#include "utility/sdp_logging.h"

struct sdp_FunctionExampleA
{
    int a;
    int b;
    float c;
    float* workarea;
};

sdp_FunctionExampleA* sdp_function_example_a_create_plan(
        int a,
        int b,
        float c,
        sdp_Error* status
)
{
    if (*status) return NULL;
    if (a == 10)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Error creating sdp_FunctionExampleA (parameter 'a' cannot be 10)");
        return NULL;
    }
    sdp_FunctionExampleA* plan = (sdp_FunctionExampleA*) calloc(1, sizeof(sdp_FunctionExampleA));
    plan->a = a;
    plan->b = b;
    plan->c = c;
    plan->workarea = (float*) calloc(a * b, sizeof(float));
    SDP_LOG_INFO("Created sdp_FunctionExampleA");
    return plan;
}


void sdp_function_example_a_exec(
    sdp_FunctionExampleA* plan,
    sdp_Mem *output,
    sdp_Error* status
)
{
    if (*status || !plan) return;

    const sdp_MemLocation location = sdp_mem_location(output);
    if (location != SDP_MEM_CPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("Unsupported platform (GPU)");
        return;
    }
    const int64_t num_elements = sdp_mem_num_elements(output);
    if (num_elements < (plan->a * plan->b))
    {
        *status = SDP_ERR_MEM_ALLOC_FAILURE;
        SDP_LOG_ERROR("Output is not big enough");
        return;
    }
    const sdp_MemType type = sdp_mem_type(output);
    if (type != SDP_MEM_FLOAT)
    {
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Output must be FP32");
        return;
    }

    printf("Running processing function A: %d, %d, %.3f\n",
            plan->a, plan->b, plan->c);

    float* output_pointer = (float*) sdp_mem_data(output);
    for (int f = 0; f < plan->a * plan->b; f++)
    {
        float ftemp = 0.f;
        plan->workarea[f] = plan->c * f;
        for (int i = 0; i<f; i++)
        {
            ftemp = ftemp + plan->workarea[i];
        }
        output_pointer[f] = ftemp;
    }
}

void sdp_function_example_a_free_plan(sdp_FunctionExampleA* plan)
{
    if (!plan) return;
    free(plan->workarea);
    free(plan);
    SDP_LOG_INFO("Destroyed sdp_FunctionExampleA");
}
