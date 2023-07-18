/* See the LICENSE file at the top-level directory of this distribution. */

#include <cstdlib>

#include "ska-sdp-func/examples/sdp_function_example_a.h"
#include "ska-sdp-func/utility/sdp_device_wrapper.h"
#include "ska-sdp-func/utility/sdp_logging.h"

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
        *status = SDP_ERR_INVALID_ARGUMENT;
        SDP_LOG_ERROR("Error creating sdp_FunctionExampleA "
                "(parameter 'a' cannot be 10)"
        );
        return NULL;
    }
    sdp_FunctionExampleA* plan = (sdp_FunctionExampleA*) calloc(
            1, sizeof(sdp_FunctionExampleA)
    );
    plan->a = a;
    plan->b = b;
    plan->c = c;
    plan->workarea = (float*) calloc(a * b, sizeof(float));
    SDP_LOG_INFO("Created sdp_FunctionExampleA");
    return plan;
}


void sdp_function_example_a_exec(
        sdp_FunctionExampleA* plan,
        sdp_Mem* output,
        sdp_Error* status
)
{
    if (*status || !plan) return;
    if (sdp_mem_type(output) != SDP_MEM_FLOAT)
    {
        *status = SDP_ERR_DATA_TYPE;
        SDP_LOG_ERROR("Output data type must be FP32");
        return;
    }
    if (sdp_mem_location(output) != SDP_MEM_CPU)
    {
        *status = SDP_ERR_MEM_LOCATION;
        SDP_LOG_ERROR("GPU platform not supported");
        return;
    }
    if (sdp_mem_is_read_only(output))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output is not writable");
        return;
    }
    const int64_t num_elements = sdp_mem_num_elements(output);
    if (num_elements < (plan->a * plan->b))
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Output is not big enough");
        return;
    }

    SDP_LOG_INFO("Running processing function A: %d, %d, %.3f",
            plan->a, plan->b, plan->c
    );

    float* output_pointer = (float*) sdp_mem_data(output);
    for (int f = 0; f < num_elements; f++)
    {
        float ftemp = 0.f;
        plan->workarea[f] = plan->c * f;
        for (int i = 0; i < f; i++)
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
