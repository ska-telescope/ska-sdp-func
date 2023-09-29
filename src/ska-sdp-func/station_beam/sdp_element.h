#include "../utility/sdp_errors.h"
#include "../utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* sdp_element_t;

typedef sdp_element_t (*make_function_t)(const char* name);

typedef void (*destroy_function_t)(sdp_element_t element_response);

typedef void (*evaluate_function_t)(
        sdp_element_t element_response,
        int num_points,
        const sdp_Mem* theta,
        const sdp_Mem* phi,
        sdp_Mem* response
);

[[gnu::visibility("default")]] int sdp_register_element(
        const char* name,
        const make_function_t make_function,
        const destroy_function_t destroy_function,
        const evaluate_function_t evaluate_function
);

[[gnu::visibility("default")]] sdp_Error sdp_get_element_functions(
        const char* name,
        make_function_t& make_function,
        destroy_function_t& destroy_function,
        evaluate_function_t& evaluate_function
);

#ifdef __cplusplus
}
#endif
