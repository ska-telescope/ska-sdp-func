#include "sdp_element.h"

#include <map>
#include <string>

namespace {
std::map<std::string,
        std::tuple<make_function_t, destroy_function_t,
        evaluate_function_t> >&get_sdp_element_function_map()
{
    static std::map<std::string, std::tuple<make_function_t, destroy_function_t,
            evaluate_function_t> >
    function_map;
    return function_map;
}
}


int sdp_register_element(
        const char* name,
        const make_function_t make_function,
        const destroy_function_t destroy_function,
        const evaluate_function_t evaluate_function
)
{
    std::map<std::string, std::tuple<make_function_t, destroy_function_t,
            evaluate_function_t> >& function_map =
            get_sdp_element_function_map();
    function_map[name] =
            std::make_tuple(make_function, destroy_function, evaluate_function);
    return 0;
}


sdp_Error sdp_get_element_functions(
        const char* name,
        make_function_t& make_function,
        destroy_function_t& destroy_function,
        evaluate_function_t& evaluate_function
)
{
    std::map<std::string, std::tuple<make_function_t, destroy_function_t,
            evaluate_function_t> >& function_map =
            get_sdp_element_function_map();

    if (!function_map.count(name)) return SDP_ERR_INVALID_ARGUMENT;

    std::tie(make_function, destroy_function, evaluate_function) =
            function_map[name];
    return SDP_SUCCESS;
}
