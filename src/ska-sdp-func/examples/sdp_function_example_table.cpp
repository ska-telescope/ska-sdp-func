/* See the LICENSE file at the top-level directory of this distribution. */

#include "ska-sdp-func/examples/sdp_function_example_table.h"
#include "ska-sdp-func/utility/sdp_logging.h"


void sdp_function_example_table(
        sdp_Table* dataset,
        sdp_Error* status
)
{
    if (*status || !dataset) return;
    sdp_Mem* vis = sdp_table_get_column(dataset, "vis");
    sdp_Mem* uvw = sdp_table_get_column(dataset, "uvw");
    if (!vis || !uvw)
    {
        *status = SDP_ERR_RUNTIME;
        SDP_LOG_ERROR("Supplied dataset must contain 'vis' and 'uvw' arrays");
        return;
    }
    const int64_t num_times     = sdp_mem_shape_dim(uvw, 0);
    const int64_t num_baselines = sdp_mem_shape_dim(vis, 1);
    double* uvw_data = (double*) sdp_mem_data(uvw);
    for (int64_t t = 0; t < num_times; ++t)
    {
        for (int64_t b = 0; b < num_baselines; ++b)
        {
            uvw_data[3 * (t * num_baselines + b) + 0] = 1 * t + b / 10.0;
            uvw_data[3 * (t * num_baselines + b) + 1] = 10 * t + b / 10.0;
            uvw_data[3 * (t * num_baselines + b) + 2] = 100 * t + b / 10.0;
        }
    }
}
