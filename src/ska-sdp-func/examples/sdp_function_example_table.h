/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_FUNCTION_EXAMPLE_TABLE_H_
#define SKA_SDP_PROC_FUNC_FUNCTION_EXAMPLE_TABLE_H_

/**
 * @file sdp_function_example_table.h
 */

#include "ska-sdp-func/utility/sdp_table.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup ex_table_func
 * @{
 */

/**
 * @brief Demonstrate a function that uses a sdp_Table.
 *
 * @param table Handle to table.
 * @param status Error status.
 */
void sdp_function_example_table(
        sdp_Table* table,
        sdp_Error* status
);

/** @} */ /* End group ex_table_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
