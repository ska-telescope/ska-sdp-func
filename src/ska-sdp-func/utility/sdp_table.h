/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_TABLE_H_
#define SKA_SDP_PROC_FUNC_TABLE_H_

/**
 * @file sdp_table.h
 */

#include "ska-sdp-func/utility/sdp_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup Table_struct
 * @{
 */

/**
 * @struct sdp_Table
 *
 * @brief Simple wrapper for a collection of named sdp_Mem arrays.
 *
 * The sdp_Table structure is a wrapper for a set of named sdp_Mem arrays,
 * much like an xarray Dataset is a collection of named numpy arrays.
 */
struct sdp_Table;

/** @} */ /* End group Table_struct. */

/* Typedefs. */
typedef struct sdp_Table sdp_Table;

/**
 * @defgroup Table_func
 * @{
 */

/**
 * @brief Create a new, empty table.
 *
 * @return ::sdp_Table* Handle to allocated table.
 */
sdp_Table* sdp_table_create();

/**
 * @brief Decrement the reference counter.
 *
 * If the reference counter reaches zero, any memory owned by the handle
 * will be released, and the handle destroyed.
 *
 * This is an alias for ::sdp_table_ref_dec().
 *
 * @param table Handle to table.
 */
void sdp_table_free(sdp_Table* table);

/**
 * @brief Return a handle to the array in a specific column.
 *
 * @param table Handle to table.
 * @param column_name Name of column.
 * @return sdp_Mem* Handle to array in column.
 */
sdp_Mem* sdp_table_get_column(sdp_Table* table, const char* column_name);

/**
 * @brief Returns the number of columns in the table.
 *
 * @param table Handle to table.
 * @return int64_t The number of columns in the table.
 */
int64_t sdp_table_num_columns(const sdp_Table* table);

/**
 * @brief Decrement the reference counter.
 *
 * If the reference counter reaches zero, any memory owned by the handle
 * will be released, and the handle destroyed.
 *
 * This is an alias for ::sdp_table_free().
 *
 * @param table Handle to table.
 */
void sdp_table_ref_dec(sdp_Table* table);

/**
 * @brief Increment the reference counter.
 *
 * Call if ownership of the handle needs to be transferred without incurring
 * the cost of an actual copy.
 *
 * @param table Handle to table.
 * @return ::sdp_Table* Handle to table (same as @p table).
 */
sdp_Table* sdp_table_ref_inc(sdp_Table* table);

/**
 * @brief Set the array used for the specified column.
 *
 * The reference count of the column will be increased, so this call will
 * effectively take ownership of the passed array.
 *
 * @param table Handle to table.
 * @param column_name Name of column.
 * @param column Handle to array to use for the column.
 */
void sdp_table_set_column(
        sdp_Table* table,
        const char* column_name,
        sdp_Mem* column
);

/** @} */ /* End group Table_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
