/* See the LICENSE file at the top-level directory of this distribution. */

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_table.h"

using std::map;
using std::string;
using std::vector;

// Private implementation.
struct sdp_Table
{
    int32_t ref_count; // Reference counter.
    map<string, sdp_Mem*> columns;
    vector<string> column_names;
};


sdp_Table* sdp_table_create()
{
    sdp_Table* table = new sdp_Table;
    table->ref_count = 1;
    return table;
}


void sdp_table_free(sdp_Table* table)
{
    if (!table) return;
    if (--table->ref_count > 0) return;
    for (map<string, sdp_Mem*>::iterator i = table->columns.begin();
            i != table->columns.end(); ++i)
    {
        sdp_mem_ref_dec(i->second);
    }
    delete table;
}


sdp_Mem* sdp_table_get_column(sdp_Table* table, const char* column_name)
{
    try
    {
        return (!table) ? 0 : table->columns.at(string(column_name));
    }
    catch (...)
    {
        SDP_LOG_ERROR("No column found with name '%s'", column_name);
        return 0;
    }
}


const sdp_Mem* sdp_table_get_column_const(
        const sdp_Table* table,
        const char* column_name
)
{
    try
    {
        return (!table) ? 0 : table->columns.at(string(column_name));
    }
    catch (...)
    {
        SDP_LOG_ERROR("No column found with name '%s'", column_name);
        return 0;
    }
}


int64_t sdp_table_num_columns(const sdp_Table* table)
{
    return (!table) ? 0 : table->columns.size();
}


void sdp_table_ref_dec(sdp_Table* table)
{
    sdp_table_free(table);
}


sdp_Table* sdp_table_ref_inc(sdp_Table* table)
{
    if (!table) return 0;
    table->ref_count++;
    return table;
}


void sdp_table_set_column(
        sdp_Table* table,
        const char* column_name,
        sdp_Mem* column
)
{
    if (!table || !column_name || !column) return;
    sdp_mem_ref_inc(column);
    table->columns[string(column_name)] = column;
}
