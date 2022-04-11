/* See the LICENSE file at the top-level directory of this distribution. */

#include <stdarg.h>
#include <time.h>
#include <sys/time.h>

#include "ska-sdp-func/utility/sdp_logging.h"

// Minimal SKA-compatible logging function.
void sdp_log_message(
        const char* level,
        FILE* stream,
        const char* func,
        const char* file,
        int line,
        const char* message, ...)
{
    // Get timestamp, as a string.
    char time_str[48];
    struct timeval tv;
    const time_t unix_time = time(0);
    struct tm* timeinfo = gmtime(&unix_time);
    gettimeofday(&tv, 0);
    snprintf(time_str, sizeof(time_str),
        "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
        timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
        timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec,
        (int)(tv.tv_usec) / 1000);

    // Print message to stream using SKA log formatting.
    va_list args;
    va_start(args, message);
    fprintf(stream, "1|%s|%s||%s|%s#%i|| ", time_str, level, func, file, line);
    vfprintf(stream, message, args);
    fprintf(stream, "\n");
    va_end(args);
}
