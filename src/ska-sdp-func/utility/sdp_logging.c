/* See the LICENSE file at the top-level directory of this distribution. */

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "ska-sdp-func/utility/sdp_logging.h"

static sdp_LogLevel log_filter = SDP_LOG_LEVEL_UNDEF;


// Minimal SKA-compatible logging function.
void sdp_log_message(
        sdp_LogLevel level,
        FILE* stream,
        const char* func,
        const char* file,
        int line,
        const char* message,
        ...
)
{
    // Check environment variable for log filter, if not defined.
    if (log_filter == SDP_LOG_LEVEL_UNDEF)
    {
        const char* env = getenv("SKA_SDP_FUNC_LOG_LEVEL");
        if (env)
        {
            if (!strncmp(env, "debug", 5) || !strncmp(env, "DEBUG", 5))
            {
                log_filter = SDP_LOG_LEVEL_DEBUG;
            }
            else if (!strncmp(env, "info", 4) || !strncmp(env, "INFO", 4))
            {
                log_filter = SDP_LOG_LEVEL_INFO;
            }
            else if (!strncmp(env, "warn", 4) || !strncmp(env, "WARN", 4))
            {
                log_filter = SDP_LOG_LEVEL_WARNING;
            }
            else if (!strncmp(env, "err", 3) || !strncmp(env, "ERR", 3))
            {
                log_filter = SDP_LOG_LEVEL_ERROR;
            }
            else if (!strncmp(env, "crit", 4) || !strncmp(env, "CRIT", 4))
            {
                log_filter = SDP_LOG_LEVEL_CRITICAL;
            }
        }
        else
        {
            log_filter = SDP_LOG_LEVEL_DEBUG;
        }
    }

    // Check filter.
    if (level < log_filter) return;

    // Get timestamp, as a string.
    char time_str[48];
    struct timeval tv;
    const time_t unix_time = time(0);
    struct tm* timeinfo = gmtime(&unix_time);
    gettimeofday(&tv, 0);
    if (snprintf(time_str, sizeof(time_str),
            "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
            timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
            timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec,
            (int)(tv.tv_usec) / 1000
            ) >= (int) sizeof(time_str))
    {
        (void) fprintf(stream, "Failed to generate time stamp!");
        return;
    }

    // Convert level to string.
    const char* level_str = 0;
    switch (level)
    {
    case SDP_LOG_LEVEL_UNDEF:
        level_str = "UNDEF";
        break;
    case SDP_LOG_LEVEL_DEBUG:
        level_str = "DEBUG";
        break;
    case SDP_LOG_LEVEL_INFO:
        level_str = "INFO";
        break;
    case SDP_LOG_LEVEL_WARNING:
        level_str = "WARNING";
        break;
    case SDP_LOG_LEVEL_ERROR:
        level_str = "ERROR";
        break;
    case SDP_LOG_LEVEL_CRITICAL:
        level_str = "CRITICAL";
        break;
    default:
        level_str = "UNKNOWN";
        break;
    }

    // Print message to stream using SKA log formatting.
    va_list args;
    va_start(args, message);
    (void) fprintf(stream, "1|%s|%s||%s|%s#%i|| ",
            time_str, level_str, func, file, line
    );
    (void) vfprintf(stream, message, args);
    (void) fprintf(stream, "\n");
    va_end(args);
}
