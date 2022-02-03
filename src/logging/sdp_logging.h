/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_LOGGING_H_
#define SKA_SDP_PROC_FUNC_LOGGING_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Log severity levels are defined at
 * https://confluence.skatelescope.org/display/SWSI/SKA+Log+Message+Format
 *
 * As required, macros are used to insert the file and line number
 * from where the log was called.
 */
#define SDP_LOG_CRITICAL(...) \
        sdp_log_message(\
                "CRITICAL", stderr, __func__, FILENAME, __LINE__, __VA_ARGS__)

#define SDP_LOG_ERROR(...) \
        sdp_log_message(\
                "ERROR", stderr, __func__, FILENAME, __LINE__, __VA_ARGS__)

#define SDP_LOG_WARNING(...) \
        sdp_log_message(\
                "WARNING", stderr, __func__, FILENAME, __LINE__, __VA_ARGS__)

#define SDP_LOG_INFO(...) \
        sdp_log_message(\
                "INFO", stdout, __func__, FILENAME, __LINE__, __VA_ARGS__)

#define SDP_LOG_DEBUG(...) \
        sdp_log_message(\
                "DEBUG", stdout, __func__, FILENAME, __LINE__, __VA_ARGS__)

#ifndef SOURCE_PATH_SIZE
#define SOURCE_PATH_SIZE 0
#endif

#define FILENAME ((__FILE__) + SOURCE_PATH_SIZE)

void sdp_log_message(
        const char* level,
        FILE* stream,
        const char* func,
        const char* file,
        int line,
        const char* message, ...);

#ifdef __cplusplus
}
#endif

#endif /* include guard */