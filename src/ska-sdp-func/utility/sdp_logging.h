/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SKA_SDP_PROC_FUNC_LOGGING_H_
#define SKA_SDP_PROC_FUNC_LOGGING_H_

/**
 * @file sdp_logging.h
 */

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

/**
 * @brief Writes a log message to stderr, with severity "CRITICAL".
 */
#define SDP_LOG_CRITICAL(...) \
        sdp_log_message( \
        SDP_LOG_LEVEL_CRITICAL, stderr, __func__, FILENAME, __LINE__, \
        __VA_ARGS__ \
        )

/**
 * @brief Writes a log message to stderr, with severity "ERROR".
 */
#define SDP_LOG_ERROR(...) \
        sdp_log_message( \
        SDP_LOG_LEVEL_ERROR, stderr, __func__, FILENAME, __LINE__, \
        __VA_ARGS__ \
        )

/**
 * @brief Writes a log message to stderr, with severity "WARNING".
 */
#define SDP_LOG_WARNING(...) \
        sdp_log_message( \
        SDP_LOG_LEVEL_WARNING, stderr, __func__, FILENAME, __LINE__, \
        __VA_ARGS__ \
        )

/**
 * @brief Writes a log message to stdout, with severity "INFO".
 */
#define SDP_LOG_INFO(...) \
        sdp_log_message( \
        SDP_LOG_LEVEL_INFO, stdout, __func__, FILENAME, __LINE__, \
        __VA_ARGS__ \
        )

/**
 * @brief Writes a log message to stdout, with severity "DEBUG".
 */
#define SDP_LOG_DEBUG(...) \
        sdp_log_message( \
        SDP_LOG_LEVEL_DEBUG, stdout, __func__, FILENAME, __LINE__, \
        __VA_ARGS__ \
        )

#ifndef SOURCE_PATH_SIZE
#define SOURCE_PATH_SIZE 0
#endif

#define FILENAME ((__FILE__) + SOURCE_PATH_SIZE)

enum sdp_LogLevel
{
    SDP_LOG_LEVEL_UNDEF,
    SDP_LOG_LEVEL_DEBUG,
    SDP_LOG_LEVEL_INFO,
    SDP_LOG_LEVEL_WARNING,
    SDP_LOG_LEVEL_ERROR,
    SDP_LOG_LEVEL_CRITICAL
};
typedef enum sdp_LogLevel sdp_LogLevel;

void sdp_log_message(
        sdp_LogLevel level,
        FILE* stream,
        const char* func,
        const char* file,
        int line,
        const char* message,
        ...
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
