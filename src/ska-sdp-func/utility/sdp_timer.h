/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_TIMER_H_
#define SDP_TIMER_H_

/**
 * @file sdp_timer.h
 */

#include "ska-sdp-func/sdp_func_global.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sdp_Timer;
typedef struct sdp_Timer sdp_Timer;

enum sdp_TimerType
{
    SDP_TIMER_NATIVE,
    SDP_TIMER_CUDA
};
typedef enum sdp_TimerType sdp_TimerType;

/**
 * @brief Creates a timer.
 *
 * @details
 * Creates a timer. The timer is created in a paused state.
 *
 * The \p type parameter may take the values:
 *
 * - SDP_TIMER_NATIVE
 * - SDP_TIMER_CUDA
 *
 * These timers are used to measure performance of, respectively,
 * native code, or CUDA kernels.
 *
 * @param[in,out] timer Pointer to timer.
 * @param[in] type Type of timer to create.
 */
SDP_EXPORT
sdp_Timer* sdp_timer_create(sdp_TimerType type);

/**
 * @brief Destroys the timer.
 *
 * @details
 * Destroys the timer.
 *
 * @param[in,out] timer Pointer to timer.
 */
SDP_EXPORT
void sdp_timer_free(sdp_Timer* timer);

/**
 * @brief Returns the total elapsed time, in seconds.
 *
 * @details
 * Returns the number of seconds since the timer was started.
 *
 * @param[in,out] timer Pointer to timer.
 *
 * @return The number of seconds since the timer was started.
 */
SDP_EXPORT
double sdp_timer_elapsed(sdp_Timer* timer);

/**
 * @brief Pauses the timer.
 *
 * @details
 * Pauses the timer.
 *
 * @param[in,out] timer Pointer to timer.
 */
SDP_EXPORT
void sdp_timer_pause(sdp_Timer* timer);

/**
 * @brief Resets a timer.
 *
 * @details
 * Resets a timer back to its initial state.
 * The timer is zeroed and paused.
 *
 * @param[in,out] timer Pointer to timer.
 */
SDP_EXPORT
void sdp_timer_reset(sdp_Timer* timer);

/**
 * @brief Resumes the timer.
 *
 * @details
 * Resumes the timer from a paused state.
 *
 * @param[in,out] timer Pointer to timer.
 */
SDP_EXPORT
void sdp_timer_resume(sdp_Timer* timer);

/**
 * @brief Restarts the timer.
 *
 * @details
 * Restarts the timer.
 *
 * @param[in,out] timer Pointer to timer.
 */
SDP_EXPORT
void sdp_timer_restart(sdp_Timer* timer);

/**
 * @brief Starts and resets the timer.
 *
 * @details
 * Starts and resets the timer, clearing the current elapsed time.
 *
 * @param[in,out] timer Pointer to timer.
 */
SDP_EXPORT
void sdp_timer_start(sdp_Timer* timer);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
