/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef SDP_TIMER_H_
#define SDP_TIMER_H_

/**
 * @file sdp_timer.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup Timer_struct
 * @{
 */

/**
 * @struct sdp_Timer
 *
 * @brief Provides a basic timer.
 *
 * Call ::sdp_timer_create() to create a timer.
 * The timer is created in a paused state.
 *
 * To resume the timer, pause the timer, or find the current elapsed time, call
 * ::sdp_timer_resume(), ::sdp_timer_pause() or ::sdp_timer_elapsed(),
 * respectively.
 *
 * Release memory used by the timer by calling ::sdp_timer_free().
 */
struct sdp_Timer;

/**
 * @struct sdp_TimerNode
 *
 * @brief Internal data used to create a timer hierarchy.
 */
struct sdp_TimerNode;

/**
 * @struct sdp_Timers
 *
 * @brief Provides a timer hierarchy.
 *
 * Use ::sdp_timers_push() to start timing a section of code until the
 * next ::sdp_timers_pop(). The elapsed time will be reported using the name
 * supplied to ::sdp_timers_push().
 * If a timer already exists with that name, the elapsed time will be
 * added to that timer, otherwise a new timer will be created.
 *
 * Use ::sdp_timers_pop_push() to pause the current timer and start another.
 *
 * Use ::sdp_timers_report() to print out a timing report for all the timers
 * in the hierarchy.
 *
 * Using the macros ::SDP_TMR_CREATE(), ::SDP_TMR_PUSH(), ::SDP_TMR_POP(),
 * ::SDP_TMR_POP_PUSH(), ::SDP_TMR_REPORT() may be clearer and more convenient
 * in many cases.
 * The macro ::SDP_TMR_HANDLE can be used to obtain the name of the handle
 * to the timer hierarchy returned in ::SDP_TMR_CREATE().
 *
 * Call ::sdp_timers_free() or ::SDP_TMR_FREE() to release memory held by
 * the timer hierarchy.
 *
 * For example:
 *
 * @code{.cpp}
 * sdp_Timers* timers = sdp_timers_create(
 *         "A collection of timers", SDP_TIMER_NATIVE, 1
 * );
 *
 * sdp_timers_push(timers, "Doing something", 0, 0);
 * do_something_that_takes_a_while();
 *
 * sdp_timers_pop_push(timers, "Doing something else", 0, 0);
 * do_something_else_that_takes_a_while();
 * sdp_timers_pop(timers, 0, 0);
 *
 * sdp_timers_report(timers, __func__, FILENAME, __LINE__);
 * sdp_timers_free(timers);
 * @endcode
 *
 * Or, equivalently:
 *
 * @code{.cpp}
 * SDP_TMR_CREATE("A collection of timers", SDP_TIMER_NATIVE, 1);
 *
 * SDP_TMR_PUSH("Doing something");
 * do_something_that_takes_a_while();
 *
 * SDP_TMR_POP_PUSH("Doing something else");
 * do_something_else_that_takes_a_while();
 * SDP_TMR_POP;
 *
 * SDP_TMR_REPORT;
 * SDP_TMR_FREE;
 * @endcode
 *
 * In a multi-threaded environment, each thread will need its own timer.
 * Before starting the threads, call ::sdp_timers_create_set(), specifying
 * the name of the timer and the number of threads it will be accessed from,
 * to allocate a slot for each.
 * Then supply the thread ID in relevant calls to ::sdp_timers_push()
 * and ::sdp_timers_pop().
 * In a single-threaded environment, the state can be managed internally,
 * but when using multiple threads, the state is returned to the caller and
 * must be explicitly passed.
 *
 * For example:
 *
 * @code{.cpp}
 * SDP_TMR_CREATE("A collection of timers", SDP_TIMER_NATIVE, num_threads);
 * SDP_TMR_CREATE_SET("Doing something", num_threads);
 *
 * // In multi-threaded code:
 * sdp_TimerNode* node = SDP_TMR_ROOT;
 * node = sdp_timers_push(SDP_TMR_HANDLE, "Doing something", thread_id, node);
 * do_something_that_takes_a_while();
 * node = sdp_timers_pop(SDP_TMR_HANDLE, thread_id, node);
 *
 * // In single-threaded code:
 * SDP_TMR_PUSH("Doing something else");
 * do_something_else_that_takes_a_while();
 * SDP_TMR_POP;
 *
 * SDP_TMR_REPORT;
 * SDP_TMR_FREE;
 * @endcode
 *
 * The minimum, maximum, median and interquartile range is reported for
 * all timer names which have multiple index values.
 */
struct sdp_Timers;

/** @} */ /* End group Timer_struct. */

/**
 * @defgroup Timer_enum
 * @{
 */

enum sdp_TimerType
{
    SDP_TIMER_NATIVE,
    SDP_TIMER_CUDA
};

/** @} */ /* End group Timer_enum. */

/* Typedefs. */
typedef struct sdp_Timer sdp_Timer;
typedef struct sdp_Timers sdp_Timers;
typedef struct sdp_TimerNode sdp_TimerNode;
typedef enum sdp_TimerType sdp_TimerType;

/**
 * @defgroup Timer_func
 * @{
 */

/**
 * @brief Creates a timer.
 *
 * The timer is created in a paused state.
 *
 * The \p type parameter may take the values:
 *
 * - SDP_TIMER_NATIVE
 * - SDP_TIMER_CUDA
 *
 * These timers are used to measure performance of, respectively,
 * native code, or CUDA kernels.
 *
 * @param type Type of timer to create.
 * @return Pointer to created timer.
 */
sdp_Timer* sdp_timer_create(sdp_TimerType type);

/**
 * @brief Destroys the timer.
 *
 * @param timer Pointer to timer.
 */
void sdp_timer_free(sdp_Timer* timer);

/**
 * @brief Returns the total elapsed time since starting the timer, in seconds.
 *
 * @param timer Pointer to timer.
 *
 * @return The number of seconds since the timer was started.
 */
double sdp_timer_elapsed(sdp_Timer* timer);

/**
 * @brief Pauses the timer.
 *
 * @param timer Pointer to timer.
 */
void sdp_timer_pause(sdp_Timer* timer);

/**
 * @brief Resets a timer back to its initial state.
 *
 * The timer is zeroed and paused.
 *
 * @param timer Pointer to timer.
 */
void sdp_timer_reset(sdp_Timer* timer);

/**
 * @brief Resumes the timer from a paused state.
 *
 * @param timer Pointer to timer.
 */
void sdp_timer_resume(sdp_Timer* timer);

/**
 * @brief Restarts the timer.
 *
 * @param timer Pointer to timer.
 */
void sdp_timer_restart(sdp_Timer* timer);

/**
 * @brief Starts and resets the timer, clearing the current elapsed time.
 *
 * @param timer Pointer to timer.
 */
void sdp_timer_start(sdp_Timer* timer);

/**
 * @brief Creates a new timer hierarchy with the given @p name.
 *
 * See also ::SDP_TMR_CREATE()
 *
 * @param name String to display at the root of the timer hierarchy.
 * @param type Enumerated type of timers to create.
 * @param num_threads Number of threads to display next to name string.
 * @return Pointer to timer hierarchy.
 */
sdp_Timers* sdp_timers_create(
        const char* name,
        sdp_TimerType type,
        int num_threads
);

/**
 * @brief Destroys the timer hierarchy.
 *
 * See also ::SDP_TMR_FREE
 *
 * @param timers Pointer to timer hierarchy.
 */
void sdp_timers_free(sdp_Timers* timers);

/**
 * @brief Allocate slots for multiple timers of the given @p name.
 *
 * This is only required for timers in regions with multiple threads.
 * Call this before entering the parallel region.
 *
 * See also ::SDP_TMR_CREATE_SET()
 *
 * @param timers Pointer to timer hierarchy.
 * @param name Name of timer node to create.
 * @param num Number of timer slots to create on the node (one per thread).
 * @param parent Optional pointer to timer's parent node.
 * If NULL, this is taken from the current internal state.
 * @return A pointer to the (new) timer node.
 */
sdp_TimerNode* sdp_timers_create_set(
        sdp_Timers* timers,
        const char* name,
        int num,
        sdp_TimerNode* parent
);

/**
 * @brief Pause the current timer.
 *
 * This should be called after a call to ::sdp_timers_push().
 *
 * The internal state will be updated unless @p timer is specified.
 * Note that this is only thread safe when @p timer is non-NULL.
 *
 * See also ::SDP_TMR_POP
 *
 * @param timers Pointer to timer hierarchy.
 * @param idx Index of timer to pause.
 * @param timer Optional pointer to timer node.
 * If NULL, this is taken from the current internal state.
 * @return If @p timer was specified, a pointer to the timer's parent node.
 */
sdp_TimerNode* sdp_timers_pop(
        sdp_Timers* timers,
        int idx,
        sdp_TimerNode* timer
);

/**
 * @brief Resumes the named timer, creating it first if necessary.
 *
 * The timer will be created as a child of the specified @p parent,
 * or as a child of the current timer if @p parent is not specified.
 * The internal state will be updated unless @p parent is specified.
 * Note that this is only thread safe when @p parent is non-NULL.
 *
 * See also ::SDP_TMR_PUSH()
 *
 * @param timers Pointer to timer hierarchy.
 * @param name Name of timer node.
 * @param idx Index of timer to resume.
 * @param parent Optional pointer to the timer's parent node.
 * If NULL, this is taken from the current internal state.
 * @return If @p parent was specified, a pointer to the timer node.
 */
sdp_TimerNode* sdp_timers_push(
        sdp_Timers* timers,
        const char* name,
        int idx,
        sdp_TimerNode* parent
);

/**
 * @brief Calls ::sdp_timers_pop() followed by ::sdp_timers_push().
 *
 * The internal state will be updated unless @p timer is specified.
 * Note that this is only thread safe when @p timer is non-NULL.
 *
 * See also ::SDP_TMR_POP_PUSH()
 *
 * @param timers Pointer to timer hierarchy.
 * @param name Name of timer node.
 * @param idx Index of timer to pause and resume.
 * @param timer Optional pointer to current timer node.
 * If NULL, this is taken from the current internal state.
 * @return If @p timer was specified, a pointer to the timer node.
 */
sdp_TimerNode* sdp_timers_pop_push(
        sdp_Timers* timers,
        const char* name,
        int idx,
        sdp_TimerNode* timer
);

/**
 * @brief Print a timing report from all timers.
 *
 * See also ::SDP_TMR_REPORT
 *
 * @param timers Pointer to timer hierarchy.
 * @param func Function name to print in log messages.
 * @param file File name to print in log messages.
 * @param line Line number to print in log messages.
 */
void sdp_timers_report(
        const sdp_Timers* timers,
        const char* func,
        const char* file,
        int line
);

/**
 * @brief Return a pointer to the timer hierarchy's root node.
 *
 * See also ::SDP_TMR_ROOT
 *
 * @param timers Pointer to timer hierarchy.
 * @return A pointer to the root node.
 */
sdp_TimerNode* sdp_timers_root(sdp_Timers* timers);

/**
 * @brief Convenience macro to return the handle to the timer hierarchy.
 */
#define SDP_TMR_HANDLE _sdp_timers_handle_p__

/**
 * @brief Convenience macro which calls ::sdp_timers_create().
 */
#define SDP_TMR_CREATE(NAME, TYPE, NUM_THREADS) \
    sdp_Timers* SDP_TMR_HANDLE = sdp_timers_create(NAME, TYPE, NUM_THREADS)

/**
 * @brief Convenience macro which calls ::sdp_timers_create_set().
 */
#define SDP_TMR_CREATE_SET(NAME, NUM_THREADS) \
    (void) sdp_timers_create_set(SDP_TMR_HANDLE, NAME, NUM_THREADS, 0)

/**
 * @brief Convenience macro which calls ::sdp_timers_free().
 */
#define SDP_TMR_FREE sdp_timers_free(SDP_TMR_HANDLE)

/**
 * @brief Convenience macro which calls ::sdp_timers_push().
 */
#define SDP_TMR_PUSH(NAME) (void) sdp_timers_push(SDP_TMR_HANDLE, NAME, 0, 0)

/**
 * @brief Convenience macro which calls ::sdp_timers_pop_push().
 */
#define SDP_TMR_POP_PUSH(NAME) \
    (void) sdp_timers_pop_push(SDP_TMR_HANDLE, NAME, 0, 0)

/**
 * @brief Convenience macro which calls ::sdp_timers_pop().
 */
#define SDP_TMR_POP (void) sdp_timers_pop(SDP_TMR_HANDLE, 0, 0)

/**
 * @brief Convenience macro which calls ::sdp_timers_report().
 */
#define SDP_TMR_REPORT \
    sdp_timers_report(SDP_TMR_HANDLE, __func__, FILENAME, __LINE__)

/**
 * @brief Convenience macro which calls ::sdp_timers_root().
 */
#define SDP_TMR_ROOT sdp_timers_root(SDP_TMR_HANDLE)

/** @} */ /* End group Timer_func. */

#ifdef __cplusplus
}
#endif

#endif /* include guard */
