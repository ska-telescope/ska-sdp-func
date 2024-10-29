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

/** @} */ /* End group Timer_func. */

#ifdef __cplusplus
}

#include <map>
#include <string>
#include <vector>

/**
 * @brief Utility class for a timer hierarchy.
 *
 * Use @ref push("<timer name>") to start timing a section of code until the
 * next @ref pop(). The elapsed time will be reported using the given name.
 * If a timer already exists with that name, the elapsed time will be
 * added to that timer, otherwise a new timer will be created.
 *
 * Use @ref pop_push("<timer name>") to pause the current timer and start
 * another.
 *
 * Use @ref report() to print out a timing report for all the known timers.
 *
 * In a multi-threaded environment, each thread will need its own timer.
 * Before starting the threads, call
 * @ref create_timers("<timer name>", num_threads), specifying the name of the
 * timer and the number of threads it will be accessed from,
 * to allocate a slot for each.
 * Then supply the thread ID in relevant calls to @ref push() and @ref pop().
 * In a single-threaded environment, the state can be managed internally,
 * but when using multiple threads, the state is returned to the caller and
 * must be explicitly passed using the last argument to @ref push()
 * and @ref pop().
 *
 * For example:
 *
 * @code{.cpp}
 * sdp_Timers timers("A collection of timers", SDP_TIMER_NATIVE);
 * timers.create_timers("Processing something", num_threads);
 *
 * // In multi-threaded code:
 * sdp_Timers::TimerNode* node = timers.root();
 * node = timers.push("Processing something", thread_id, node);
 * process_something_that_takes_a_while();
 * node = timers.pop(thread_id, node);
 * @endcode
 *
 * The minimum, maximum, median and interquartile range is reported for
 * all timers which have multiple index values.
 */
class sdp_Timers
{
public:
    /**
     * @brief Internal class used to create the timer hierarchy.
     */
    class TimerNode
    {
        friend class sdp_Timers;
        typedef std::map<std::string, TimerNode> MapType;
        typedef std::pair<MapType::const_iterator, double> PairType;
        TimerNode* parent_;
        std::vector<sdp_Timer*> tmr_;
        MapType child_;

        // Internal function used to sort elapsed times for the report.
        static int cmp_fn(const PairType& a, const PairType& b);

        // Get the maximum elapsed time from the set of timers in the node.
        static double get_max_time(const TimerNode* node);

    public:
        // Creates a timer node.
        TimerNode(TimerNode* parent = 0) : parent_(parent)
        {
        }

        // Destroys a timer node.
        ~TimerNode();

        // Print a timing report.
        void report(
                const std::string& root_name,
                int depth,
                const char* func,
                const char* file,
                int line
        ) const;
    };

private:
    std::string name_;
    sdp_TimerType type_;
    int num_threads_;
    TimerNode root_;
    TimerNode* current_;

    // Internal function to return a timer node pointer for the named timer.
    static TimerNode* timer_node(const std::string& name, TimerNode* parent);

public:
    /**
     * @brief Creates a new timer hierarchy with the given name.
     *
     * @param name String to display at the root of the timer hierarchy.
     * @param type Enumerated type of timers to create.
     * @param num_threads Number of threads to display next to name string.
     */
    sdp_Timers(const std::string& name, sdp_TimerType type, int num_threads);

    /**
     * @brief Allocate slots for multiple timers of the given name.
     *
     * This is only required for timers in regions with multiple threads.
     * Call this before entering the parallel region.
     *
     * @param name Name of timer node to create.
     * @param num Number of timer slots to create on the node (one per thread).
     * @param parent Optional pointer to timer's parent node.
     * If NULL, this is taken from the current internal state.
     * @return A pointer to the (new) timer node.
     */
    TimerNode* create_timers(
            const std::string& name,
            int num,
            TimerNode* parent = NULL
    );

    /**
     * @brief Pause the current timer.
     *
     * This should be called after a call to @ref push().
     *
     * The internal state will be updated unless @p timer is specified.
     * Note that this method is only thread safe when @p timer is non-NULL.
     *
     * @param idx Optional index of timer to pause.
     * @param timer Optional pointer to timer node.
     * If NULL, this is taken from the current internal state.
     * @return If @p timer was specified, a pointer to the timer's parent node.
     */
    TimerNode* pop(int idx = 0, TimerNode* timer = NULL);

    /**
     * @brief Resumes the named timer, creating it first if necessary.
     *
     * The internal state will be updated unless @p parent is specified.
     * Note that this method is only thread safe when @p parent is non-NULL.
     *
     * @param name Name of timer node.
     * @param idx Optional index of timer to resume.
     * @param parent Optional pointer to the timer's parent node.
     * If NULL, this is taken from the current internal state.
     * @return If @p parent was specified, a pointer to the timer node.
     */
    TimerNode* push(
            const std::string& name,
            int idx = 0,
            TimerNode* parent = NULL
    );

    /**
     * @brief Convenience method that calls @ref pop() followed by @ref push().
     *
     * The internal state will be updated unless @p timer is specified.
     * Note that this method is only thread safe when @p timer is non-NULL.
     *
     * @param name Name of timer node.
     * @param idx Optional index of timer to resume.
     * @param timer Optional pointer to current timer node.
     * If NULL, this is taken from the current internal state.
     * @return If @p timer was specified, a pointer to the timer node.
     */
    TimerNode* pop_push(
            const std::string& name,
            int idx = 0,
            TimerNode* timer = NULL
    );

    /**
     * @brief Print a timing report from all timers.
     *
     * @param func Function name to print in log messages.
     * @param file File name to print in log messages.
     * @param line Line number to print in log messages.
     */
    void report(const char* func, const char* file, int line) const;

    /**
     * @brief Return a pointer to the timer hierarchy's root node.
     *
     * @return A pointer to the root node.
     */
    TimerNode* root();
};

#endif

#endif /* include guard */
