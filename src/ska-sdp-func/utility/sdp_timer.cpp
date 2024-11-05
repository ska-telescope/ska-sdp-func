/* See the LICENSE file at the top-level directory of this distribution. */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ska-sdp-func/utility/sdp_logging.h"
#include "ska-sdp-func/utility/sdp_timer.h"

#ifndef SDP_OS_WIN
#include <sys/time.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

#ifdef SDP_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

using std::map;
using std::string;
using std::vector;

struct sdp_Timer
{
#ifdef SDP_HAVE_CUDA
    cudaEvent_t start_cuda, end_cuda;
#endif
    double start, elapsed;
#ifdef SDP_OS_WIN
    double freq;
#endif
    sdp_TimerType type;
    int paused;
};


static double sdp_get_wtime(sdp_Timer* timer)
{
#if defined(SDP_OS_WIN)
    /* Windows-specific version. */
    LARGE_INTEGER cntr;
    QueryPerformanceCounter(&cntr);
    return (double)(cntr.QuadPart) / timer->freq;
#elif _POSIX_MONOTONIC_CLOCK > 0
    /* Use monotonic clock if available. */
    struct timespec ts;
    (void)timer;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#else
    /* Use gettimeofday() as fallback. */
    struct timeval tv;
    (void)timer;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec / 1e6;
#endif
}


sdp_Timer* sdp_timer_create(sdp_TimerType type)
{
    sdp_Timer* timer = 0;
#ifdef SDP_OS_WIN
    LARGE_INTEGER freq;
#endif
    timer = (sdp_Timer*) calloc(1, sizeof(sdp_Timer));
#ifdef SDP_OS_WIN
    QueryPerformanceFrequency(&freq);
    timer->freq = (double)(freq.QuadPart);
#endif
    timer->type = type;
    timer->paused = 1;
#ifdef SDP_HAVE_CUDA
    if (timer->type == SDP_TIMER_CUDA)
    {
        cudaEventCreate(&timer->start_cuda);
        cudaEventCreate(&timer->end_cuda);
    }
#endif
    return timer;
}


void sdp_timer_free(sdp_Timer* timer)
{
    if (!timer) return;
#ifdef SDP_HAVE_CUDA
    if (timer->type == SDP_TIMER_CUDA)
    {
        cudaEventDestroy(timer->start_cuda);
        cudaEventDestroy(timer->end_cuda);
    }
#endif
    free(timer);
}


double sdp_timer_elapsed(sdp_Timer* timer)
{
    /* If timer is paused, return immediately with current elapsed time. */
    if (timer->paused) return timer->elapsed;

#ifdef SDP_HAVE_CUDA
    if (timer->type == SDP_TIMER_CUDA)
    {
        float millisec = 0.0f;

        /* Get elapsed time since start. */
        cudaEventRecord(timer->end_cuda, 0);
        cudaEventSynchronize(timer->end_cuda);
        cudaEventElapsedTime(&millisec, timer->start_cuda, timer->end_cuda);

        /* Increment elapsed time and restart. */
        timer->elapsed += millisec / 1000.0;
        cudaEventRecord(timer->start_cuda, 0);
        return timer->elapsed;
    }
#endif
    const double now = sdp_get_wtime(timer);

    /* Increment elapsed time and restart. */
    timer->elapsed += (now - timer->start);
    timer->start = now;
    return timer->elapsed;
}


void sdp_timer_pause(sdp_Timer* timer)
{
    if (timer->paused) return;
    (void)sdp_timer_elapsed(timer);
    timer->paused = 1;
}


void sdp_timer_reset(sdp_Timer* timer)
{
    timer->paused = 1;
    timer->start = 0.0;
    timer->elapsed = 0.0;
}


void sdp_timer_resume(sdp_Timer* timer)
{
    if (!timer->paused) return;
    sdp_timer_restart(timer);
}


void sdp_timer_restart(sdp_Timer* timer)
{
    timer->paused = 0;
#ifdef SDP_HAVE_CUDA
    if (timer->type == SDP_TIMER_CUDA)
    {
        cudaEventRecord(timer->start_cuda, 0);
        return;
    }
#endif
    timer->start = sdp_get_wtime(timer);
}


void sdp_timer_start(sdp_Timer* timer)
{
    timer->elapsed = 0.0;
    sdp_timer_restart(timer);
}


// sdp_TimerNode implementation.


#define WRITE_LINE(TXT, TIME) \
    sdp_log_message(SDP_LOG_LEVEL_INFO, stdout, func, file, line, \
        "| %s%c %-22s: %.3f sec (%.1f%%)", \
        space.c_str(), symbol, TXT, TIME, 100 * (TIME) / total \
    )

#define WRITE_TIME(TXT, TIME) \
    sdp_log_message(SDP_LOG_LEVEL_INFO, stdout, func, file, line, \
        "| %s%c %-22s: %.3f sec", space.c_str(), symbol, TXT, TIME \
    )


struct sdp_TimerNode
{
    typedef std::map<string, sdp_TimerNode> MapType;
    typedef std::pair<MapType::const_iterator, double> PairType;
    sdp_TimerNode* parent;
    vector<sdp_Timer*> tmr;
    MapType child;

    // Creates a timer node.
    sdp_TimerNode(sdp_TimerNode* parent = 0) : parent(parent)
    {
    }

    // Destroys a timer node.
    ~sdp_TimerNode()
    {
        for (size_t i = 0; i < tmr.size(); ++i)
        {
            sdp_timer_free(tmr[i]);
        }
    }

    // Internal function used to sort elapsed times for the report.
    static int cmp_fn(const PairType& a, const PairType& b)
    {
        return a.second > b.second;
    }

    // Internal function to return a timer node pointer for the named timer.
    static sdp_TimerNode* get(const string& name, sdp_TimerNode* parent)
    {
        sdp_TimerNode::MapType::iterator it = parent->child.find(name);
        if (it == parent->child.end())
        {
            it = parent->child.insert(
                    std::make_pair(name, sdp_TimerNode(parent))
            ).first;
        }
        return &(it->second);
    }

    // Get the maximum elapsed time from the set of timers in the node.
    static double get_max_time(const sdp_TimerNode* node)
    {
        vector<double> time(node->tmr.size());
        for (size_t i = 0; i < time.size(); ++i)
        {
            time[i] = node->tmr[i] ? sdp_timer_elapsed(node->tmr[i]) : 0;
        }
        return *(std::max_element(time.begin(), time.end()));
    }

    // Print a timing report for this node and its child nodes.
    void report(
            const string& root_name,
            int depth,
            const char* func,
            const char* file,
            int line
    ) const
    {
        const double total = get_max_time(this);
        const char symbols[] = {'+', '-', '*'};
        const char symbol = symbols[depth % 3];
        string space;
        if (depth == 0) WRITE_TIME(root_name.c_str(), total);
        vector<PairType> child_data;
        for (MapType::const_iterator it = child.cbegin();
                it != child.cend(); ++it)
        {
            child_data.push_back(
                    std::make_pair(it, get_max_time(&(it->second)))
            );
        }
        if (child_data.size() > 0)
        {
            std::sort(child_data.begin(), child_data.end(), cmp_fn);
        }
        double sum = 0;
        space = string((depth + 1) * 2, ' ');
        for (size_t i = 0; i < child_data.size(); ++i)
        {
            const string& tm_name = child_data[i].first->first;
            const sdp_TimerNode& node = child_data[i].first->second;
            const double elapsed  = child_data[i].second;
            WRITE_LINE(tm_name.c_str(), elapsed);
            sum += elapsed;
            if (node.tmr.size() > 1)
            {
                const int num = (int) node.tmr.size(), quart = num / 4;
                vector<double> time(num);
                for (int i = 0; i < num; ++i)
                {
                    time[i] = node.tmr[i] ? sdp_timer_elapsed(node.tmr[i]) : 0;
                }
                std::sort(time.begin(), time.end());
                const string space = string((depth + 2) * 2, ' ');
                const char symbol = '>';
                WRITE_LINE("Minimum", time[0]);
                WRITE_LINE("Maximum", time[num - 1]);
                WRITE_LINE("Median",  time[num / 2]);
                WRITE_TIME("Interquartile range",
                        (time[3 * quart] - time[quart])
                );
            }
            if (node.child.size() > 0)
            {
                node.report(root_name, depth + 1, func, file, line);
            }
        }
        if (sum < 0.99 * total) WRITE_LINE("(unaccounted time)", total - sum);
    }
};


// sdp_Timers implementation.


struct sdp_Timers
{
    string name;
    sdp_TimerType type;
    sdp_TimerNode* root_node;
    sdp_TimerNode* current_node;

    sdp_Timers(const string& name, sdp_TimerType type) :
        name(name), type(type), root_node(0), current_node(0)
    {
        current_node = root_node = new sdp_TimerNode;
        root_node->tmr.resize(1);
        root_node->tmr[0] = sdp_timer_create(type);
        sdp_timer_resume(root_node->tmr[0]);
    }

    ~sdp_Timers()
    {
        delete root_node;
    }
};


sdp_Timers* sdp_timers_create(
        const char* name,
        sdp_TimerType type,
        int num_threads
)
{
    char buf[20];
    std::snprintf(buf, 20, "%d", num_threads);
    string str = string(name) + string(", ") + string(buf) + string(" thread");
    if (num_threads > 1) str += string("s");
    return new sdp_Timers(str, type);
}


void sdp_timers_free(sdp_Timers* timers)
{
    delete timers;
}


sdp_TimerNode* sdp_timers_create_set(
        sdp_Timers* timers,
        const char* name,
        int num,
        sdp_TimerNode* parent
)
{
    sdp_TimerNode* node = sdp_TimerNode::get(
            name, parent ? parent : timers->current_node
    );
    if (node->tmr.size() == 0) node->tmr.resize(num, NULL);
    return node;
}


sdp_TimerNode* sdp_timers_pop(
        sdp_Timers* timers,
        int idx,
        sdp_TimerNode* timer
)
{
    sdp_TimerNode* node = timer ? timer : timers->current_node;
    if (node == timers->root_node) return timer ? node : NULL;
    sdp_timer_pause(node->tmr[idx]);
    if (!timer) timers->current_node = node->parent;
    return timer ? node->parent : NULL;
}


sdp_TimerNode* sdp_timers_pop_push(
        sdp_Timers* timers,
        const char* name,
        int idx,
        sdp_TimerNode* timer
)
{
    sdp_TimerNode* node = sdp_timers_pop(timers, idx, timer);
    return sdp_timers_push(timers, name, idx, node);
}


sdp_TimerNode* sdp_timers_push(
        sdp_Timers* timers,
        const char* name,
        int idx,
        sdp_TimerNode* parent
)
{
    sdp_TimerNode* node = sdp_TimerNode::get(
            name, parent ? parent : timers->current_node
    );
    if (node->tmr.size() == 0) node->tmr.resize(1, NULL);
    if (!node->tmr[idx]) node->tmr[idx] = sdp_timer_create(timers->type);
    sdp_timer_resume(node->tmr[idx]);
    if (!parent) timers->current_node = node;
    return parent ? node : NULL;
}


void sdp_timers_report(
        const sdp_Timers* timers,
        const char* func,
        const char* file,
        int line
)
{
    timers->root_node->report(timers->name, 0, func, file, line);
}


sdp_TimerNode* sdp_timers_root(sdp_Timers* timers)
{
    return timers->root_node;
}
