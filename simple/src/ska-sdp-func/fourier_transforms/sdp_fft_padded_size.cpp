/* See the LICENSE file at the top-level directory of this distribution. */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "ska-sdp-func/fourier_transforms/sdp_fft_padded_size.h"
#include "ska-sdp-func/utility/sdp_logging.h"

using namespace std;


// Helper functions for the min-heap.
static void heapify_up(int64_t heap[], int index)
{
    int parent = (index - 1) / 2;
    while (index > 0 && heap[parent] > heap[index])
    {
        const int64_t temp = heap[parent];
        heap[parent] = heap[index];
        heap[index] = temp;
        index = parent;
        parent = (index - 1) / 2;
    }
}


static void heapify_down(int64_t heap[], int heap_size, int index)
{
    int smallest = index;
    const int left = 2 * index + 1;
    const int right = 2 * index + 2;

    if (left < heap_size && heap[left] < heap[smallest])
    {
        smallest = left;
    }
    if (right < heap_size && heap[right] < heap[smallest])
    {
        smallest = right;
    }
    if (smallest != index)
    {
        const int64_t temp = heap[index];
        heap[index] = heap[smallest];
        heap[smallest] = temp;
        heapify_down(heap, heap_size, smallest);
    }
}


static int64_t heap_extract_min(int64_t heap[], int* heap_size)
{
    const int64_t min = heap[0];
    heap[0] = heap[--(*heap_size)];
    heapify_down(heap, *heap_size, 0);
    return min;
}


static void heap_insert(
        int64_t** heap,
        int* heap_size,
        int* heap_capacity,
        int64_t value
)
{
    // Check if we need to resize the heap.
    if (*heap_size >= *heap_capacity)
    {
        *heap_capacity *= 2;
        *heap = (int64_t*) realloc(*heap, (*heap_capacity) * sizeof(int64_t));
        if (!*heap)
        {
            SDP_LOG_CRITICAL("Memory reallocation failed");
            return;
        }
    }

    // Insert the new value at the end of the heap.
    (*heap)[(*heap_size)++] = value;
    heapify_up(*heap, *heap_size - 1);
}


int sdp_fft_padded_size(int n, double padding_factor)
{
    const int primes[] = {2, 3, 5, 7, 11};
    const int num_primes = sizeof(primes) / sizeof(primes[0]);

    // Allocate a heap with an initial capacity.
    int heap_size = 0;
    int heap_capacity = 100;
    int64_t* heap = (int64_t*) malloc(heap_capacity * sizeof(int64_t));
    if (!heap)
    {
        SDP_LOG_CRITICAL("Memory allocation failed");
        return 0;
    }

    // Insert the initial number 2, to ensure the result is divisible by 2.
    heap_insert(&heap, &heap_size, &heap_capacity, 2);

    int64_t prev = 0, next = 0;
    n = (int) ceil(n * padding_factor);
    const int64_t limit = (int64_t) n * 2;
    while (heap_size > 0)
    {
        next = heap_extract_min(heap, &heap_size);
        if (next >= n) break;
        if (next == prev) continue;
        prev = next;
        for (int j = 0; j < num_primes; j++)
        {
            const int64_t trial = next * primes[j];
            if (trial <= limit)
            {
                heap_insert(&heap, &heap_size, &heap_capacity, trial);
            }
        }
    }
    free(heap);
    return (int) next;
}
