// OpenMP helper routines
#pragma once

#include <other/core/math/min.h>
#include <other/core/utility/debug.h>
#include <other/core/utility/range.h>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace other {

#ifndef _OPENMP
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
#endif

// Partition a loop into chunks based on the total number of threads.  Returns a half open interval.
inline Range<int> partition_loop(const int loop_steps, const int threads, const int thread) {
  OTHER_ASSERT(threads>0 && (unsigned)thread<(unsigned)threads);
  const int steps_per_thread = loop_steps/threads, // Round down, so some threads will get one more step
            extra_steps = loop_steps%threads, // The first extra_steps threads will get one extra step
            start = steps_per_thread*thread+min(extra_steps,thread),
            end = start+steps_per_thread+(thread<extra_steps);
  return Range<int>(start,end);
}

// Same as above, but grab the thread counts from OpenMP
inline Range<int> partition_loop(const int loop_steps) {
  return partition_loop(loop_steps,omp_get_num_threads(),omp_get_thread_num());
}

// Inverse of partition_loop: map an index to the thread that owns it
inline int partition_loop_inverse(const int loop_steps, const int threads, const int index) {
  OTHER_ASSERT(threads>0 && (unsigned)index<(unsigned)loop_steps);
  const int steps_per_thread = loop_steps/threads, // Round down, so some threads will get one more step
            extra_steps = loop_steps%threads, // The first extra_steps threads will get one extra step
            threshold = (steps_per_thread+1)*extra_steps; // Before this point, all threads have an extra step
  return index<threshold  ? index/(steps_per_thread+1)
       : steps_per_thread ? extra_steps+(index-threshold)/steps_per_thread
                          : 0; // Only occurs if loop_steps==0
}

}
