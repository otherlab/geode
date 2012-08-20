// OpenMP helper routines
#pragma once

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
inline Range<int> partition_loop(int loop_steps, int threads, int thread) {
  OTHER_ASSERT(threads>0 && (unsigned)thread<(unsigned)threads);
  int steps_per_thread = loop_steps/threads, // rounds down, so some threads with get one more step
      extra_steps = loop_steps%threads, // the first extra_steps threads will get one extra step
      start = steps_per_thread*thread+min(extra_steps,thread),
      end = start+steps_per_thread+(thread<extra_steps);
  return Range<int>(start,end);
}

// Same as above, but grab the thread counts from OpenMP
inline Range<int> partition_loop(int loop_steps) {
  return partition_loop(loop_steps,omp_get_num_threads(),omp_get_thread_num());
}

}
