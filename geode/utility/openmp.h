// OpenMP helper routines
#pragma once

#include <geode/math/min.h>
#include <geode/utility/debug.h>
#include <geode/utility/range.h>
#include <geode/utility/type_traits.h>
#include <boost/integer.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace geode {

#ifndef _OPENMP
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
#endif

// Partition a loop into chunks based on the total number of threads.  Returns a half open interval.
template<class I,class TI> inline Range<I> partition_loop(const I loop_steps, const TI threads, const TI thread) {
  static_assert(is_integral<I>::value && is_integral<TI>::value && sizeof(TI)<=sizeof(I),"");
  typedef typename boost::uint_t<8*sizeof(TI)>::exact TUI;
  GEODE_ASSERT(threads>0 && TUI(thread)<TUI(threads));
  const I steps_per_thread = loop_steps/threads, // Round down, so some threads will get one more step
          extra_steps = loop_steps%threads, // The first extra_steps threads will get one extra step
          start = steps_per_thread*thread+min(extra_steps,thread),
          end = start+steps_per_thread+(I(thread)<extra_steps);
  return Range<I>(start,end);
}

// Same as above, but grab the thread counts from OpenMP
template<class I> inline Range<I> partition_loop(const I loop_steps) {
  return partition_loop(loop_steps,omp_get_num_threads(),omp_get_thread_num());
}

// Inverse of partition_loop: map an index to the thread that owns it
template<class I,class TI> inline TI partition_loop_inverse(const I loop_steps, const TI threads, const I index) {
  static_assert(is_integral<I>::value && is_integral<TI>::value,"");
  typedef typename boost::uint_t<8*sizeof(I)>::exact UI;
  GEODE_ASSERT(threads>0 && UI(index)<UI(loop_steps));
  const I steps_per_thread = loop_steps/threads, // Round down, so some threads will get one more step
          extra_steps = loop_steps%threads, // The first extra_steps threads will get one extra step
          threshold = (steps_per_thread+1)*extra_steps; // Before this point, all threads have an extra step
  return TI(     index<threshold  ? index/(steps_per_thread+1)
            : steps_per_thread ? extra_steps+(index-threshold)/steps_per_thread
                               : 0); // Only occurs if loop_steps==0
}

}
