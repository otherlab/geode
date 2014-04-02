// OpenMP helper routines

#include <geode/utility/openmp.h>
#include <geode/random/Random.h>
namespace geode {

void partition_loop_test(const int loop_steps, const int threads) {
  for (int i : range(loop_steps))
    GEODE_ASSERT(partition_loop(loop_steps,threads,partition_loop_inverse(loop_steps,threads,i)).contains(i));
  for (int thread : range(threads))
    for (int i : partition_loop(loop_steps,threads,thread))
      GEODE_ASSERT(partition_loop_inverse(loop_steps,threads,i)==thread);
}

void large_partition_loop_test(const uint64_t loop_steps, const int threads, const int samples) {
  // Test partition loop
  const uint64_t chunk = loop_steps/threads;
  uint64_t offset = 0;
  for (const int thread : range(threads)) {
    const auto box = partition_loop(loop_steps,threads,thread);
    GEODE_ASSERT(offset==box.lo);
    GEODE_ASSERT(box.size()==chunk || box.size()==chunk+1);
    offset = box.hi;
  }
  GEODE_ASSERT(offset==loop_steps);

  // Test that partition_loop_inverse really is an inverse
  const auto random = new_<Random>(1381);
  for (int s=0;s<samples;s++) {
    const auto i = random->uniform<uint64_t>(0,loop_steps);
    const auto t = partition_loop_inverse(loop_steps,threads,i);
    static_assert(is_same<decltype(t),const int>::value,"");
    const auto box = partition_loop(loop_steps,threads,t);
    GEODE_ASSERT(box.contains(i));
  }
}

}
