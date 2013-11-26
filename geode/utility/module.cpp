//#####################################################################
// Module utility
//#####################################################################
#include <geode/utility/Log.h>
#include <geode/utility/openmp.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <vector>
namespace geode {

using std::pair;
using std::make_pair;
using std::vector;

static void log_print(const char* str) {
  Log::cout<<str<<std::endl;
}

static void log_error(const char* str) {
  Log::cerr<<str<<std::endl;
}

static void log_flush() {
  Log::cout<<std::flush;
}

static void partition_loop_test(const int loop_steps, const int threads) {
  for (int i : range(loop_steps))
    GEODE_ASSERT(partition_loop(loop_steps,threads,partition_loop_inverse(loop_steps,threads,i)).contains(i));
  for (int thread : range(threads))
    for (int i : partition_loop(loop_steps,threads,thread))
      GEODE_ASSERT(partition_loop_inverse(loop_steps,threads,i)==thread);
}

static void large_partition_loop_test(const uint64_t loop_steps, const int threads, const int samples) {
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
    static_assert(boost::is_same<decltype(t),const int>::value,"");
    const auto box = partition_loop(loop_steps,threads,t);
    GEODE_ASSERT(box.contains(i));
  }
}

}
using namespace geode;

void wrap_utility() {
  using namespace python;

  function("log_initialized",Log::initialized);
  function("log_configure",Log::configure);
  function("log_cache_initial_output",Log::cache_initial_output);
  function("log_copy_to_file",Log::copy_to_file);
  function("log_finish",Log::finish);

  function("log_push_scope",Log::push_scope);
  function("log_pop_scope",Log::pop_scope);
  GEODE_FUNCTION(log_print)
  GEODE_FUNCTION(log_flush)
  GEODE_FUNCTION(log_error)

  GEODE_FUNCTION(partition_loop_test)
  GEODE_FUNCTION(large_partition_loop_test)

  GEODE_WRAP(base64)
  GEODE_WRAP(resource)
  GEODE_WRAP(format)
  GEODE_WRAP(process)
}
