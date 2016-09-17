//#####################################################################
// Module utility
//#####################################################################
#include <geode/utility/Log.h>
#include <geode/utility/openmp.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <vector>
namespace geode {

static_assert(GEODE_SIZEOF_SIZE_T == sizeof(size_t), "GEODE_SIZEOF_SIZE_T incorrect on this platform");

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
    static_assert(is_same<decltype(t),const int>::value,"");
    const auto box = partition_loop(loop_steps,threads,t);
    GEODE_ASSERT(box.contains(i));
  }
}

static bool geode_endian_matches_native() {
  uint8_t a0 = 0xa0, a1 = 0xa1, a2 = 0xa2, a3 = 0xa3;
  uint32_t test_int = 0;
  char* dst = static_cast<char*>(static_cast<void*>(&test_int));
  memcpy(dst+0, &a0, 1);
  memcpy(dst+1, &a1, 1);
  memcpy(dst+2, &a2, 1);
  memcpy(dst+3, &a3, 1);
  static constexpr uint32_t little_endian_order = 0xa3a2a1a0;
  static constexpr uint32_t big_endian_order = 0xa0a1a2a3;
  GEODE_ASSERT(test_int == little_endian_order || test_int == big_endian_order);
#if GEODE_ENDIAN == GEODE_LITTLE_ENDIAN
  return test_int == little_endian_order;
#elif GEODE_ENDIAN == GEODE_BIG_ENDIAN
  return test_int == big_endian_order;
#else
  #error Unknown machine endianness
#endif
}

} //namespace geode
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

  GEODE_FUNCTION(geode_endian_matches_native)

  GEODE_WRAP(base64)
  GEODE_WRAP(resource)
  GEODE_WRAP(format)
  GEODE_WRAP(process)
}
