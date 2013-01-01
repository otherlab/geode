//#####################################################################
// Module utility
//#####################################################################
#include <other/core/utility/Log.h>
#include <other/core/utility/openmp.h>
#include <other/core/python/module.h>
#include <vector>
namespace other {

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

static void partition_loop_test(int loop_steps, int threads) {
  for (int i : range(loop_steps))
    OTHER_ASSERT(partition_loop(loop_steps,threads,partition_loop_inverse(loop_steps,threads,i)).contains(i));
  for (int thread : range(threads))
    for (int i : partition_loop(loop_steps,threads,thread))
      OTHER_ASSERT(partition_loop_inverse(loop_steps,threads,i)==thread);
}

}
using namespace other;

void wrap_utility() {
  using namespace python;

  function("log_configure",Log::configure);
  function("log_cache_initial_output",Log::cache_initial_output);
  function("log_copy_to_file",Log::copy_to_file);
  function("log_finish",Log::finish);

  function("log_push_scope",Log::push_scope);
  function("log_pop_scope",Log::pop_scope);
  OTHER_FUNCTION(log_print)
  OTHER_FUNCTION(log_flush)
  OTHER_FUNCTION(log_error)

  OTHER_FUNCTION(partition_loop_test)

  OTHER_WRAP(base64)
  OTHER_WRAP(resource)
}
