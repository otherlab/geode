//#####################################################################
// Module utility
//#####################################################################
#include <other/core/utility/Log.h>
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
  function("log_print",log_print);
  function("log_flush",log_flush);
  function("log_error",log_error);
}
