#pragma once
#include <geode/utility/time.h>
namespace geode {

struct TimedCallHelper {
  std::string msg;
  real start_time = get_time();
  TimedCallHelper(const string& new_msg)
   : msg(new_msg)
  { }
  ~TimedCallHelper() {
    const auto end_time = get_time();
    std::cout << msg << " took " << (end_time - start_time) << " seconds.\n";
  }
};

template<class Fn, class... Args> auto timed_call(const string& msg, Fn&& fn, Args&&... args) -> decltype(fn(std::forward<Args>(args)...)) {
  GEODE_UNUSED const auto& helper = TimedCallHelper{msg};
  return fn(std::forward<Args>(args)...);
}

// Call a function printing execution time to cout
#define GEODE_TIMED_CALL(fn,...) geode::timed_call(#fn,fn,__VA_ARGS__)

} // geode namespace