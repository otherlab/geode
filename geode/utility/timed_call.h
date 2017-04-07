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

#if 0
// Record total time spent inside a block over the lifetime of the program
// This macro defines a static variable that accumulates execution time across multiple runs and a local variable that tracks the current run
// Usage of this is not thread safe
#define GEODE_TIMED_BLOCK(block_id) \
  static TimedBlock block_id{#block_id}; \
  const auto& block_id##_current_execution = TimedBlockExecutionScope{block_id};

class TimedBlock;
class TimedBlockExecutionScope {
 public:
  TimedBlock& block;
  const double start_time = get_time();
  static TimedBlock* currently_executing = nullptr;
  TimedBlockExecutionScope(const TimedBlockExecutionScope&) = delete;
  TimedBlockExecutionScope(TimedBlockExecutionScope&&) = delete;
  TimedBlockExecutionScope(TimedBlock& new_block);
  ~TimedBlockExecutionScope() {
    double end_time = get_time();
    block.consumed_time += (end_time - start_time);
  }
};

class TimedBlock {
 protected:
  TimedBlock* parent = nullptr;
  TimedBlock* first_child = nullptr;
  TimedBlock* last_child = nullptr;
  TimedBlock* prev_sibling = nullptr;
  TimedBlock* next_sibling = nullptr;
  double consumed_time = 0.; // self time + children time
  friend class TimedBlockExecutionScope;
 public:
  const std::string name;
  TimedBlock(const std::string new_name) : name(new_name) {
    if(TimedBlockExecutionScope::currently_executing) {
      TimedBlockExecutionScope::currently_executing->block.append_child(*this);
    }
  }
  TimedBlock(const TimedBlock&) = delete;
  TimedBlock(TimedBlock&&) = delete;

  void append_child(TimedBlock& new_child);
  void unlink();

  ~TimedBlock() { unlink(); }

  void print_summery(int current_depth=0) const;
};
#endif

} // geode namespace