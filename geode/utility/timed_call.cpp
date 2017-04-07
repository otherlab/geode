#include "timed_call.h"
namespace geode {
#if 0
void TimedBlock::append_child(TimedBlock& new_child) {
  assert(new_child.prev_sibling == nullptr);
  assert(new_child.next_sibling == nullptr);
  assert(new_child.parent == nullptr);
  new_child.parent = this;
  if(!last_child) {
    assert(!first_child);
    last_child = &new_child;
    first_child = &new_child;
  }
  else {
    assert(last_child->next_sibling == nullptr);
    last_child->next_sibling = &new_child
    new_child.prev_sibling = last_child;
    last_child = &new_child;
  }
}

void TimedBlock::unlink() {
  if(parent && parent->first_child == this) {
    parent->first_child = next_sibling;
  }
  if(parent && parent->last_child == this) {
    parent->last_child = prev_sibling;
  }
  if(prev_sibling) {
    assert(prev_sibling->next_sibling == this);
    prev_sibling->next_sibling = next_sibling;
  }
  if(next_sibling) {
    assert(next_sibling->prev_sibling == this);
    next_sibling->prev_sibling = prev_sibling;
  }
  for(auto* child = first_child; child != nullptr; child = child->next_sibling) {
    assert(child->parent == this);
    child->parent = nullptr;
  }
  parent = nullptr;
  first_child = nullptr;
  last_child = nullptr;
  prev_sibling = nullptr;
  next_sibling = nullptr;
}
struct Percentage {
  double fraction;
  Percentage(double new_fraction) : fraction(new_fraction) { }
  static inline friend ostream& operator<<(ostream& os, const Percentage& rhs) {
    const auto old_width = os.width();
    const auto old_precision = os.precision();
    const auto old_flags = os.flags();
    os.setf(std::ios::fixed);
    os.precision(3);
    os.width(3);
    os << rhs.fraction*100. << '%';
    os.setf(old_flags);
    os.precision(old_precision);
    os.flags(old_flags);
  }
}

void TimedBlock::print_summery(int current_depth, double root_time, double parent_time) const {
  std::vector<TimedBlock*> children;
  auto self_time = consumed_time;
  for(auto* child = first_child; child != nullptr; child = child->next_sibling) {
    children.push_back(child);
    self_time -= child->consumed_time;
  }
  std::sort(children.begin(),children.end(),[](TimedBlock* lhs, TimedBlock* rhs) {
    return lhs->consumed_time < rhs->consumed_time;
  });
  const auto indent_string = std::string{current_depth*2,' '};
  std::cerr << indent_string;
  if(parent_time > 0) std::cerr << Percentage{consumed_time / parent_time} << ' '; 
  std::cerr << name << ": " << consumed_time << " seconds (" << self_time << " in self)";
  if(total_time > 0) std::cerr << ' ' << Percentage{consumed_time / root_time} << " of total";
  std::cerr << '\n';
  for(auto* child : children) {
    child->print_summery(current_depth+1, root_time, consumed_time);
  }
}
#endif
} // geode namespace