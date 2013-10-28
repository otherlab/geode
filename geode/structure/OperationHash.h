//#####################################################################
// Class OperationHash
//#####################################################################
#pragma once

#include <geode/array/Array.h>
namespace geode {

class OperationHash {
public:
  Array<unsigned int> operations;
  unsigned int current_operation;

  OperationHash(const int m=0)
    : operations(m), current_operation(1) {}

  void initialize(const int m) {
    if (m==operations.size())
      return next_operation();
    operations.resize(m,false,false);
    operations.fill(0);
    current_operation = 1;
  }

  void mark(const int i) {
    operations[i] = current_operation;
  }

  void unmark(const int i) {
    operations[i] = 0;
  }

  bool is_marked_current(const int i) const {
    return operations[i]==current_operation;
  }

  void next_operation() {
    current_operation++;
    if (!current_operation) {
      current_operation = 1;
      operations.fill(0); // reset everything if overflow
    }
  }

  void remove_duplicates(Array<int>& list) {
    next_operation();
    for (int i=list.size()-1;i>1;i--) {
      if (is_marked_current(list[i]))
        list.remove_index_lazy(i);
      else
        mark(list[i]);
    }
    next_operation();
  }
};
}
