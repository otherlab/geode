//#####################################################################
// Class Stack
//#####################################################################
#pragma once

#include <geode/array/Array.h>
namespace geode {

template<class T>
class Stack {
public:
  Array<T> array;

  Stack() {}

  void clean_memory() {
    array.clean_memory();
  }

  void preallocate(const int max_size) {
    array.preallocate(max_size);
  }

  void increase_size(const int size) {
    array.preallocate(size+array.m);
  }

  void compact() {
    array.compact();
  }

  void push(const T& element) GEODE_ALWAYS_INLINE {
    array.append(element);
  }

  T pop() {
    return array.pop();
  }

  const T& peek() const {
    return array.back();
  }

  bool empty() const {
    return array.m==0;
  }

  void clear() {
    array.clear();
  }

  void swap(Stack& stack) {
    array.swap(stack.array);
  }
};

}
namespace std {
template<class T> void swap(geode::Stack<T>& stack1,geode::Stack<T>& stack2) {
  stack1.swap(stack2);
}
}
