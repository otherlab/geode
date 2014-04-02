// A flexible binary heap template

#include <geode/structure/Heap.h>
#include <geode/array/Array.h>
namespace geode {

namespace {
struct HeapTest : public HeapBase<HeapTest> {
  typedef HeapBase<HeapTest> Base;
  Array<int> heap;

  HeapTest(Array<int> heap)
    : heap(heap) {}

  int size() const { return heap.size(); }
  bool first(const int i, const int j) const { return heap[i] <= heap[j]; }
  void swap(const int i, const int j) { std::swap(heap[i],heap[j]); }

  int pop() {
    GEODE_ASSERT(size());
    const auto x = heap[0];
    swap(0,size()-1);
    heap.pop();
    if (size())
      Base::move_downward(0);  
    return x;
  }
};
}

Array<int> heapsort_test(RawArray<const int> input) {
  HeapTest H(input.copy());
  H.make();
  GEODE_ASSERT(H.is_heap());
  Array<int> order; 
  while (H.size()) {
    order.append(H.pop());
    GEODE_ASSERT(H.is_heap());
  }
  return order;
}

}
