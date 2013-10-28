//#####################################################################
// Class BoxTree
//#####################################################################
//
// BoxTree represents a bounding box hierarchy as a complete binary tree, with
// leaves containing a user-specified number of primitives.
//
// We store the topology of the tree as a complete binary tree packed into
// an array, so node n has parent (n-1)/2 and children 2n+1 and 2n+2.
//
// For templatized visitor-based traversal, include traversal.h.
//
//#####################################################################
#pragma once

#include <geode/array/Array.h>
#include <geode/geometry/Box.h>
#include <geode/python/Object.h>
#include <geode/vector/Vector.h>
#include <geode/utility/range.h>
namespace geode {

template<class TV> class BoxTree : public Object
{
  typedef typename TV::Scalar T;
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;

  const int leaf_size;
  const Range<int> leaves;
  const int depth; // max path size from root to leaf counting both ends
  const Array<const int> p; // index permutation
  const Array<const Range<int>> ranges;
  const Array<Box<TV>> boxes;

protected:
  GEODE_CORE_EXPORT BoxTree(RawArray<const TV> geo,int leaf_size);
  GEODE_CORE_EXPORT BoxTree(RawArray<const Box<TV>> geo,int leaf_size);
  GEODE_CORE_EXPORT BoxTree(const BoxTree<TV>& other); // Shares ownership with everything except boxes
public:
  ~BoxTree();

  int nodes() const
  {return boxes.size();}

  Box<TV> bounding_box() const
  {return boxes.size()?boxes[0]:Box<TV>();}

  bool is_leaf(int node) const {
    assert(ranges.valid(node));
    return node>=leaves.lo;
  }

  RawArray<const int> prims(int leaf) const {
    assert(is_leaf(leaf));
    return p.slice(ranges[leaf].lo,ranges[leaf].hi);
  }

  static int parent(int node)
  {assert(node>0);return (node-1)/2;}

  static int child(int node,int i)
  {assert(unsigned(i)<2);return 2*node+1+i;}

  GEODE_CORE_EXPORT void update_nonleaf_boxes();
  void check(RawArray<const TV> x) const;

  // Warning: Doesn't know about structure without each tree leaf
  template<class Shape>
  GEODE_CORE_EXPORT bool any_box_intersection(const Shape& shape) const;
};

}
