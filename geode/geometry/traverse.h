// Visitor-based box tree traversal
//
// All functions are static since specific visitor instantiations should appear only in .cpp files.
#pragma once

#include <geode/array/alloca.h>
#include <geode/array/RawStack.h>
#include <geode/array/view.h>
#include <geode/geometry/BoxTree.h>
namespace geode {

// Traverse one box tree.  There is no automatic culling: the visitor is responsible for everything.
template<class Visitor,class TV> static void single_traverse(const BoxTree<TV>& tree, Visitor&& visitor) {
  if (!tree.nodes())
    return;
  const int internal = tree.leaves.lo;
  RawStack<int> stack(GEODE_RAW_ALLOCA(tree.depth,int));
  stack.push(0);
  while (stack.size()) {
    const int n = stack.pop();
    if (visitor.cull(n))
      continue;
    if (n < internal) {
      stack.push(2*n+1);
      stack.push(2*n+2);
    } else
      visitor.leaf(n);
  }
}

// Helper function for doubly recursive traversal of two box trees starting at given nodes
template<class Visitor,class Thickness,class TV> static void
double_traverse_helper(const BoxTree<TV>& tree0, const BoxTree<TV>& tree1, Visitor&& visitor,
                       RawStack<Vector<int,2>> stack, const int n0, const int n1, Thickness thickness) {
  assert((is_same<Thickness,Zero>::value || thickness));
  const int internal0 = tree0.leaves.lo,
            internal1 = tree1.leaves.lo;
  RawArray<const Box<TV>> boxes0 = tree0.boxes,
                          boxes1 = tree1.boxes;
  stack.push(vec(n0,n1));
  while (stack.size()) {
    const auto n = stack.pop();
    if (visitor.cull(n.x,n.y) || !boxes0[n.x].intersects(boxes1[n.y],thickness))
      continue;
    if (n.x < internal0) {
      if (n.y < internal1) {
        stack.push(vec(2*n.x+1,2*n.y+1));
        stack.push(vec(2*n.x+1,2*n.y+2));
        stack.push(vec(2*n.x+2,2*n.y+1));
        stack.push(vec(2*n.x+2,2*n.y+2));
      } else {
        stack.push(vec(2*n.x+1,n.y));
        stack.push(vec(2*n.x+2,n.y));
      }
    } else {
      if (n.y < internal1) {
        stack.push(vec(n.x,2*n.y+1));
        stack.push(vec(n.x,2*n.y+2));
      } else
        visitor.leaf(n.x,n.y);
    }
  }
}

// Helper function traversing two hierarchies starting at the roots.  Identical trees are not treated specially.
template<class Visitor,class Thickness,class TV> static void
double_traverse_helper(const BoxTree<TV>& tree0, const BoxTree<TV>& tree1, Visitor&& visitor, Thickness thickness) {
  if (!tree0.nodes() || !tree1.nodes())
    return;
  const int buffer_size = 3*max(tree0.depth,tree1.depth);
  RawStack<Vector<int,2>> stack(GEODE_RAW_ALLOCA(buffer_size,Vector<int,2>));
  double_traverse_helper(tree0,tree1,visitor,stack,0,0,thickness);
}

// Helper function traversing a hierarchy against itself starting at the roots.
template<class Visitor,class Thickness,class TV> static void
double_traverse_helper(const BoxTree<TV>& tree, Visitor&& visitor, Thickness thickness) {
  if (!tree.nodes())
    return;
  const int internal = tree.leaves.lo;
  RawStack<int> stack(GEODE_RAW_ALLOCA(6*tree.depth,int));
  stack.push(0);
  while (stack.size()) {
    const int n = stack.pop();
    if (visitor.cull(n))
      continue;
    if (n < internal) {
      stack.push(2*n+1);
      stack.push(2*n+2);
      const int s = stack.size();
      RawStack<Vector<int,2>> rest(vector_view<2>(stack.data.slice(s,s+((stack.data.size()-s)&~1))));
      double_traverse_helper(tree,tree,visitor,rest,2*n+1,2*n+2,thickness);
    } else
      visitor.leaf(n);
  }
}

// Traverse all intersecting pairs of leaf boxes between two distinct hierarchies.  Box/box intersection culling
// is automatic, but the visitor can provide additional culling by returning true from visitor.cull(...).
template<class Visitor,class TV> static void
double_traverse(const BoxTree<TV>& tree0, const BoxTree<TV>& tree1, Visitor&& visitor, typename TV::Scalar thickness) {
  GEODE_ASSERT(&tree0 != &tree1,"Identical trees should use the dedicated routine below");
  if (thickness)
    double_traverse_helper(tree0,tree1,visitor,thickness);
  else
    double_traverse_helper(tree0,tree1,visitor,Zero());
}
template<class Visitor,class TV> static void
double_traverse(const BoxTree<TV>& tree0, const BoxTree<TV>& tree1, Visitor&& visitor) {
  GEODE_ASSERT(&tree0 != &tree1,"Identical trees should use the dedicated routine below");
  double_traverse_helper(tree0,tree1,visitor,Zero());
}

// Traverse all intersecting pairs of leaf boxes between a hierarchy and itself.
template<class Visitor,class TV> static void
double_traverse(const BoxTree<TV>& tree, Visitor&& visitor, typename TV::Scalar thickness) {
  if (thickness)
    double_traverse_helper(tree,visitor,thickness);
  else
    double_traverse_helper(tree,visitor,Zero());
}
template<class Visitor,class TV> static void double_traverse(const BoxTree<TV>& tree, Visitor&& visitor) {
  double_traverse_helper(tree,visitor,Zero());
}

}
