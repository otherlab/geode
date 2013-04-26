//#####################################################################
// Templatized visitor-based box tree traversal
//#####################################################################
//
// All functions are static since specific visitor instantiations should
// appear only in .cpp files.
//
//#####################################################################
#pragma once

#include <other/core/array/alloca.h>
#include <other/core/array/RawStack.h>
#include <other/core/array/view.h>
#include <other/core/geometry/BoxTree.h>
namespace other {

// Traverse one box tree.  There is no automatic culling: the visitor is responsible for everything.
template<class Visitor,class TV> static void traverse(const BoxTree<TV>& tree, Visitor&& visitor) {
  if (!tree.nodes())
    return;
  const int internal = tree.leaves.lo;
  RawStack<int> stack(OTHER_RAW_ALLOCA(tree.depth,int));
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
template<class Visitor,class Thickness,class TV> static void traverse_helper(const BoxTree<TV>& tree0, const BoxTree<TV>& tree1, Visitor&& visitor, RawStack<Vector<int,2>> stack, int n0, int n1, Thickness thickness) {
  assert((boost::is_same<Thickness,Zero>::value || !!thickness));
  const int internal0 = tree0.leaves.lo,
            internal1 = tree1.leaves.lo;
  RawArray<const Box<TV>> boxes0 = tree0.boxes,
                          boxes1 = tree1.boxes;
  stack.push(vec(n0,n1));
  while (stack.size()) {
    int n0,n1;stack.pop().get(n0,n1);
    if (visitor.cull(n0,n1) || !boxes0[n0].intersects(boxes1[n1],thickness))
      continue;
    if (n0 < internal0) {
      if (n1 < internal1) {
        stack.push(vec(2*n0+1,2*n1+1));
        stack.push(vec(2*n0+1,2*n1+2));
        stack.push(vec(2*n0+2,2*n1+1));
        stack.push(vec(2*n0+2,2*n1+2));
      } else {
        stack.push(vec(2*n0+1,n1));
        stack.push(vec(2*n0+2,n1));
      }
    } else {
      if (n1 < internal1) {
        stack.push(vec(n0,2*n1+1));
        stack.push(vec(n0,2*n1+2));
      } else
        visitor.leaf(n0,n1);
    }
  }
}

// Helper function traversing two hierarchies starting at the roots.  Identical trees are not treated specially.
template<class Visitor,class Thickness,class TV> static void traverse_helper(const BoxTree<TV>& tree0, const BoxTree<TV>& tree1, Visitor&& visitor, Thickness thickness) {
  if (!tree0.nodes() || !tree1.nodes())
    return;
  const int buffer_size = 3*max(tree0.depth,tree1.depth);
  RawStack<Vector<int,2>> stack(OTHER_RAW_ALLOCA(buffer_size,Vector<int,2>));
  traverse_helper(tree0,tree1,visitor,stack,0,0,thickness);
}

// Helper function traversing a hierarchy against itself starting at the roots.
template<class Visitor,class Thickness,class TV> static void traverse_helper(const BoxTree<TV>& tree, Visitor&& visitor, Thickness thickness) {
  if (!tree.nodes())
    return;
  const int internal = tree.leaves.lo;
  RawStack<int> stack(OTHER_RAW_ALLOCA(6*tree.depth,int));
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
      traverse_helper(tree,tree,visitor,rest,2*n+1,2*n+2,thickness);
    } else
      visitor.leaf(n);
  }
}

// Traverse all intersecting pairs of leaf boxes between two hierarchies.  Identical trees are not treated specially.
// Box/box intersection culling is automatic, but the visitor can add additional culling if desired by returning true from visitor.cull(...).
template<class Visitor,class TV> static void traverse(const BoxTree<TV>& tree0, const BoxTree<TV>& tree1, Visitor&& visitor, real thickness) {
  OTHER_ASSERT(&tree0 != &tree1); // Identical trees should almost certainly use the dedicated routine below.  We can remove this assertion if an application is found.
  if (thickness)
    traverse_helper(tree0,tree1,visitor,thickness);
  else
    traverse_helper(tree0,tree1,visitor,Zero());
}

// Traverse all intersecting pairs of leaf boxes between a hierarchy and itself.
template<class Visitor,class TV> static void traverse(const BoxTree<TV>& tree, Visitor&& visitor, real thickness) {
  if (thickness)
    traverse_helper(tree,visitor,thickness);
  else
    traverse_helper(tree,visitor,Zero());
}

}
