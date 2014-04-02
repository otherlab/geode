// Triangulation of segments in the plane using sweep and orientation tests.
// Intended for use inside mesh CSG; use Delaunay when triangulating normal points in a plane.
#pragma once

#include <geode/exact/config.h>
#include <geode/exact/debug.h>
#include <geode/mesh/TriangleTopology.h>
namespace geode {

/*
 * These triangulation routines make use of a policy class defining the necessary predicates and
 * constructions.  Both triangulate_monotone_polygon and add_constraint_edge make use of the two predicates
 *   bool below(VertexId, VertexId) const
 *   bool triangle_oriented(VertexId, VertexId, VertexId) const
 * In order to support intersecting constraints, add_constraint_edge uses the construction routine
 *   VertexId construct_segment_intersection(MutableTriangleTopology&, VertexId, VertexId, VertexId, VertexId)
 * as well as the following predicate which is a special case of triangle_oriented:
 *   bool line_point_oriented(Line, VertexId) const
 */

// Triangulate an upwards monotone polygon consisting of left and right contours between bottom and top vertices.
// Any triangles created are added to the given mesh, flipped if desired.  The mesh is otherwise untouched.
// The stack is used as temporary storage.
template<class Policy,class Left,class Right> static void
triangulate_monotone_polygon(const Policy& P, MutableTriangleTopology& mesh,
                             const VertexId lo, const VertexId hi,
                             const Left& left, const Right& right,
                             Array<VertexId>& stack) {
  if (CHECK) {
    // Verify that both contours are monotone
    auto u = lo;
    for (const auto v : left) {
      GEODE_ASSERT(P.below(u,v));
      u = v;
    }
    GEODE_ASSERT(P.below(u,hi));
    u = lo;
    for (const auto v : right) {
      GEODE_ASSERT(P.below(u,v));
      u = v;
    }
    GEODE_ASSERT(P.below(u,hi));
  }

  // If there are no vertices on either side, there's nothing to do.
  // Polygons with two vertices have zero triangles.
  const int Ln = left.size(),
            Rn = right.size();
  if (!Ln && !Rn)
    return;

  // Maintain a stack of vertices in one or the other chain, curving in the wrong direction
  stack.copy(vec(lo));

  // Process all vertices in upwards order
  int stack_side = 0; // -1 for left, 0 for lo/hi, +1 for right
  int L = 0, R = 0;
  for (;;) {
    // Pick the next vertex
    VertexId next;
    int next_side;
    if (L==Ln && R==Rn) {
      next = hi;
      next_side = 0;
    } else if (R==Rn || (L<Ln && P.below(left[L],right[R]))) {
      next = left[L++];
      next_side = -1;
    } else {
      next = right[R++];
      next_side = +1;
    }

    if (stack_side && stack_side!=next_side) {
      // We've switched chains, so add all triangles with the new vertex against the old chain
      assert(stack.size() >= 2);
      for (int i=1;i<stack.size();i++)
        mesh.add_face(stack_side<0 ? vec(next,stack[i],stack[i-1])
                                   : vec(next,stack[i-1],stack[i]));
      if (next==hi)
        return;
      stack.copy(vec(stack.back(),next));
    } else {
      // Compact the stack until it curves all in the same direction
      while (stack.size() >= 2) {
        const auto a = stack[stack.size()-2],
                   b = stack.back();
        if ((stack_side<0) == P.triangle_oriented(next,a,b))
          break;
        mesh.add_face(stack_side<0 ? vec(next,b,a)
                                   : vec(next,a,b));
        stack.pop();
      }

      // Add the new vertex to our stack
      stack.append(next);
    }
    stack_side = next_side;
  }
}

// Constrain an edge, respecting the orientation of lines
template<class Policy,class Line> static inline void
constrain(const Policy& P, Hashtable<Vector<VertexId,2>,Line>& constrained, VertexId v0, VertexId v1, Line v01) {
  if (v1 < v0) {
    swap(v0,v1);
    v01 = P.reverse_line(v01);
  }
  constrained.set(vec(v0,v1),v01);
}

template<class Policy,class Line> static void
add_constraint_edge(Policy& P, MutableTriangleTopology& mesh,
                    Hashtable<Vector<VertexId,2>,Line>& constrained,
                    VertexId v0, VertexId v1, Line v01) {
  // Some code copied from add_constraint_edges in delaunay.cpp
  GEODE_ASSERT(mesh.valid(v0) && mesh.valid(v1));

  {
    // Check if the edge already exists in the triangulation.  To ensure optimal complexity,
    // we loop around both vertices interleaved so that our time is O(min(degree(v0),degree(v1))).
    const auto s0 = mesh.halfedge(v0),
               s1 = mesh.halfedge(v1);
    {
      auto e0 = s0,
           e1 = s1;
      do {
        if (mesh.dst(e0)==v1 || mesh.dst(e1)==v0)
          goto success; // The edge already exists, so there's nothing to be done
        e0 = mesh.left(e0);
        e1 = mesh.left(e1);
      } while (e0!=s0 && e1!=s1);
    }

    // Find a triangle touching v0 or v1 containing part of the v0-v1 segment.
    // As above, we loop around both vertices interleaved
    auto e0 = s0;
    {
      auto e1 = s1;
      if (mesh.is_boundary(e0)) e0 = mesh.left(e0);
      if (mesh.is_boundary(e1)) e1 = mesh.left(e1);
      const auto e0d = mesh.dst(e0),
                 e1d = mesh.dst(e1);
      bool e0o = !P.line_point_oriented(v01,e0d),
           e1o =  P.line_point_oriented(v01,e1d);
      for (;;) { // No need to check for an end condition, since we're guaranteed to terminate
        const auto n0 = mesh.left(e0),
                   n1 = mesh.left(e1);
        const auto n0d = mesh.dst(n0),
                   n1d = mesh.dst(n1);
        const bool n0o = !P.line_point_oriented(v01,n0d),
                   n1o =  P.line_point_oriented(v01,n1d);
        if (e0o && !n0o)
          break;
        if (e1o && !n1o) {
          // Swap v0 with v1 and e0 with e1 so that our ray starts at v0
          swap(v0,v1);
          swap(e0,e1);
          v01 = P.reverse_line(v01);
          break;
        }
        e0 = n0;
        e1 = n1;
        e0o = n0o;
        e1o = n1o;
      }
    }

    // Walk from v0 to v1, retriangulating the cavities as we go.  For each cavity,
    // we maintain a stack of vertices curving away from our new constraint edge.
    // If the constraint hits a previous constrained edge, we must create a new
    // face-face-face vertex at the intersection.
    //
    // Instead of storing the stacks explicitly, we store each one as "bent" edges in
    // the mesh, together with the total stack size.  The flips required to create
    // these bent edges are occasionally invalid, producing multiple distinct edges
    // between the same vertex pair.  Since everything will be valid by the end of the
    // walk, we dodge this problem by calling unsafe_flip_edge directly.
    int left = 0,
        right = 0;
    auto cut = mesh.reverse(mesh.next(e0));
    for (;;) {
      // If we intersect a constraint edge, create a new vertex at the intersection
      VertexId mid;
      const auto c = mesh.vertices(cut).sorted();
      if (const Line* c01p = constrained.get_pointer(c)) {
        const auto c01 = *c01p;
        mid = P.construct_segment_intersection(mesh,v01,c01);
        constrain(P,constrained,c.x,mid,c01);
        constrain(P,constrained,mid,c.y,c01);
        constrain(P,constrained,v0,mid,v01);
        // Here we use the fact that split_edge(h) changes h to be the first half
        mesh.split_edge(cut,mid);
        cut = mesh.reverse(mesh.prev(mesh.reverse(cut)));
      } else {
        mid = v1;
        assert(!mesh.is_boundary(cut) && !mesh.is_boundary(mesh.reverse(cut)));
        cut = mesh.unsafe_flip_edge(cut); // cut is now on the right
      }
      const auto v = mesh.src(cut);
      if (v == mid) {
        // If necessary, restart the walk on the other side of the newly created intersection point
        const auto left_e = mesh.prev(mesh.reverse(cut)),
                   right_e = mesh.next(cut);
        bool done = true;
        if (v1 != mid) {
          v0 = mid;
          cut = mesh.left(cut);
          const auto u = mesh.opposite(cut);
          if (u != v1) {
            done = false;
            if (!P.line_point_oriented(v01,u))
              cut = mesh.left(cut);
            cut = mesh.reverse(mesh.next(cut));
          }
        }
        // Finish off left cavity
        for (auto e = left_e; left--;)
          e = mesh.next(mesh.unsafe_flip_edge(e));
        // Finish off right cavity
        for (auto e = right_e; right--;)
          e = mesh.prev(mesh.reverse(mesh.unsafe_flip_edge(e)));
        if (done)
          goto success;
        left = right = 0;
      } else if (P.line_point_oriented(v01,v)) {
        // Add new vertex to left cavity, and reduce
        left++;
        const auto u2 = mesh.src(cut);
        while (left) {
          const auto e = mesh.prev(mesh.reverse(cut));
          const auto u0 = left>1 ? mesh.opposite(mesh.reverse(e)) : v0,
                     u1 = mesh.src(e);
          if (P.triangle_oriented(u0,u1,u2))
            break;
          if (--left)
            (void)mesh.unsafe_flip_edge(e);
        }
        cut = mesh.reverse(mesh.prev(cut));
      } else {
        // Add new vertex to right cavity, and reduce
        cut = mesh.reverse(cut); // Move cut to other side so that it's stable
        right++;
        const auto u2 = mesh.dst(cut);
        while (right) {
          const auto e = mesh.next(mesh.reverse(cut));
          const auto u0 = right>1 ? mesh.opposite(mesh.reverse(e)) : v0,
                     u1 = mesh.dst(e);
          if (P.triangle_oriented(u0,u2,u1)) // Order used by CASE(FFF,FFF,FFF) in mesh_csg.cpp
            break;
          if (--right)
            (void)mesh.unsafe_flip_edge(e);
        }
        cut = mesh.reverse(mesh.next(cut));
      }
    }
  }

  success:
  constrain(P,constrained,v0,v1,v01);
}

// For testing purposes
void simple_triangulate_test(const int seed, const int left, const int right,
                                             const int interior, const int edges);

}
