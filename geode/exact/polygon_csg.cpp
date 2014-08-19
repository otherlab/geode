// Robust constructive solid geometry for polygons in the plane
#include <geode/array/ConstantMap.h>
#include <geode/exact/polygon_csg.h>
#include <geode/exact/constructions.h>
#include <geode/exact/ExactSegmentGraph.h>
#include <geode/exact/predicates.h>
#include <geode/exact/perturb.h>
#include <geode/exact/quantize.h>
#include <geode/exact/scope.h>
#include <geode/array/amap.h>
#include <geode/array/sort.h>
#include <geode/geometry/BoxTree.h>
#include <geode/geometry/polygon.h>
#include <geode/geometry/traverse.h>
#include <geode/structure/Hashtable.h>
#include <geode/utility/Log.h>
#include <geode/utility/str.h>
#include <geode/utility/time.h>

namespace geode {

std::string str(const FillRule rule) {
  switch (rule) {
    case FillRule::Greater: return "Greater";
    case FillRule::Parity: return "Parity";
    case FillRule::NotEqual: return "NotEqual";
  }
}

typedef exact::Vec2 EV;
using exact::Perturbed2;
using Log::cout;
using std::endl;

static Array<Box<EV>> segment_boxes(RawArray<const int> next, RawArray<const EV> X) {
  Array<Box<EV>> boxes(X.size(),uninit);
  for (int i=0;i<X.size();i++)
    boxes[i] = bounding_box(X[i],X[next[i]]);
  return boxes;
}

// Does x1 + t*dir head outwards from the local polygon portion x0,x1,x2?
// This version is specialized to dir = (1,0): does x1 + t*(1,0) head outwards from x0,x1,x2?
static inline bool local_outwards_x_axis(const Perturbed2 x0, const Perturbed2 x1, const Perturbed2 x2) {
  // If x1 is convex,  we're outwards if dir is to the right of *either* segment.
  // If x1 is concave, we're outwards if dir is to the right of *both* segments.
  const bool out0 = upwards(x0,x1),
             out1 = upwards(x1,x2);
  return out0==out1 ? out0 : triangle_oriented(x0,x1,x2);
}

Nested<EV> exact_split_polygons(Nested<const EV> polys, const int depth) {
  IntervalScope scope;
  RawArray<const EV> X = polys.flat;

  // We index segments by the index of their first point in X.  For convenience, we make an array to keep track of wraparounds.
  Array<int> next = (arange(X.size())+1).copy();
  for (int i=0;i<polys.size();i++) {
    GEODE_ASSERT(polys.size(i)>=3,"Degenerate polygons are not allowed");
    next[polys.offsets[i+1]-1] = polys.offsets[i];
  }

  // Compute all nontrivial intersections between segments
  struct Pairs {
    const BoxTree<EV>& tree;
    RawArray<const int> next;
    RawArray<const EV> X;
    Array<Vector<int,2>> pairs;

    Pairs(const BoxTree<EV>& tree, RawArray<const int> next, RawArray<const EV> X)
      : tree(tree), next(next), X(X) {}

    bool cull(const int n) const { return false; }
    bool cull(const int n0, const int box1) const { return false; }
    void leaf(const int n) const { assert(tree.prims(n).size()==1); }

    void leaf(const int n0, const int n1) {
      assert(tree.prims(n0).size()==1 && tree.prims(n1).size()==1);
      const int i0 = tree.prims(n0)[0], i1 = next[i0],
                j0 = tree.prims(n1)[0], j1 = next[j0];
      if (!(i0==j0 || i0==j1 || i1==j0 || i1==j1)) {
        const auto a0 = Perturbed2(i0,X[i0]), a1 = Perturbed2(i1,X[i1]),
                   b0 = Perturbed2(j0,X[j0]), b1 = Perturbed2(j1,X[j1]);
        if (segments_intersect(a0,a1,b0,b1))
          pairs.append(vec(i0,j0));
      }
    }
  };
  const auto tree = new_<BoxTree<EV>>(segment_boxes(next,X),1);
  Pairs pairs(tree,next,X);
  double_traverse(*tree,pairs);

  // Group intersections by segment.  Each pair is added twice: once for each order.
  Array<int> counts(X.size());
  for (auto pair : pairs.pairs) {
    counts[pair.x]++;
    counts[pair.y]++;
  }
  Nested<int> others(counts,uninit);
  for (auto pair : pairs.pairs) {
    others(pair.x,--counts[pair.x]) = pair.y;
    others(pair.y,--counts[pair.y]) = pair.x;
  }
  pairs.pairs.clean_memory();
  counts.clean_memory();

  // Walk all original polygons, recording which subsegments occur in the final result
  Hashtable<Vector<int,2>,int> graph; // If (i,j) -> k, the output contains the portion of segment j from ij to jk
  for (const int p : range(polys.size())) {
    const auto poly = range(polys.offsets[p],polys.offsets[p+1]);
    // Compute the depth of the first point in the polygon by firing a ray along the positive x axis.
    struct Depth {
      const BoxTree<EV>& tree;
      RawArray<const int> next;
      RawArray<const EV> X;
      const Perturbed2 start;
      int depth;

      Depth(const BoxTree<EV>& tree, RawArray<const int> next, RawArray<const EV> X, const int prev, const int i)
        : tree(tree), next(next), X(X)
        , start(i,X[i])
        // If we intersect no other segments, the depth depends on the orientation of direction = (1,0) relative to segments prev and i
        , depth(-!local_outwards_x_axis(Perturbed2(prev,X[prev]),start,Perturbed2(next[i],X[next[i]]))) {}

      bool cull(const int n) const {
        const auto box = tree.boxes(n);
        return box.max.x<start.value().x || box.max.y<start.value().y || box.min.y>start.value().y;
      }

      void leaf(const int n) {
        assert(tree.prims(n).size()==1);
        const int i0 = tree.prims(n)[0], i1 = next[i0];
        if (start.seed()!=i0 && start.seed()!=i1) {
          const auto a0 = Perturbed2(i0,X[i0]),
                     a1 = Perturbed2(i1,X[i1]);
          const bool above0 = upwards(start,a0),
                     above1 = upwards(start,a1);
          if (above0!=above1 && above1==triangle_oriented(a0,a1,start))
            depth += above1 ? 1 : -1;
        }
      }
    };
    Depth ray(tree,next,X,poly.back(),poly[0]);
    single_traverse(*tree,ray);

    // Walk around the polygon, recording all subsegments at the desired depth
    int delta = ray.depth-depth;
    int prev = poly.back();
    for (const int i : poly) {
      const int j = next[i];
      const Vector<Perturbed2,2> segment(Perturbed2(i,X[i]),Perturbed2(j,X[j]));
      const auto other = others[i];
      // Sort intersections along this segment
      if (other.size() > 1) {
        struct PairOrder {
          RawArray<const int> next;
          RawArray<const EV> X;
          const Vector<Perturbed2,2> segment;

          PairOrder(RawArray<const int> next, RawArray<const EV> X, const Vector<Perturbed2,2>& segment)
            : next(next), X(X), segment(segment) {}

          bool operator()(const int j, const int k) const {
            if (j==k)
              return false;
            const int jn = next[j],
                      kn = next[k];
            return segment_intersections_ordered(segment.x,segment.y,
                                                 Perturbed2(j,X[j]),Perturbed2(jn,X[jn]),
                                                 Perturbed2(k,X[k]),Perturbed2(kn,X[kn]));
          }
        };
        sort(other,PairOrder(next,X,segment));
      }
      // Walk through each intersection of this segment, updating delta as we go and remembering the subsegment if it has the right depth
      for (const int o : other) {
        if (!delta)
          graph.set(vec(prev,i),o);
        const int on = next[o];
        delta += segment_directions_oriented(segment.x,segment.y,Perturbed2(o,X[o]),Perturbed2(on,X[on])) ? -1 : 1;
        prev = o;
      }
      if (!delta)
        graph.set(vec(prev,i),next[i]);
      // Advance to the next segment
      prev = i;
    }
  }

  // Walk the graph to produce output polygons
  Hashtable<Vector<int,2>> seen;
  Nested<EV,false> output;
  for (const auto& start : graph)
    if (seen.set(start.x)) {
      auto ij = start.x;
      for (;;) {
        const int i = ij.x, j = ij.y, in = next[i], jn = next[j];
        output.flat.append(j==next[i] ? X[j] : segment_segment_intersection(Perturbed2(i,X[i]),Perturbed2(in,X[in]),Perturbed2(j,X[j]),Perturbed2(jn,X[jn])));
        ij = vec(j,graph.get(ij));
        if (ij == start.x)
          break;
        seen.set(ij);
      }
      output.offsets.append(output.flat.size());
    }
  return output;
}

static inline bool include_face(int delta, const FillRule rule) {
  switch(rule) {
    case FillRule::Greater: return delta > 0;
    case FillRule::Parity: return (delta & 1);
    case FillRule::NotEqual: return !delta;
  }
}

Nested<Vec2> split_polygons(Nested<const Vec2> polys, const int depth) {
  const auto quant = quantizer(bounding_box(polys));
  return amap(quant.inverse,exact_split_polygons(amap(quant,polys),depth));
}

Nested<Vec2> exact_split_polygons_with_rule(Nested<const Vec2> polys, const int depth, const FillRule rule) {
  const auto g = ExactSegmentGraph(polys);
  const auto edge_windings = Field<int, EdgeId>(constant_map(g.topology->n_edges(), 1).copy());
  const auto face_winding_depths = compute_winding_numbers(g.topology, g.boundary_face(), edge_windings);
  auto included_faces = Field<bool, FaceId>(g.topology->n_faces(), uninit);
  for(const FaceId fid : included_faces.id_range())
    included_faces[fid] = include_face(face_winding_depths[fid] - depth, rule);
  const auto contours = extract_region(g.topology, included_faces);
  auto result = Nested<Vec2>::empty_like(contours);
  for(int i : range(contours.flat.size())) {
    const auto he = contours.flat[i];
    result.flat[i] = g.vertices[g.topology->src(he)].approx;
  }
  return result;
}

Nested<Vec2> split_polygons_with_rule(Nested<const Vec2> polys, const int depth, const FillRule rule) {
  const auto quant = quantizer(bounding_box(polys));
  return amap(quant.inverse,exact_split_polygons_with_rule(amap(quant,polys),depth,rule));
}

Nested<Vec2> split_polygons_greater(Nested<const Vec2> polys, const int depth) {
 return split_polygons_with_rule(polys,depth,FillRule::Greater); }
Nested<Vec2> split_polygons_parity(Nested<const Vec2> polys, const int depth) {
 return split_polygons_with_rule(polys,depth,FillRule::Parity); }
Nested<Vec2> split_polygons_neq(Nested<const Vec2> polys, const int depth) {
 return split_polygons_with_rule(polys,depth,FillRule::NotEqual); }

static bool segment_intersections_possibly_same(const Vec2 v0, const Vec2 v1) {
  return (v0 - v1).maxabs() <= segment_segment_intersection_threshold;
}
static bool all_close(const Nested<const Vec2> p0, const Nested<const Vec2> p1) {
  if(p0.offsets != p1.offsets)
    return false;
  assert(p0.flat.size() == p1.flat.size());
  for(const int i : range(p0.flat.size())) {
    if(!segment_intersections_possibly_same(p0.flat[i],p1.flat[i]))
      return false;
  }
  return true;
}

Nested<Vec2> compare_splitting_algorithms(Nested<const Vec2> polys, const int depth) {
  // Call into non-graph based implementation
  const auto t0 = get_time();
  const auto simple_result = canonicalize_polygons(split_polygons(polys, depth));
  const auto simple_time = get_time() - t0;
  // Call into graph based implementation
  const auto t1 = get_time();
  const auto graph_result = canonicalize_polygons(split_polygons_greater(polys,depth));
  const auto graph_time = get_time() - t1;
  std::cout << "Time using graph  : " << graph_time << ", Time without graph: " << simple_time << '\n'
            << format("input N: %d, output N: %d, graph/without: %f\n", polys.total_size(), graph_result.total_size(), graph_time/simple_time);

  // Orders of arguments to segment_segment_intersection can slightly alter constructed points
  // If minimal points found in canonicalize_polygons are different this check could fail (current test cases don't trigger this)
  // If this assert is only triggered when FORCE_CANONICAL_CONSTRUCTION_ARGUMENTS is false, it might be necessary to implement more robust matching
  GEODE_ASSERT(all_close(graph_result,simple_result));
  #if FORCE_CANONICAL_CONSTRUCTION_ARGUMENTS
  // With FORCE_CANONICAL_CONSTRUCTION_ARGUMENTS, results should be bit-for-bit identical
  GEODE_ASSERT(graph_result == simple_result);
  #endif

  return graph_result;
}

} // namespace geode
