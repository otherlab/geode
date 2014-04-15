// Robust constructive solid geometry for circular arc polygons in the plane

#include <geode/array/convert.h>
#include <geode/array/sort.h>
#include <geode/exact/circle_csg.h>
#include <geode/exact/circle_predicates.h>
#include <geode/exact/scope.h>
#include <geode/exact/Exact.h>
#include <geode/exact/math.h>
#include <geode/exact/perturb.h>
#include <geode/geometry/BoxTree.h>
#include <geode/geometry/polygon.h>
#include <geode/geometry/traverse.h>
#include <geode/python/stl.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>

// Set to 1 to enable checks in quantization that ensure all tolerances were met
#define CHECK_CONSTRUCTIONS 0

namespace geode {

typedef RawArray<const ExactCircleArc> Arcs;
typedef RawArray<const int> Next;
typedef RawArray<const Vertex> Vertices;
typedef exact::Vec2 EV2;
using std::cout;
using std::endl;

static Array<Box<EV2>> arc_boxes(Next next, Arcs arcs, RawArray<const Vertex> vertices) {
  // Build bounding boxes for each arc
  Array<Box<EV2>> boxes(arcs.size(),false);
  for (int i1=0;i1<arcs.size();i1++) {
    const int i2 = next[i1];
    const auto v01 = vertices[i1],
               v12 = vertices[i2];
    boxes[i1] = arc_box(arcs, v01, v12);
  }
  return boxes;
}

namespace {
struct Info {
  Nested<const ExactCircleArc> arcs;

  // The following have one entry per arc
  Array<const int> next; // arcs.flat[i] is followed by arcs.flat[next[i]]
  Array<const Vertex> vertices; // vertices[i] is the start of arcs.flat[i]
  Array<const Box<EV2>> boxes; // Conservative bounding box for arcs.flat[i]

  // For each contour, a horizontal line that intersects it and the (relative) index of an arc it intersects
  Array<const HorizontalVertex> horizontals;
};
}

// Precompute information about a series of arc contours, and
static Info prune_small_contours(Nested<const ExactCircleArc> arcs) {
  // Precompute base information
  const auto next = closed_contours_next(arcs);
  const auto vertices = compute_vertices(arcs.flat,next);
  const auto boxes = arc_boxes(next,arcs.flat,vertices);

  // Find horizontal lines through each contour, and prune away contours that don't have any
  Nested<ExactCircleArc,false> pruned_arcs;
  Array<Vertex> pruned_vertices;
  Array<Box<EV2>> pruned_boxes;
  Array<HorizontalVertex> pruned_horizontals;
  for (const int c : range(arcs.size())) {
    const auto I = arcs.range(c);
    // Compute contour bounding box, and pick horizontal line through the middle
    Box<EV2> box;
    for (const auto b : boxes.slice(I))
      box.enlarge(b);
    const auto y = Quantized(floor((box.min.y+box.max.y)/2));
    // Check if the horizontal line hits the contour
    for (const int a : I) {
      if (circle_intersects_horizontal(arcs.flat,a,y)) {
        const Vertex& a01 = vertices[a];
        const Vertex& a12 = vertices[next[a]];
        const bool is_full_circle = arc_is_repeated_vertex(arcs.flat,a01,a12);
        for (const auto ay : circle_horizontal_intersections(arcs.flat,a,y)) {
          if (is_full_circle || circle_arc_contains_horizontal_intersection(arcs.flat,a01,a12,ay)) {
            // Success!  Add this contour our unpruned list
            const int shift = pruned_arcs.total_size()-arcs.offsets[c];
            pruned_arcs.append(arcs[c]);
            const auto verts = vertices.slice(I);
            for (auto& v : verts) {
              v.i0 += shift;
              v.i1 += shift;
            }
            pruned_vertices.extend(verts);
            pruned_boxes.extend(boxes.slice(I));
            pruned_horizontals.append(ay);
            pruned_horizontals.back().arc += shift;
            goto next_contour;
          }
        }
      }
    }
    next_contour:;
  }

  // All done!
  Info info;
  info.arcs = pruned_arcs;
  info.next = closed_contours_next(pruned_arcs);
  info.vertices = pruned_vertices;
  info.boxes = pruned_boxes;
  info.horizontals = pruned_horizontals;
  return info;
}

Nested<const ExactCircleArc> preprune_small_circle_arcs(Nested<const ExactCircleArc> arcs) {
  IntervalScope scope;
  return prune_small_contours(arcs).arcs;
}

Nested<ExactCircleArc> exact_split_circle_arcs(Nested<const ExactCircleArc> unpruned, const int depth) {
  // Check input consistency
  for (const int p : range(unpruned.size())) {
    const auto contour = unpruned[p];

    // This assert checks for contours that are a single repeated point
    // prune_small_contours will remove these, but they shouldn't be generated in the first place.
    assert(!(contour.size()==2 && contour[0].left!=contour[1].left));

    for (const auto& arc : contour)
      GEODE_ASSERT(arc.radius>0,"Radii must be positive so that symbolic perturbation doesn't make them negative");
  }

  // Prepare for interval arithmetic
  IntervalScope scope;

  // Prune arcs which don't intersect with horizontal lines
  auto info = prune_small_contours(unpruned);
  Arcs arcs = info.arcs.flat;
  Next next = info.next;
  Vertices vertices = info.vertices;

  // Compute all nontrivial intersections between segments
  struct Intersections {
    const BoxTree<EV2>& tree;
    Next next;
    Arcs arcs;
    Vertices vertices;
    Array<Vertex> found;

    Intersections(const BoxTree<EV2>& tree, Next next, Arcs arcs, Vertices vertices)
      : tree(tree), next(next), arcs(arcs), vertices(vertices) {}

    bool cull(const int n) const { return false; }
    bool cull(const int n0, const int box1) const { return false; }
    void leaf(const int n) const { assert(tree.prims(n).size()==1); }

    void leaf(const int n0, const int n1) {
      if (n0 != n1) {
        assert(tree.prims(n0).size()==1 && tree.prims(n1).size()==1);
        const int i1 = tree.prims(n0)[0], i2 = next[i1],
                  j1 = tree.prims(n1)[0], j2 = next[j1];
        if (   !(i2==j1 && j2==i1) // If we're looking at the two arcs of a length two contour, there's nothing to do
            && (   i1==j2 || i2==j1
                || (!arcs_from_same_circle(arcs,i1,j1) && circles_intersect(arcs,i1,j1)))) { // Ignore intersections of arc and itself
          // We can get here even if the two arcs are adjacent, since we may need to detect the other intersection of two adjacent circles.
          const auto a01 = vertices[i1],
                     a12 = vertices[i2],
                     b01 = vertices[j1],
                     b12 = vertices[j2];
          for (const auto ab : circle_circle_intersections(arcs,i1,j1))
            if (   ab!=a01.reverse() && ab!=a12 && ab!=b01 && ab!=b12.reverse()
                && circle_arcs_intersect(arcs,a01,a12,b01,b12,ab))
              found.append(ab);
        }
      }
    }
  };
  const auto tree = new_<BoxTree<EV2>>(info.boxes,1);
  Intersections pairs(tree,next,arcs,vertices);
  double_traverse(*tree,pairs);
  info.boxes.clean_memory();

  // Group intersections by segment.  Each pair is added twice: once for each order.
  Array<int> counts(arcs.size());
  for (auto v : pairs.found) {
    counts[v.i0]++;
    counts[v.i1]++;
  }
  Nested<Vertex> others(counts,false); // Invariant: if v in others[i], v.i0 = i.  This implies some wasted space, unfortunately.
  for (auto v : pairs.found) {
    others(v.i0,--counts[v.i0]) = v;
    others(v.i1,--counts[v.i1]) = v.reverse();
  }
  pairs.found.clean_memory();
  counts.clean_memory();

  // Walk all original polygons, recording which subsegments occur in the final result
  Hashtable<Vertex,Vertex> graph; // If u -> v, the output contains the portion of segment j from ij_a to jk_b
  for (const int p : range(info.arcs.size())) {
    const auto poly = info.arcs.range(p);

    const bool is_full_circle = (poly.size() == 1); // Check if this poly is a full circle (which won't have meaningful vertices)
    assert(is_full_circle == (next[poly.front()] == poly.front()));

    // Sort intersections along each segment
    for (const int i : poly) {
      const auto other = others[i];
      // Sort intersections along this segment
      // For a full circle starting vertex is arbitrary so even with two points we won't need to sort
      if (other.size() > 1 + is_full_circle) {
        struct PairOrder {
          Arcs arcs;
          const Vertex start; // The start of the segment

          PairOrder(Arcs arcs, Vertex start)
            : arcs(arcs)
            , start(start) {}

          bool operator()(const Vertex b0, const Vertex b1) const {
            assert(start.i1==b0.i0 && b0.i0==b1.i0);
            if (b0.i1==b1.i1 && b0.left==b1.left)
              return false;
            return circle_arc_intersects_circle(arcs,start,b1,b0);
          }
        };
        const Vertex sort_start = is_full_circle ? other[0].reverse() : vertices[i]; // For a full circle use an arbitrary vertex for the start
        sort(other.slice(is_full_circle,other.size()),PairOrder(arcs,sort_start));
      }
    }

    // Find which subarc the horizontal line intersects to determine the start point for walking
    const auto horizontal = info.horizontals[p];
    const int start = horizontal.arc;
    const auto seed_arc_others = others[start];
    // For a full circle, vertices[start] is just a placeholder so we prefer to use a 'real' intersections from others[start] instead
    const bool skip_placeholder = (is_full_circle && !seed_arc_others.empty());
    const Vertex seed_search_start = skip_placeholder ? seed_arc_others.front().reverse() : vertices[start];
    int substart = skip_placeholder; // Search from 1 if we are using the first intersection as our seed, otherwise start from 0
    for (;substart<seed_arc_others.size();substart++) { // Grow subsegment until we find the horizontal intersection
      if (circle_arc_contains_horizontal_intersection(arcs,seed_search_start,seed_arc_others[substart],horizontal))
        break;
    }

    // Compute the depth immediately outside our start point by firing a ray along the positive x axis.
    struct Depth {
      const BoxTree<EV2>& tree;
      Next next;
      Arcs arcs;
      Vertices vertices;
      const HorizontalVertex start;
      const Quantized start_xmin;
      int depth;

      Depth(const BoxTree<EV2>& tree, Next next, Arcs arcs, Vertices vertices, const HorizontalVertex start)
        : tree(tree), next(next), arcs(arcs), vertices(vertices)
        , start(start)
        , start_xmin(ceil(start.x.box().min)) // Safe to round up since we'll be comparing against conservative integer boxes
        // If we intersect no other arcs, the depth depends on the orientation of direction = (1,0) relative to the starting arc
        , depth(local_horizontal_depth(arcs,start)) {}

      bool cull(const int n) const {
        const auto box = tree.boxes(n);
        return box.max.x<start_xmin || box.max.y<start.y || box.min.y>start.y;
      }

      void leaf(const int n) {
        assert(tree.prims(n).size()==1);
        const int j = tree.prims(n)[0];
        if (circle_intersects_horizontal(arcs,j,start.y)) {
          for (const auto h : circle_horizontal_intersections(arcs,j,start.y)) {
            // If start.left == h.left and start.arc and h.arc are from the same circle, start and h refer to the same symbolic point even though start!=h
            // This will trigger an "identically zero predicate" error when calling horizontal_intersections_rightwards
            // Checking circle_arc_contains_horizontal_intersection before horizontal_intersections_rightwards avoids this unless the arcs overlap
            // Symbolically identical arcs that overlap aren't intended to be handled by splitting and can raise an assert here
            if (   start!=h
                && circle_arc_contains_horizontal_intersection(arcs,vertices[j],vertices[next[j]],h)
                && horizontal_intersections_rightwards(arcs,start,h))
              depth -= horizontal_depth_change(arcs,h);
          }
        }
      }
    };
    Depth ray(tree,next,arcs,vertices,horizontal);
    single_traverse(*tree,ray);
    // Walk around the contour, recording all subarcs at the desired depth
    int delta = ray.depth-depth;
    const auto seed_vertex = substart ? others(start,substart-1).reverse() : vertices[start];
    auto prev = seed_vertex;
    int index = start,
        sub = substart;
    do {
      if (sub==others.size(index)) { // At end of subsegments, step between the input contour arcs
        index++;
        if (index==poly.hi)
          index = poly.lo;
        sub = 0;
        const auto next = vertices[index];
        if(!skip_placeholder) {
          if (!delta) // Remember this subsegment if it has the right depth
            graph.set(prev,next);
          prev = next; // Now look at the segment starting from end of the current one
          continue; // Check if we returned to the seed and keep going
        }
        else {
          // If we have other intersections, we ignore any placeholder vertices and use the subsegments
          assert(others.size(index) >= 1); // To do this we must have at least one intersection
          assert(next != seed_vertex); // Make sure we don't try to skip over our start
          // We won't have updated prev at this point so it is important that we fall through to subsegment handling without checking loop condition
        }
      }
      // If we didn't hit the continue above we need to advance across an arc intersection to a new subsegment
      sub++;
      const auto next = others(index,sub-1);
      if (!delta) // Remember this subsegment if it has the right depth
        graph.set(prev,next);
      delta += next.left ^ arcs[next.i0].positive ^ arcs[next.i1].positive ? -1 : 1; // Update depth
      prev = next.reverse(); // Use this as our new start
    } while (prev != seed_vertex);
  }

  // Walk the graph to produce output polygons
  Hashtable<Vertex> seen;
  Nested<ExactCircleArc,false> output;
  for (const auto& it : graph) {
    const auto start = it.key();
    if (seen.set(start)) {
      auto v = start;
      for (;;) {
        auto a = arcs[v.i0];
        a.left = v.left;
        output.flat.append(a);
        v = graph.get(v);
        if (v == start)
          break;
        seen.set(v);
      }
      output.offsets.append(output.flat.size());
    }
  }
  return output;
}

Box<Vector<real,2>> approximate_bounding_box(const RawArray<const CircleArc> input) {
  Box<Vector<real,2>> result;
  for (int j=0,i=input.size()-1;j<input.size();i=j++) {
    result.enlarge(bounding_box(input[i].x,input[j].x).thickened(.5*abs(input[i].q)*magnitude(input[i].x-input[j].x)));
  }
  return result;
}

// Compute an approximate bounding box for all arcs
Box<Vector<real,2>> approximate_bounding_box(const Nested<const CircleArc>& input) {
  Box<Vector<real,2>> result;
  for (const auto poly : input) {
    result.enlarge(approximate_bounding_box(poly));
  }
  return result;
}

// Tweak quantized circles so that they intersect.
static bool tweak_arcs_to_intersect(RawArray<ExactCircleArc> arcs, const int i, const int j) {

  // TODO: For now, we require nonequal centers
  GEODE_ASSERT(arcs[i].center != arcs[j].center);

  bool changed = false;

  // We want dc = magnitude(Vector<double,2>(arcs[i].center-arcs[j].center))
  // But we need to use interval arithmetic to ensure the correct result
  Vector<double,2> delta = (arcs[i].center-arcs[j].center);
  const Interval dc_interval = assume_safe_sqrt(sqr(Interval(delta.x))+sqr(Interval(delta.y)));
  Quantized &ri = arcs[i].radius,
            &rj = arcs[j].radius;

  // Conservatively check if circles might be too far apart to intersect (i.e. ri+rj <= dc)
  if(!certainly_less(dc_interval,Interval(ri)+Interval(rj))) {
    const auto d_interval = (dc_interval-Interval(ri)-Interval(rj))*Interval(.5);
    const auto d = Quantized(floor(d_interval.box().max + 1)); // Quantize up
    ri += d;
    rj += d;
    changed = true;
  }
  // Conservatively check if inner circle is too small to intersect (i.e. abs(ri-rj) >= dc)
  if(!certainly_less(Interval(abs(ri-rj)),dc_interval)) {
    Quantized& small_r = ri<rj?ri:rj; // We will grow the smaller radius
    small_r = max(ri,rj)-Quantized(ceil(dc_interval.box().min-1));
    changed = true;
  }

  return changed;
}

// Tweak quantized circles so that they intersect.
void tweak_arcs_to_intersect(RawArray<ExactCircleArc> arcs) {
  IntervalScope scope;
  const int n = arcs.size();

  // Iteratively enlarge radii until we're nondegenerately intersecting.  TODO: Currently, this is worst case O(n^2).
  for(;;) {
    bool done = true;
    for (int j=0,i=n-1;j<n;i=j++) {
      done &= !tweak_arcs_to_intersect(arcs, i, j);
    }
    if(done) break;
  }
}

void tweak_arcs_to_intersect(Nested<ExactCircleArc>& arcs) {
  for (const auto contour : arcs) {
    tweak_arcs_to_intersect(contour);
  }
}

template<int a> GEODE_PURE static inline Exact<a> ceil_half(const Exact<a> x) {
  assert(!is_negative(x));
  Exact<a> r = x;
  ++r; // Add 1 to round up
  r >>= 1;
  return r;
}

// Return integers num,denom both with magnitude less than or equal to exact::bound such that num/denom approximately equals x.
// If abs(x) is very large result may be clamped to +/- exact::bound
static Vector<Exact<1>,2> rational_approximation(const real x) {
  assert(!isnan(x));

  // x == x/1 == x*denom / denom == num / denom
  // We want to choose denom as big as possible such that abs(x)*denom <= exact::bound
  const real scale = round(exact::bound / max(1.,ceil(abs(x))));
  assert(0 <= scale && scale <= exact::bound); // scale is exact::bound divided by some value >= 1, so must be <= exact::bound
  const ExactInt denom = ExactInt(scale);
  const real approx_num = round(x*scale);
  // Since scale*ceil(abs(x)) <= exact::bound we know scale*abs(x) <= exact::bound
  // Thus abs(approx_num) will be <= exact::bound
  assert(abs(approx_num) <= exact::bound);
  if(denom > 0) {
    assert(sign(approx_num) == sign(x));
    return vec(Exact<1>(approx_num), Exact<1>(denom));
  }
  else {
    // If abs(x) is on the order of 0.5*exact::bound or larger, denom could have been rounded to zero
    // Compute a numerator in the valid range and use 1 for the denom
    const ExactInt safe_num = ExactInt(clamp(round(x), real(-exact::bound), real(exact::bound)));
    assert(sign(safe_num) == sign(x));
    return vec(Exact<1>(safe_num), Exact<1>(1));
  }
}

#if CHECK_CONSTRUCTIONS
static bool test_circles_intersect(const Vector<Quantized,2> c0, const Quantized r0, const Vector<Quantized,2> c1, const Quantized r1) {
  const Vector<ExactCircleArc, 2> test_arcs(ExactCircleArc(c0, r0, 0, true, true), ExactCircleArc(c1, r1, 1, true, true));
  return circles_intersect(asarray(test_arcs),0,1);
}
#endif

// A circle that is centered at x0 or x1 with a radius of constructed_arc_max_endpoint_error() will always intersect
// an ExactCircleArc with radius and center as returned from construct_circle_radius_and_center(x0,x1,q)
Quantized constructed_arc_endpoint_error_bound() { return 2; } // ceil(sqrt(2)/2)

// Returns a center and radius for a circle that passes within constructed_arc_endpoint_error_bound() units of each quantized vertex and has approxamently the correct curvature
// x0 and x1 should be integer points (from quantizer)
// WARNING: If endpoints have been quantized to the same point, a radius of 0 (invalid for an ExactCircleArc) will be returned to indicate no arc is needed
// As long as x0 != x1, a circle of radius constructed_arc_endpoint_error_bound() centered at x0 or x1 will always intersect the returned circle
Tuple<Vector<Quantized,2>, Quantized> construct_circle_center_and_radius(const Vector<Quantized, 2> x0, const Vector<Quantized, 2> x1, const real q) {
  if(x0 == x1) {
    return tuple(x0, Quantized(0));
  }

  const Vector<Exact<1>,2> ex0(x0),
                           ex1(x1);

  const Vector<Exact<1>,2> delta = ex1 - ex0;

  const Exact<1> max_r = Exact<1>(exact::bound/8); // FIXME: I'm not convinced this is correct, but it will work for now

  const Vector<Exact<1>,2> q_fract = rational_approximation(q);

  const SmallShift exact_two = (One()<<1);
  const Vector<Exact<1>,2> d_perp = delta.orthogonal_vector();
  const Vector<Exact<3>,2> H_num = emul(sqr(q_fract.y) - sqr(q_fract.x), d_perp);
  const Exact<2> H_den_times_half = q_fract.y * q_fract.x * exact_two;
  const Exact<2> H_den = H_den_times_half * exact_two;
  const Exact<3> H_num_l1 = H_num.L1_Norm();

  // Compare length of H (using l-one norm for simplicity) and max_r to ensure we aren't going to generate a center that is out of bounds
  const bool must_scale_H = (H_num_l1 >= max_r * abs(H_den));

  Vector<Quantized,2> center = must_scale_H
    // If H is too big, we need to scale it so that biggest component is <= max_r
    // We need: (H_num_l_inf / abs(H_den)) * s = max_r
    // Solve for s: s = (max_r * abs(H_den)) / H_num_l1
    // Plug in s: H * s = (H_num / H_den) * ((max_r * abs(H_den)) / H_num_l1)
    // H_den cancels except for sign and we have: H_prime = ((H_den < 0 : -1 : 1) * max_r * H_num) / H_num_l1
    // C = (0.5 * (x0 + x1)) + H_prime
    ? snap_div(emul((is_negative(H_den) ? -max_r : max_r)*exact_two, H_num) + emul(H_num_l1, ex0 + ex1), H_num_l1*exact_two, false)
    // Otherwise we do:
    // C = (0.5 * (x0 + x1)) + H
    // Combine into a single fraction and snap
    : snap_div(H_num + emul(H_den_times_half, ex0 + ex1), H_den, false);

  assert(is_nonzero(H_den) || must_scale_H);

  // Since C is exact up to rounding, we know that distance to endpoints will differ by at most sqrt(2)
  // Instead of multiple calls to snap_div, we just take average of squared distances to each endpoint
  // This value will be at or between the two options and is symmetric with respect to x0 and x1
  auto ecenter = Vector<Exact<1>,2>(center);
  const Exact<2> sum_dist_sqr = esqr_magnitude(ecenter - ex0) + esqr_magnitude(ecenter - ex1);
  Quantized r = snap_div(sum_dist_sqr, Exact<1>(2), true);

  // We need a circle centered at x0 or x1 with radius == constructed_arc_endpoint_error_bound() to intersect the constructed circle
  // As computed above, center and r will ensure this happens except when r is too small and the constructed arc is fully inside the endpoint error circles
  if(r <= constructed_arc_endpoint_error_bound()) {
    // For an endpoint outside the circle, it is safe to increase r since that will decrease error to that endpoint
    // We only grow radius up to the error bound so it is impossible to overshoot by too much
    // Any endpoints that start inside the circle will be ok unless they are exactly at the center of the constructed arc
    if(ecenter == ex0 || ecenter == ex1) {
      // Center point was exactly on the line of points equidistant to x0 and x1
      // Rounding it moved it by at most sqrt(0.5) so distance from x0 to x1 is <= 2*sqrt(0.5) == sqrt(2)
      // We move center by the orthogonal delta between x0 and x1 to ensure we don't move it on top of the other endpoint
      center += ((q >= 0) ? (x1 - x0) : (x0 - x1)).orthogonal_vector();
      assert(sqrt(2) < constructed_arc_endpoint_error_bound());
      // Distance from center to endpoint that center was on top of will be equal to length of orthogonal vector which is >= 1 and <= sqrt(2)
      // Distance from center to other endpoint will be sqrt(2) * length of orthogonal vector <= sqrt(2)*sqrt(2) == 2
      // Since we used the orthogonal vector, we won't have moved center on top of the other endpoint
    }
    r = constructed_arc_endpoint_error_bound();
  }


#if CHECK_CONSTRUCTIONS
  const auto old_center = ecenter;
  ecenter = Vector<Exact<1>,2>(center);
  const auto est_mid = 0.5*((x0 + x1) + q*rotate_right_90(x1 - x0));

  GEODE_ASSERT(test_circles_intersect(center, r, x0, constructed_arc_endpoint_error_bound()));
  GEODE_ASSERT(test_circles_intersect(center, r, x1, constructed_arc_endpoint_error_bound()));

  // Limits on maximum radius can result in larger errors for midpoint of straight (or almost so) arcs (when q is close to zero)
  // If we were forced to scale H we need to allow a larger error margin to account for added curvature.
  // I haven't been able to work out a guaranteed error bound at the midpoint, but found no errors larger than 6 units during testing.
  GEODE_ASSERT(test_circles_intersect(center, r, vec(round(est_mid.x),round(est_mid.y)), constructed_arc_endpoint_error_bound()*(must_scale_H ? 3 : 1)));

#endif

  return tuple(center, r);
}

#if CHECK_CONSTRUCTIONS
static Tuple<real, Vec2> arc_radius_and_center(const Vec2 x0, const Vec2 x1, const real q) {
  const auto dx = x1-x0;
  const auto L = magnitude(dx);
  // Compute radius, quantize, then compute center from quantized radius to reduce endpoint error
  const auto radius = .25*L*abs(q+1/q);
  const auto center = L ? .5*(x0+x1)+((q>0)^(abs(q)>1)?1:-1)*sqrt(max(0.,sqr(radius/L)-.25))*rotate_left_90(dx) : x0;

  return tuple(radius, center);
}
#endif

// Starting at v0 and walking around arcs[on_circle] in arcs[on_circle].positive, is v0 reached before v1?
// This function is used to sort intersections in unravel_helper_arc and could probably be replaced with direct calls to circle_arc_intersects_circle with some very careful case analysis
static bool vertices_in_order(const Arcs arcs, const int on_circle, const Vertex& v0, const Vertex& v1, const Vertex& v2) {
  const Vertex a01 = (v0.i1 == on_circle) ? v0 : v0.reverse();
  const Vertex a12 = (v1.i0 == on_circle) ? v1 : v1.reverse();
  const Vertex a1b = (v2.i0 == on_circle) ? v2 : v2.reverse();

  assert(a01.i1 == on_circle);
  assert(a01.i1==a12.i0 && a12.i0==a1b.i0);

  return !circle_arc_intersects_circle(arcs, a01, a12, a1b);
}

// Choose directions and left flags for intersections with c1 to try and avoid any crossings
// For use on helper arcs created when c0 and c2 don't intersect
static void unravel_helper_arc(const RawArray<ExactCircleArc> arcs, const int c0, const int c1, const int c2) {
  #if CHECK_CONSTRUCTIONS
    GEODE_ASSERT(circles_intersect(arcs, c0, c1));
    GEODE_ASSERT(circles_intersect(arcs, c1, c2));
  #endif
  enum IID { R01=0, L01=1, R12=2, L12=3 }; // Intersection ID
  const auto verts = Vector<Vertex,4>(circle_circle_intersections(arcs, c0, c1),
                                      circle_circle_intersections(arcs, c1, c2));
  // There are 4 intersections with c1 and its neighbors
  auto arc_order = Vector<IID,4>(R01,L01,R12,L12);

  // These swaps sort arc_order (one predicate could be eliminated by using the fact that arc_order[2] can't be L01 after sorting)
  if(!vertices_in_order(arcs, c1, verts[R01], verts[arc_order[1]], verts[arc_order[2]]))
    swap(arc_order[1],arc_order[2]);
  if(!vertices_in_order(arcs, c1, verts[R01], verts[arc_order[2]], verts[arc_order[3]]))
    swap(arc_order[2],arc_order[3]);
  if(!vertices_in_order(arcs, c1, verts[R01], verts[arc_order[1]], verts[arc_order[2]]))
    swap(arc_order[1],arc_order[2]);

  assert(arc_order[0] == R01); // Nothing should have changed our reference arc
  if(!(arc_order[2] != L01)) {
    // GEODE_WARNING("Intersection detected inside helper arc");
    return; // For now leave these as is
  }
  //assert(arc_order[2] != L01); // arc_order[2] should only be L01 if c0 and c2 intersect inside c1 (which means we shouldn't have created a helper arc)

  // We want to arc from R01 in whichever direction doesn't cross L01
  // Since R01 and L01 are adjacent we have either [R01, L01, X12, Y12] or [R01, X12, Y12, L01] (with X and Y as L/R or R/L)
  // R01 will arc to whichever one of arc_order[1] or arc_order[3] isn't L01
  const IID R01_connection = (arc_order[1] != L01) ? arc_order[1] : arc_order[3];
  // In both cases L01 arcs to the intersection farthest (going forward or backwards) from R01
  const IID L01_connection = arc_order[2];

  // If R01's neighbor is in same direction as circle then next entry in arc order won't be L01
  // If not, we will need to reverse this arc
  const bool R01_pair_correctly_oriented = arc_order[1] != L01;

  // We now need to choose between R01's pair and L01's pair. Since the choice of connections ensures we don't introduce unneccesary topological features it might be
  // better to keep existing value of arcs[c0].left. For now we use smallest approximate distances.
  const auto R01_pair_dist_sqr = (verts[R01_connection].rounded - verts[R01].rounded).sqr_magnitude();
  const auto L01_pair_dist_sqr = (verts[L01_connection].rounded - verts[L01].rounded).sqr_magnitude();
  const bool left_01 = L01_pair_dist_sqr < R01_pair_dist_sqr; // Choose based on smaller distance

  const bool left_12 = (left_01 ? L01_connection : R01_connection) & 1; // Use low bit of enum to get left flag

  const bool c1_correctly_oriented = left_01
    ? !R01_pair_correctly_oriented // L01's pair needs opposite orientation from R01
    : R01_pair_correctly_oriented;

  // Shouldn't be able to have two helper arcs in a row so setting these flags won't conflict with previous calls to unravel_helper_arc
  arcs[c0].left = left_01;
  arcs[c1].left = left_12;

  // Set orientation of c1 so that we get the simpler arc
  if(!c1_correctly_oriented) {
    arcs[c1].positive = !arcs[c1].positive;
  }
}

// Compute q values for a quantized arc accounting for perturbations
// Returns both q value for the arc and for the opposite arc on the same circle
// One of the returned values will have magnitude <= 1 and the other will have magnitude >= 1
// opp_q should be -1/q, but this function will correctly handle sign if q is 0
static Vec2 arc_q_and_opp_q(const Arcs& arcs, const Vertex& v01, const Vertex& v02) {
  assert(v01.i0 == v02.i0);
  const auto& arc = arcs[v01.i0];
  const auto r = arc.radius;
  const auto l = min(r, 0.5 * (v01.rounded - v02.rounded).magnitude());
  const auto root = sqrt(max(0.,sqr(r) - sqr(l)));

  const int d0 = arc.positive ? 1 : -1;
  const int d1 = circle_intersections_ccw(arcs, v01, v02) ? 1 : -1;
  const bool is_small_arc = (d0*d1 > 0);

  // If l is 0 or very small both q and opposite_q will be 0. We would like to get 0 and +/-inf.
  // q = (d0*l / (r + d0*d1*root)) or q = ((r - d0*d1*root) / ( d0*l)) are algebraically equivalent, but the latter is numerically unstable as l approaches 0
  // We factor out signs and get expression for the smaller and larger q values
  const real abs_q_short = l / (r + root); // ...use this to ensure q will go to zero as l goes to zero
  const real abs_q_long = (r + root) / l; // i.e. 1./abs_q_short

  const real q =      d0 * (is_small_arc ? abs_q_short : abs_q_long);
  const real opp_q = -d0 * (is_small_arc ? abs_q_long : abs_q_short);

  assert(max(abs(q), abs(opp_q)) >= 1 && min(abs(q),abs(opp_q)) <= 1);

  return Vec2(q, opp_q);
}

// This function must have GEODE_NEVER_INLINE to ensure Clang doesn't move IntervalScope initialization above non-interval arithmetic
GEODE_NEVER_INLINE Tuple<Quantizer<real,2>,Nested<ExactCircleArc>> quantize_circle_arcs_scoped(const Nested<const CircleArc> input, const Quantizer<Quantized, 2> quant) {
  IntervalScope scope;
  const Quantized allowed_vertex_error = constructed_arc_endpoint_error_bound() + Vertex::tolerance() + 1;
  const Quantized allowed_vertex_error_sqr = sqr(allowed_vertex_error);

  // Quantize and implicitize each arc
  auto output = Nested<ExactCircleArc,false>();
  auto new_contour = Array<ExactCircleArc>();

  const int num_in_arcs = input.total_size();
  int next_helper_index = num_in_arcs; // permution index used for helper arcs
  #if CHECK_CONSTRUCTIONS
  struct SourceData {
    bool is_helper;
    Vec2 x0;
    real q;
    Vec2 x1;
  };
  Hashtable<int, SourceData> source_data;
  #endif

  for (const int p : range(input.size())) {
    const int base = input.offsets[p];
    const auto in = input[p];
    new_contour.clear();
    const int n = in.size();
    for(int i = 0; i<n; ++i) {
      const int j = (i+1)%n;
      const auto arc_start = quant(in[i].x),
                 arc_end = quant(in[j].x); // Quantize the start and end points of the arc
      const auto arc_q = in[i].q; // q is dimensionless and can be used without scaling

      if(arc_start == arc_end)
        continue; // Ignore any 0 size arcs

      const auto center_and_radius = construct_circle_center_and_radius(arc_start, arc_end, arc_q);
      ExactCircleArc e;
      e.center = center_and_radius.x;
      e.radius = center_and_radius.y;
      e.index = base+i;
      e.positive = arc_q >= 0;
      e.left = false; // Set to false to avoid compiler warning. Actual value set below
      assert(&(input.flat[e.index]) == &(input(p,i))); // We assume e.index can be used to lookup origional q values
      assert(e.radius > 0); // We should have a valid radius as long as arc_start != arc_end
      #if CHECK_CONSTRUCTIONS
      GEODE_ASSERT(!source_data.contains(e.index));
      source_data[e.index] = SourceData({false, in[i].x, arc_q, in[j].x});
      #endif

      new_contour.append(e);
    }

    const int out_n = new_contour.size();

    // Fill in left flags
    for (int j=0,i=out_n-1;j<out_n;i=j++) {
      const int source_index = (new_contour[i].index - base + 1) % n;
      const auto x = in[source_index].x,
                 c0 = quant.inverse(new_contour[i].center),
                 c1 = quant.inverse(new_contour[j].center);
      new_contour[i].left = cross(c1-c0,x-c0)>0;
    }

    // Check in case quantization created a single repeated point
    if(new_contour.size() == 2 && (new_contour[0].left != new_contour[1].left)) {
      continue; // Discard it if so
    }
    else { // Assuming we want to keep the contour...
      output.append_empty(); // Allocate new subarray in the output
      for (int i=0;i<out_n;++i) {
        const int j = (i+1)%out_n;
        output.append_to_back(new_contour[i]); // Add this arc
        const int source_index = (new_contour[i].index - base + 1) % n;
        const auto target_position = quant(in[source_index].x); // This is quantized value for position of arc

        if(circles_intersect(new_contour, i, j)) { // Check if we already have an intersection
          const Vertex vert = circle_circle_intersections(new_contour, i, j)[new_contour[i].left];
          const auto vert_error_sqr = (vert.rounded - target_position).sqr_magnitude();
          if(vert_error_sqr <= allowed_vertex_error_sqr)
            continue; // If error between intersection and target is small we don't need to do anything
        }

        // If we fall through we either don't have an intersection or the intersection point is too far from intended position
        // In either case we need to add a new helper arc that will have vertices close to the intersection
        ExactCircleArc new_e;
        new_e.center = target_position;
        new_e.radius = constructed_arc_endpoint_error_bound();
        new_e.positive = true;
        new_e.left = true;
        new_e.index = next_helper_index++;

        #if CHECK_CONSTRUCTIONS
          GEODE_ASSERT(!source_data.contains(new_e.index));
          source_data[new_e.index] = SourceData({true, quant.inverse(target_position), 1, quant.inverse(target_position)});
        #endif

        output.append_to_back(new_e);

        #if CHECK_CONSTRUCTIONS
          const int out_i = output.back().size()-2;
          GEODE_ASSERT(circles_intersect(output.back(),out_i, out_i + 1));
          const auto new_verts = circle_circle_intersections(output.back(), out_i, out_i + 1);
          const auto new_dists_sqr = vec((new_verts[0].rounded - target_position).sqr_magnitude(),(new_verts[1].rounded - target_position).sqr_magnitude());
          GEODE_ASSERT(new_dists_sqr.min() <= allowed_vertex_error_sqr);
        #endif
      }
    }
  }

  // Fix directions and left flags for any helper arcs we added
  const Array<const int> next = closed_contours_next(output);
  const RawArray<ExactCircleArc> arcs = output.flat;
  for(const int i0 : range(next.size())) {
    const int i1 = next[i0];
    if(arcs[i1].index >= num_in_arcs) { // Any helper arc will have index >= number of original arcs
      const int i2 = next[i1];
      unravel_helper_arc(arcs,i0,i1,i2);
    }
  }

  // Make sure we didn't turn any small short lines int giant arcs
  const auto verts = compute_vertices(arcs, next);
  for(const int i0 : range(next.size())) {
    auto& arc = arcs[i0];
    if(arc.index >= num_in_arcs)
      continue; // skip over helper arcs

    // arc.positive selects one of the two arcs between v10 and v12. However, if two input vertices are close together
    // their relative angle around the circle may appear flipped after constructions. This can cause us to grab the opposite
    // arc turning very small angles into very large ones or vice versa. We attempt to compensate for that here.
    // Cases where we have trouble choosing the correct arc should only arise when endpoints are within our quantization
    // error of each other. If we had quantized both endpoints to the same point we wouldn't have violated error bounds
    // therefore we assume any short arc between them should be acceptable.
    const int i1 = next[i0];
    const Vertex v10 = verts[i0].reverse();
    const Vertex v12 = verts[i1];
    const real in_q = input.flat[arc.index].q;

    // Since we can only switch arc.positive without affecting adjacent arcs we check which result gets closest to the input and assume
    // one of the two will be good enough. We could instead compute a padded safety margin and always use the smaller arcs if vertices
    // are close together, but this would increase worst case quantization errors (small arc just above quantization limits with q = 1.001
    // could be swapped to q = -0.999 and midpoint would be moved across the circle).

    // We rely on arc_q_and_opp_q to correctly handle degeneracies by checking the appropriate exact predicates and returning 0 or +/-inf
    const Vec2 q_and_opp_q = arc_q_and_opp_q(arcs, v10, v12);
    const real& norm_q = q_and_opp_q[0];
    const real& opp_q = q_and_opp_q[1];
    // If vertices are degenerate, one of opp_q or norm_q should be infinite, and we will always choose the small arc
    assert(isfinite(in_q)); // in_q should be inf or nan and would probably break other things first
    if(abs(opp_q - in_q) < abs(norm_q - in_q)) {
      // If the opposite arc's q value is closer to the input q value we assume we have degenerate input
      arc.positive = !arc.positive;

      #if CHECK_CONSTRUCTIONS
      {
        // If we swapped endpoints they should have overlapping uncertanty
        const auto& src = source_data[arc.index];
        const auto x0 = Box<exact::Vec2>(quant(src.x0)).thickened(0.5 + constructed_arc_endpoint_error_bound() + Vertex::tolerance());
        const auto x1 = Box<exact::Vec2>(quant(src.x1)).thickened(0.5 + constructed_arc_endpoint_error_bound() + Vertex::tolerance());
        GEODE_ASSERT(x0.intersects(x1));
        GEODE_ASSERT(abs(in_q) < 2.); // Not an error in quantization, but if caller is creating degenerate arcs with large q values we should warn them.
      }
      #endif
    }
  }

  #if CHECK_CONSTRUCTIONS
  {
    const Array<int> prev(next.size());
    for(int i : range(next.size()))
      prev[next[i]] = i;

    const auto result = output.freeze();
    const auto verts = compute_vertices(arcs, next);
    for(const int i0 : range(next.size())) {
      const auto& arc = arcs[i0];
      GEODE_ASSERT(source_data.contains(arc.index));
      SourceData data = source_data[arc.index];

      const int i1 = next[i0];
      const Vertex v0 = verts[i0].reverse();
      const Vertex v1 = verts[i1];
      GEODE_ASSERT(v0.i0 == i0);
      GEODE_ASSERT(v1.i0 == i0);
      GEODE_ASSERT(v0.i0==v1.i0);

      const real error_margin = 2.*allowed_vertex_error*quant.inverse.inv_scale;
      const real v0_error = magnitude(quant.inverse(v0.rounded) - data.x0);
      const real v1_error = magnitude(quant.inverse(v1.rounded) - data.x1);

      // FIXME: Can we work out a strict error bound for error of radius and center?
      if(false) {
        const auto expected_radius_and_center = arc_radius_and_center(data.x0,data.x1,data.q);
        const auto actual_radius_and_center = tuple(quant.inverse.inv_scale * arc.radius, quant.inverse(arc.center));

        const real radius_error = abs(expected_radius_and_center.x - actual_radius_and_center.x);
        const real center_error = magnitude(expected_radius_and_center.y - actual_radius_and_center.y);
        if(radius_error > error_margin || center_error > error_margin) {
          GEODE_WARNING(format("Center and radius errors larger than expected: center_error: %f, radius_error: %f", center_error, radius_error));
        }
      }

      if(v0_error > error_margin || v1_error > error_margin) {
        GEODE_FATAL_ERROR("Botched construction!");
      }
    }
  }
  #endif
  return tuple(quant,output.freeze());
}

Quantizer<real,2> make_arc_quantizer(const Box<Vector<real,2>> arc_bounds) {
  // Enlarge box quite a lot so that we can closely approximate lines.
  // The error in approximating a straight segment of length L by a circular arc of radius R is
  //   er = L^2/(8*R)
  // If the maximum radius is R, the error due to quantization is
  //   eq = R/bound
  // Equating these, we get
  //   R/bound = L^2/(8*R)
  //   R^2 = L^2*bound/8
  //   R = L*sqrt(bound/8)
  const real max_radius = sqrt(exact::bound/8)*arc_bounds.sizes().max();
  return quantizer(arc_bounds.thickened(max_radius));
}

Tuple<Quantizer<real,2>,Nested<ExactCircleArc>> quantize_circle_arcs(const Nested<const CircleArc> input, const Box<Vector<real,2>> min_bounds) {
  Box<Vector<real,2>> box = min_bounds;
  box.enlarge(approximate_bounding_box(input));
  return quantize_circle_arcs_scoped(input, make_arc_quantizer(box));
}

#if 0
// This version doesn't cull small arcs

Nested<CircleArc> unquantize_circle_arcs(const Quantizer<real,2> quant, Nested<const ExactCircleArc> input) {
  IntervalScope scope;
  const auto output = Nested<CircleArc>::empty_like(input);
  for (const int p : range(input.size())) {
    const int base = input.offsets[p];
    const auto in = input[p];
    const auto out = output[p];
    const int n = in.size();
    for (int j=0,i=n-1;j<n;i=j++)
      out[j].x = quant.inverse(circle_circle_intersections(input.flat,base+i,base+j)[in[i].left].rounded);
    for (int j=0,i=n-1;j<n;i=j++) {
      const auto x0 = out[i].x,
                 x1 = out[j].x,
                 c = quant.inverse(in[i].center);
      const auto radius = quant.inverse.inv_scale*in[i].radius;
      const auto half_L = .5*magnitude(x1-x0);
      const int s = in[i].positive ^ (cross(x1-x0,c-x0)>0) ? -1 : 1;
      out[i].q = half_L/(radius+s*sqrt(max(0.,sqr(radius)-sqr(half_L)))) * (in[i].positive ? 1 : -1);
    }
  }
  return output;
}
#else
Nested<CircleArc> unquantize_circle_arcs(const Quantizer<real,2> quant, Nested<const ExactCircleArc> input) {
  IntervalScope scope;
  auto out = Array<CircleArc>();
  auto cull = Array<bool>();
  Nested<CircleArc,false> result;

  const auto next = closed_contours_next(input);
  const auto verts = compute_vertices(input.flat, next);

  for (const int p : range(input.size())) {
    const int base = input.offsets[p];
    const auto in = input[p];
    if(in.size() == 1) { // Catch any size one circle arcs and convert them into full circles
      const auto el = in[0].center - EV2(in[0].radius, 0);
      const auto er = in[0].center + EV2(in[0].radius, 0);
      result.append_empty();
      result.append_to_back(CircleArc(quant.inverse(er), in[0].positive ? 1 : -1));
      result.append_to_back(CircleArc(quant.inverse(el), in[0].positive ? 1 : -1));
      continue;
    }
    out.resize(in.size(),false,false);
    cull.resize(in.size(),false,false);
    const int n = in.size();
    bool culled_prev = false;
    int num_culled = 0;

    for (int i : range(in.size())) {
      out[i].x = quant.inverse(verts[base+i].rounded);
    }
    for (int j=0,i=n-1;j<n;i=j++) {
      // Since q is dimensionless we can compute it using the quantized data
      out[i].q = arc_q_and_opp_q(input.flat, verts[base+i].reverse(), verts[base+j]).x;

      if(culled_prev) { // Don't cull more than one arc in a row to avoid large errors for many small arcs (shouldn't have more than one helper arc in a row anyway)
        cull[i] = false;
        culled_prev = false;
      }
      else {
        cull[i] = in[i].radius <= constructed_arc_endpoint_error_bound();
        culled_prev = cull[i];
        num_culled += cull[i];
      }
    }

    if(out.size() - num_culled > 1) {
      result.append_empty();
      for(int i = 0; i < n; ++i) {
        if(!cull[i])
          result.append_to_back(out[i]);
      }
      // Quantization can introduce self intersections so we cull any miniscule slivers
      // This is not an ideal solution, but I don't know a better alternative
      if(result.back().size() <= 3) {
        // If the entire arc is thin, area will be less than thickness * perimeter / 2
        // We can use diagonal of bounding box as an estimate for perimeter / 2
        const real d = approximate_bounding_box(result.back()).sizes().magnitude();
        const real a = circle_arc_area(result.back());
        if(abs(a) < d * quant.inverse.unquantize_length(2*constructed_arc_endpoint_error_bound())) {
          result.pop_back();
        }
      }
    }
  }
  return result.freeze();
}
#endif

Nested<CircleArc> split_circle_arcs(Nested<const CircleArc> arcs, const int depth) {
  const auto e = quantize_circle_arcs(arcs);
  return unquantize_circle_arcs(e.x,exact_split_circle_arcs(e.y,depth));
}

ostream& operator<<(ostream& output, const CircleArc& arc) {
  return output << format("CircleArc([%g,%g],%g)",arc.x.x,arc.x.y,arc.q);
}

ostream& operator<<(ostream& output, const ExactCircleArc& arc) {
  return output << format("ExactCircleArc([%g,%g],%g,%c%c)",arc.center.x,arc.center.y,arc.radius,arc.positive?'+':'-',arc.left?'L':'R');
}

// The area between a segment of length 2 and an associated circular sector
static inline double q_factor(double q) {
  // Economized rational approximation courtesy of Mathematica.  I suppose this is a tiny binary blob?
  const double qq = q*q;
  return abs(q)<.25 ? q*(1.3804964920832707+qq*(1.018989299316004+0.14953934953934955*qq))/(1.035372369061972+qq*(0.5571675010595465+1./33*qq))
                    : .5*(atan(q)*sqr((1+qq)/q)-(1-qq)/q);
}

real circle_arc_area(RawArray<const CircleArc> arcs) {
  const int n = arcs.size();
  real area = 0;
  for (int i=n-1,j=0;j<n;i=j++)
    area += .5*cross(arcs[i].x,arcs[j].x) + .25*sqr_magnitude(arcs[j].x-arcs[i].x)*q_factor(arcs[i].q); // Triangle area plus circular sector area
  return .5*area;
}

real circle_arc_area(Nested<const CircleArc> polys) {
  real area = 0;
  for (const auto arcs : polys)
    area += circle_arc_area(arcs);
  return area;
}

void reverse_arcs(RawArray<CircleArc> arcs) {
  if(arcs.empty()) return;
  arcs.reverse();
  const auto temp_q = arcs.front().q;
  for(int i = 0,j = 1; j<arcs.size(); i=j++) {
    arcs[i].q = -arcs[j].q;
  }
  arcs.back().q = -temp_q;
}
void reverse_arcs(Nested<CircleArc> polyarcs) {
 for(auto poly : polyarcs) reverse_arcs(poly);
}

void exact_reverse_arcs(RawArray<ExactCircleArc> arcs) {
  if(arcs.empty()) return;
  arcs.reverse();

  const auto temp_left = arcs.front().left;
  for(int i = 0, j = 1; j<arcs.size(); i = j++) {
    arcs[i].positive = !arcs[i].positive;
    arcs[i].left = !arcs[j].left;
  }
  arcs.back().left = !temp_left;
  arcs.back().positive = !arcs.back().positive;
}
void exact_reverse_arcs(Nested<ExactCircleArc> polyarcs) {
 for(auto poly : polyarcs) exact_reverse_arcs(poly);
}

Nested<CircleArc> canonicalize_circle_arcs(Nested<const CircleArc> polys) {
  // Find the minimal point in each polygon under lexicographic order
  Array<int> mins(polys.size());
  for (int p=0;p<polys.size();p++) {
    const auto poly = polys[p];
    for (int i=1;i<poly.size();i++)
      if (lex_less(poly[i].x,poly[mins[p]].x))
        mins[p] = i;
  }

  // Sort the polygons
  struct Order {
    Nested<const CircleArc> polys;
    RawArray<const int> mins;
    Order(Nested<const CircleArc> polys, RawArray<const int> mins)
      : polys(polys), mins(mins) {}
    bool operator()(int i,int j) const {
      return lex_less(polys(i,mins[i]).x,polys(j,mins[j]).x);
    }
  };
  Array<int> order = arange(polys.size()).copy();
  sort(order,Order(polys,mins));

  // Copy into new array
  Nested<CircleArc> new_polys(polys.sizes().subset(order).copy(),false);
  for (int p=0;p<polys.size();p++) {
    const int base = mins[order[p]];
    const auto poly = polys[order[p]];
    const auto new_poly = new_polys[p];
    for (int i=0;i<poly.size();i++)
      new_poly[i] = poly[(i+base)%poly.size()];
  }
  return new_polys;
}

#ifdef GEODE_PYTHON

// Instantiate Python conversions for arrays of circular arcs
namespace {
template<> struct NumpyDescr<CircleArc>{static PyArray_Descr* d;static PyArray_Descr* descr(){GEODE_ASSERT(d);Py_INCREF(d);return d;}};
template<> struct NumpyIsStatic<CircleArc>:public mpl::true_{};
template<> struct NumpyRank<CircleArc>:public mpl::int_<0>{};
template<> struct NumpyArrayType<CircleArc>{static PyTypeObject* type(){return numpy_recarray_type();}};
PyArray_Descr* NumpyDescr<CircleArc>::d;
template<> struct NumpyDescr<ExactCircleArc>{static PyArray_Descr* d;static PyArray_Descr* descr(){GEODE_ASSERT(d);Py_INCREF(d);return d;}};
template<> struct NumpyIsStatic<ExactCircleArc>:public mpl::true_{};
template<> struct NumpyRank<ExactCircleArc>:public mpl::int_<0>{};
template<> struct NumpyArrayType<ExactCircleArc>{static PyTypeObject* type(){return numpy_recarray_type();}};
PyArray_Descr* NumpyDescr<ExactCircleArc>::d;
}
ARRAY_CONVERSIONS(1,CircleArc)
ARRAY_CONVERSIONS(1,ExactCircleArc)
NESTED_CONVERSIONS(CircleArc)
NESTED_CONVERSIONS(ExactCircleArc)

static void _set_circle_arc_dtypes(PyObject* inexact, PyObject* exact) {
  GEODE_ASSERT(PyArray_DescrCheck(inexact));
  GEODE_ASSERT(PyArray_DescrCheck(exact));
  GEODE_ASSERT(((PyArray_Descr*)inexact)->elsize==sizeof(CircleArc));
  GEODE_ASSERT(((PyArray_Descr*)  exact)->elsize==sizeof(ExactCircleArc));
  Py_INCREF(inexact);
  Py_INCREF(  exact);
  NumpyDescr<     CircleArc>::d = (PyArray_Descr*)inexact;
  NumpyDescr<ExactCircleArc>::d = (PyArray_Descr*)  exact;
}

static Nested<CircleArc> circle_arc_quantize_test(Nested<const CircleArc> arcs) {
  const auto e = quantize_circle_arcs(arcs);
  return unquantize_circle_arcs(e.x,e.y);
}

static Tuple<Nested<CircleArc>,Nested<CircleArc>,Nested<CircleArc>> single_circle_handling_test(int seed, int count) {
  auto rnd = new_<Random>(seed);
  Nested<ExactCircleArc, false> test_circles;
  const auto test_center_range = Box<Vec2>(Vec2(0,0)).thickened(100);
  const real max_test_r = 100.;
  const auto test_bounds = test_center_range.thickened(max_test_r);

  const auto quant = quantize_circle_arcs(Nested<CircleArc>(), test_bounds).x; // Get appropriate quantizer for test_bounds

  for(int i = 0; i < count; ++i) {
    test_circles.append_empty();
    const auto center = quant(rnd->uniform(test_center_range));
    const Quantized r = max(1, quant.quantize_length(rnd->uniform<real>(0, max_test_r)));
    test_circles.append_to_back(ExactCircleArc(center,r, test_circles.flat.size(), true, false));
  }

  const auto unquantized_input = unquantize_circle_arcs(quant, test_circles);
  const auto union_result = exact_split_circle_arcs(test_circles, 0);
  const auto unquantized_unions = unquantize_circle_arcs(quant, union_result);
  const auto overlap_result = exact_split_circle_arcs(test_circles, 1);
  const auto unquantized_overlaps = unquantize_circle_arcs(quant, overlap_result);
  return tuple(unquantized_input, unquantized_unions, unquantized_overlaps);
}

static Vector<CircleArc, 2> make_circle(Vec2 p0, Vec2 p1) { return vec(CircleArc(p0,1),CircleArc(p1,1)); }
static void random_circle_quantize_test(int seed) {
  auto r = new_<Random>(seed);

  {
    // First check that we can split without hitting any asserts
    const auto sizes = vec(1.e-3,1.e1,1.e3,1.e7);
    Nested<CircleArc, false> arcs;
    arcs.append(make_circle(Vec2(0,0),Vec2(1,0)));
    for(const auto& s : sizes) {
      for(int i = 0; i < 200; ++i) {
        arcs.append(make_circle(s*r->unit_ball<Vec2>(),s*r->unit_ball<Vec2>()));
      }
    }
    circle_arc_union(arcs);
  }

  {
    // Build a bunch of arcs that don't touch
    const auto log_options = vec(1.e-3,1.e-1,1.e1,1.e3);
    const auto max_bounds = Box<Vec2>(Vec2(0.,0.)).thickened(1.e1 * log_options.max());
    const real spacing = 1e-5*max_bounds.sizes().max();
    const real max_x = max_bounds.max.x;

    real curr_x = max_bounds.min.x;
    Nested<CircleArc, false> arcs;
    for(int i = 0; i < 50; ++i) {
      const real remaining = max_x - curr_x;
      if(remaining < spacing)
        break;
      const real log_choice = log_options[r->uniform<int>(0, log_options.size())];
      real next_r = r->uniform<real>(0., min(log_choice, remaining));
      arcs.append(make_circle(Vec2(curr_x, 0.),Vec2(curr_x+next_r, 0.)));
      curr_x += next_r + spacing;
    }

    // Take the union
    auto unioned = circle_arc_union(arcs);

    // If range of sizes is very large, some arcs could be filtered out if they are smaller than quantization threshold...
    GEODE_ASSERT(unioned.size() <= arcs.size());
  }
}

#endif

} // namespace geode
using namespace geode;

void wrap_circle_csg() {
  GEODE_FUNCTION(split_circle_arcs)
  GEODE_FUNCTION(exact_split_circle_arcs)
  GEODE_FUNCTION(canonicalize_circle_arcs)
  GEODE_FUNCTION_2(circle_arc_area,static_cast<real(*)(Nested<const CircleArc>)>(circle_arc_area))
  GEODE_FUNCTION(preprune_small_circle_arcs)
#ifdef GEODE_PYTHON
  GEODE_FUNCTION(_set_circle_arc_dtypes)
  GEODE_FUNCTION(circle_arc_quantize_test)
  GEODE_FUNCTION(random_circle_quantize_test)
  GEODE_FUNCTION(single_circle_handling_test)
#endif
}
