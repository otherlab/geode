// Robust constructive solid geometry for circular arc polygons in the plane

#include <other/core/array/convert.h>
#include <other/core/array/sort.h>
#include <other/core/exact/circle_csg.h>
#include <other/core/exact/circle_predicates.h>
#include <other/core/exact/scope.h>
#include <other/core/geometry/BoxTree.h>
#include <other/core/geometry/polygon.h>
#include <other/core/geometry/traverse.h>
#include <other/core/python/stl.h>
#include <other/core/python/wrap.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Hashtable.h>
namespace other {

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
        if(arc_is_repeated_vertex(arcs.flat,a01,a12))
          continue; // Ignore degenerate arcs
        for (const auto ay : circle_horizontal_intersections(arcs.flat,a,y)) {
          if (circle_arc_contains_horizontal_intersection(arcs.flat,a01,a12,ay)) {
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
    // prune_small_contours will removing these, but they shouldn't be generated in the first place.
    assert(!(contour.size()==2 && contour[0].left!=contour[1].left));

    for (const auto& arc : contour)
      OTHER_ASSERT(arc.radius>0,"Radii must be positive so that symbolic perturbation doesn't make them negative");
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

    // Sort intersections along each segment
    for (const int i : poly) {
      const auto other = others[i];
      // Sort intersections along this segment
      if (other.size() > 1) {
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
        sort(other,PairOrder(arcs,vertices[i]));
      }
    }

    // Find which subarc the horizontal line intersects to determine the start point for walking
    const auto horizontal = info.horizontals[p];
    const int start = horizontal.arc;
    int substart = 0;
    for (;substart<others[start].size();substart++)
      if (circle_arc_contains_horizontal_intersection(arcs,vertices[start],others(start,substart),horizontal))
        break;

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
        , start_xmin(ceil(-start.x.nlo)) // Safe to round up since we'll be comparing against conservative integer boxes
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
            // Symbolically identical arcs that overlap aren't intended to be handled by splitting and may occasionally raise an assert here
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
    const auto start_vertex = substart ? others(start,substart-1).reverse() : vertices[start];
    auto prev = start_vertex;
    int index = start,
        sub = substart;
    do {
      if (sub==others.size(index)) { // Jump to the next segment in the contour
        index++;
        if (index==poly.hi)
          index = poly.lo;
        sub = 0;
        const auto next = vertices[index];
        // Remember this subsegment if it has the right depth, the advance to the next segment
        if (!delta)
          graph.set(prev,next);
        prev = next;
      } else { // Walk across a new intersection
        sub++;
        const auto next = others(index,sub-1);
        // Remember this subsegment if it has the right depth, the advance to the next segment
        if (!delta)
          graph.set(prev,next);
        delta += next.left ^ arcs[next.i0].positive ^ arcs[next.i1].positive ? -1 : 1; 
        prev = next.reverse();
      }
    } while (prev != start_vertex);
  }

  // Walk the graph to produce output polygons
  Hashtable<Vertex> seen;
  Nested<ExactCircleArc,false> output;
  for (const auto& it : graph) {
    const auto start = it.key;
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

// Compute an approximate bounding box for all arcs
Box<Vector<real,2>> approximate_bounding_box(const Nested<const CircleArc>& input) {
  Box<Vector<real,2>> result;
  for (const auto poly : input) {
    for (int j=0,i=poly.size()-1;j<poly.size();i=j++) {
      result.enlarge(bounding_box(poly[i].x,poly[j].x).thickened(.5*abs(poly[i].q)*magnitude(poly[i].x-poly[j].x)));
    }
  }
  return result;
}

// Tweak quantized circles so that they intersect.
static bool tweak_arcs_to_intersect(RawArray<ExactCircleArc> arcs, const int i, const int j) {

  // TODO: For now, we require nonequal centers
  OTHER_ASSERT(arcs[i].center != arcs[j].center);

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
    const auto d = Quantized(floor(d_interval.hi + 1)); // Quantize up
    ri += d;
    rj += d;
    changed = true;
  }
  // Conservatively check if inner circle is too small to intersect (i.e. abs(ri-rj) >= dc)
  if(!certainly_less(Interval(abs(ri-rj)),dc_interval)) {
    Quantized& small_r = ri<rj?ri:rj; // We will grow the smaller radius
    small_r = max(ri,rj)-Quantized(ceil(-dc_interval.nlo-1));
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

Tuple<Quantizer<real,2>,Nested<ExactCircleArc>> quantize_circle_arcs(Nested<const CircleArc> input, const Box<Vector<real,2>> min_bounds) {
  Box<Vector<real,2>> box = min_bounds;
  box.enlarge(approximate_bounding_box(input));

  // Enlarge box quite a lot so that we can closely approximate lines.
  // The error in approximating a straight segment of length L by a circular arc of radius R is
  //   er = L^2/(8*R)
  // If the maximum radius is R, the error due to quantization is
  //   eq = R/bound
  // Equating these, we get
  //   R/bound = L^2/(8*R)
  //   R^2 = L^2*bound/8
  //   R = L*sqrt(bound/8)
  const real max_radius = sqrt(exact::bound/8)*box.sizes().max();
  const auto quant = quantizer(box.thickened(max_radius));

  // Quantize and implicitize each arc
  IntervalScope scope;
  auto output = Nested<ExactCircleArc,false>();
  auto out_sources = Array<int>(); // scratch array for tracking origin of merged arcs

  for (const int p : range(input.size())) {
    const int base = input.offsets[p];
    const auto in = input[p];
    out_sources.clear();
    output.append(Array<ExactCircleArc>()); // Add a zero length array which we will update

    const int n = in.size();
    for(int i = 0; i<n; ++i) {
      const int j = (i+1)%n;
      // Implicitize
      const auto x0 = in[i].x,
                 x1 = in[j].x,
                 dx = x1-x0;
      const auto L = magnitude(dx);
      const auto q = in[i].q;
      // Compute radius, quantize, then compute center from quantized radius to reduce endpoint error
      ExactCircleArc e;
      e.radius = max(Quantized(1),Quantized(round(quant.scale*min(.25*L*abs(q+1/q),max_radius))),Quantized(ceil(.5*quant.scale*L)));
      const auto radius = quant.inverse.inv_scale*e.radius;
      const auto center = L ? .5*(x0+x1)+((q>0)^(abs(q)>1)?1:-1)*sqrt(max(0.,sqr(radius/L)-.25))*rotate_left_90(dx) : x0;
      e.center = quant(center);
      e.index = base+i;
      e.positive = q > 0;
      e.left = false; // Set to false to avoid compiler warning. Actual value set below

      // All output arcs need to intersect their neighbors which will be ensured by calling tweak_arcs_to_intersect
      // Concentric arcs are ignored since those would be redundent after tweaking
      if(output.back().empty() || output.flat.back().center != e.center) {
        out_sources.append(j);
        output.append_to_back(e);
      }
    }

    // Filter out back arcs if concentric with front
    while(output.back().size() > 0 && output.back().front().center == output.back().back().center) {
      out_sources.pop(); // remove from sources list

      // We want output.back().pop(), but have to edit internal structures in the nested array
      output.offsets.back() -= 1;
      output.flat.pop();
    }

    auto out = output.back();
    const int out_n = out.size();
    assert(out_n == out_sources.size());
    // Fill in left flags
    for (int j=0,i=out_n-1;j<out_n;i=j++) {
      const auto x = in[out_sources[i]].x,
                 c0 = quant.inverse(out[i].center),
                 c1 = quant.inverse(out[j].center);
      out[i].left = cross(c1-c0,x-c0)>0;
    }

    // Check in case quantization created a single repeated point
    if(out.size() == 2 && (out[0].left != out[1].left)) {
      // Most degeneracies involve multiple arcs and are readily handled by sybolic perturbation
      // However, in setting the left flags we can create a symbolically degenerate contour which we filter here
      const int n = output.back().size();

      // Remove entire contour from output
      output.flat.pop_elements(n);
      output.offsets.pop();

      // Remove it's sources
      out_sources.pop_elements(n);
    }
  }

  auto result = output.freeze();
  // After quantization some pairs of arcs might not intersect so perform (very small) adjustments
  tweak_arcs_to_intersect(result);
  return tuple(quant,result);
}

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

void reverse_arcs(RawArray<CircleArc>& arcs) {
  arcs.reverse();
  for(int i = arcs.size()-1,j = 0; j<arcs.size(); i=j++) {
    arcs[i].q = arcs[j].q;
  }
}
void reverse_arcs(Nested<CircleArc>& polyarcs) {
 for(auto poly : polyarcs) reverse_arcs(poly);
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

#ifdef OTHER_PYTHON

// Instantiate Python conversions for arrays of circular arcs
namespace {
template<> struct NumpyDescr<CircleArc>{static PyArray_Descr* d;static PyArray_Descr* descr(){OTHER_ASSERT(d);Py_INCREF(d);return d;}};
template<> struct NumpyIsStatic<CircleArc>:public mpl::true_{};
template<> struct NumpyRank<CircleArc>:public mpl::int_<0>{};
template<> struct NumpyArrayType<CircleArc>{static PyTypeObject* type(){return numpy_recarray_type();}};
PyArray_Descr* NumpyDescr<CircleArc>::d;
template<> struct NumpyDescr<ExactCircleArc>{static PyArray_Descr* d;static PyArray_Descr* descr(){OTHER_ASSERT(d);Py_INCREF(d);return d;}};
template<> struct NumpyIsStatic<ExactCircleArc>:public mpl::true_{};
template<> struct NumpyRank<ExactCircleArc>:public mpl::int_<0>{};
template<> struct NumpyArrayType<ExactCircleArc>{static PyTypeObject* type(){return numpy_recarray_type();}};
PyArray_Descr* NumpyDescr<ExactCircleArc>::d;
}
ARRAY_CONVERSIONS(1,CircleArc)
ARRAY_CONVERSIONS(1,ExactCircleArc)

static void _set_circle_arc_dtypes(PyObject* inexact, PyObject* exact) {
  OTHER_ASSERT(PyArray_DescrCheck(inexact));
  OTHER_ASSERT(PyArray_DescrCheck(exact));
  OTHER_ASSERT(((PyArray_Descr*)inexact)->elsize==sizeof(CircleArc));
  OTHER_ASSERT(((PyArray_Descr*)  exact)->elsize==sizeof(ExactCircleArc));
  Py_INCREF(inexact);
  Py_INCREF(  exact);
  NumpyDescr<     CircleArc>::d = (PyArray_Descr*)inexact;
  NumpyDescr<ExactCircleArc>::d = (PyArray_Descr*)  exact;
}

static Nested<CircleArc> circle_arc_quantize_test(Nested<const CircleArc> arcs) {
  const auto e = quantize_circle_arcs(arcs);
  return unquantize_circle_arcs(e.x,e.y);
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
    OTHER_ASSERT(unioned.size() <= arcs.size());
    // ...but for the current values, this should be happening so check that no arcs were destroyed
    OTHER_ASSERT(unioned.size() == arcs.size());
  }
}

#endif

} // namespace other
using namespace other;

void wrap_circle_csg() {
  OTHER_FUNCTION(split_circle_arcs)
  OTHER_FUNCTION(exact_split_circle_arcs)
  OTHER_FUNCTION(canonicalize_circle_arcs)
  OTHER_FUNCTION_2(circle_arc_area,static_cast<real(*)(Nested<const CircleArc>)>(circle_arc_area))
  OTHER_FUNCTION(preprune_small_circle_arcs)
#ifdef OTHER_PYTHON
  OTHER_FUNCTION(_set_circle_arc_dtypes)
  OTHER_FUNCTION(circle_arc_quantize_test)
  OTHER_FUNCTION(random_circle_quantize_test)
#endif
}
