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
#include <other/core/structure/Hashtable.h>

namespace other {

typedef RawArray<const ExactCircleArc> Arcs;
typedef RawArray<const int> Next;
typedef RawArray<const Vertex> Vertices;
typedef exact::Vec2 EV2;

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

Nested<ExactCircleArc> exact_split_circle_arcs(Nested<const ExactCircleArc> nested, const int depth) {
  // Check input consistency
  for (const int p : range(nested.size())) {
    const auto contour = nested[p];
    if (contour.size()==2 && contour[0].left!=contour[1].left) {
      throw RuntimeError(format("exact_split_circle_arcs: contour %d is degenerate of size 2",p));
    }
    for (const auto& arc : contour)
      OTHER_ASSERT(arc.radius>0,"Radii must be positive so that symbolic perturbation doesn't make them negative");
  }

  // Prepare for interval arithmetic
  IntervalScope scope;
  Arcs arcs = nested.flat;

  // Build a convenience array of (prev,next) pairs to avoid dealing with the nested structure.
  const Array<const int> next = closed_contours_next(nested);

  // Precompute all intersections between connected arcs
  const Array<const Vertex> vertices = compute_verticies(arcs, next); // vertices[i] is the start of arcs[i]

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
            && (i1==j2 || i2==j1 ||
              (!arcs_from_same_circle(arcs, i1, j1) && circles_intersect(arcs,i1,j1)) // Ignore intersections of arc and itself
              )) {
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
  const auto tree = new_<BoxTree<EV2>>(arc_boxes(next,arcs,vertices),1);
  Intersections pairs(tree,next,arcs,vertices);
  double_traverse(*tree,pairs);

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
  for (const int p : range(nested.size())) {
    const auto poly = range(nested.offsets[p],nested.offsets[p+1]);
    // Compute the depth of the first point in the contour by firing a ray along the positive x axis.
    struct Depth {
      const BoxTree<EV2>& tree;
      Next next;
      Arcs arcs;
      Vertices vertices;
      const Vertex start;
      int depth;

      Depth(const BoxTree<EV2>& tree, Next next, Arcs arcs, Vertices vertices, const int i0, const int i1)
        : tree(tree), next(next), arcs(arcs), vertices(vertices)
        , start(vertices[i1])
        // If we intersect no other arcs, the depth depends on the orientation of direction = (1,0) relative to inwards and outwards arcs
        , depth(local_x_axis_depth(arcs,vertices[i0],start,vertices[next[i1]])) {}

      bool cull(const int n) const {
        const auto box = tree.boxes(n),
                   sbox = start.box();
        return box.max.x<sbox.min.x || box.max.y<sbox.min.y || box.min.y>sbox.max.y;
      }

      void leaf(const int n) {
        assert(tree.prims(n).size()==1);
        const int j = tree.prims(n)[0];
        if (start.i0!=j && start.i1!=j)
          depth -= horizontal_depth_change(arcs,start,vertices[j],
                                                      vertices[next[j]]);
      }
    };
    Depth ray(tree,next,arcs,vertices,poly.back(),poly[0]);
    single_traverse(*tree,ray);

    // Walk around the contour, recording all subarcs at the desired depth
    int delta = ray.depth-depth;
    Vertex prev = vertices[poly[0]];
    for (const int i : poly) {
      const auto other = others[i];
      // Sort intersections along this segment
      if (other.size() > 1) {
        struct PairOrder {
          Next next;
          Arcs arcs;
          const Vertex start; // The start of the segment

          PairOrder(Next next, Arcs arcs, Vertex start)
            : next(next), arcs(arcs)
            , start(start) {}

          bool operator()(const Vertex b0, const Vertex b1) const {
            assert(start.i1==b0.i0 && b0.i0==b1.i0);
            if (b0.i1==b1.i1 && b0.left==b1.left)
              return false;
            return circle_arc_intersects_circle(arcs,start,b1,b0);
          }
        };
        sort(other,PairOrder(next,arcs,prev));
      }
      // Walk through each intersection of this segment, updating delta as we go and remembering the subsegment if it has the right depth
      for (const auto o : other) {
        if (!delta)
          graph.set(prev,o);
        delta += o.left ^ arcs[o.i0].positive ^ arcs[o.i1].positive ? -1 : 1;
        prev = o.reverse();
      }
      // Advance to the next segment
      const auto n = vertices[next[i]];
      if (!delta)
        graph.set(prev,n);
      prev = n;
    }
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
  for (const auto poly : input)
    for (int j=0,i=poly.size()-1;j<poly.size();i=j++)
      result.enlarge(bounding_box(poly[i].x,poly[j].x).thickened(.5*abs(poly[i].q)*magnitude(poly[i].x-poly[j].x)));
  return result;
}

// Tweak quantized circles so that they intersect.
static bool tweak_arcs_to_intersect(RawArray<ExactCircleArc> arcs, const int i, const int j) {
  OTHER_WARNING("TODO: If the same circle is used in multiple arcs, it probably isn't safe to change only one of them!!!!!");

  // TODO: For now, we require nonequal centers
  OTHER_ASSERT(arcs[i].center != arcs[j].center);

  bool changed = false;

  const double dc = magnitude(Vector<double,2>(arcs[i].center-arcs[j].center));
  Quantized &ri = arcs[i].radius,
            &rj = arcs[j].radius;

  if (ri+rj <= dc) {
    const auto d = Quantized(floor((dc-ri-rj)/2+1));
    ri += d;
    rj += d;
    changed = true;
  }
  if (abs(ri-rj) >= dc) {
    (ri<rj?ri:rj) = max(ri,rj)-Quantized(ceil(dc-1));
    changed = true;
  }

  return changed;
}

// Tweak quantized circles so that they intersect.
void tweak_arcs_to_intersect(RawArray<ExactCircleArc> arcs) {
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
  auto output = Nested<ExactCircleArc>::empty_like(input);
  for (const int p : range(input.size())) {
    const int base = input.offsets[p];
    const auto in = input[p];
    const auto out = output[p];
    const int n = in.size();
    for (int j=0,i=n-1;j<n;i=j++) {
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
      e.left = false; // set to false to avoid compiler warning
      out[i] = e;
    }
    // Fill in left flags
    for (int j=0,i=n-1;j<n;i=j++) {
      const auto x = in[j].x,
                 c0 = quant.inverse(out[i].center),
                 c1 = quant.inverse(out[j].center);
      out[i].left = cross(c1-c0,x-c0)>0;
    }
  }

  // After quantization some pairs of arcs might not intersect so perform (very small) adjustments
  tweak_arcs_to_intersect(output);

  return tuple(quant,output);
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
  return output << format("ExactCircleArc([%d,%d],%d,%c%c)",arc.center.x,arc.center.y,arc.radius,arc.positive?'+':'-',arc.left?'L':'R');
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
#endif

} // namespace other
using namespace other;

void wrap_circle_csg() {
  OTHER_FUNCTION(split_circle_arcs)
  OTHER_FUNCTION(exact_split_circle_arcs)
  OTHER_FUNCTION(canonicalize_circle_arcs)
  OTHER_FUNCTION_2(circle_arc_area,static_cast<real(*)(Nested<const CircleArc>)>(circle_arc_area))
#ifdef OTHER_PYTHON
  OTHER_FUNCTION(_set_circle_arc_dtypes)
  OTHER_FUNCTION(circle_arc_quantize_test)
#endif
}
