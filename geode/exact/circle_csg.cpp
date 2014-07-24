// Robust constructive solid geometry for circular arc polygons in the plane

#include <geode/array/convert.h>
#include <geode/array/Field.h>
#include <geode/array/NestedField.h>
#include <geode/array/sort.h>
#include <geode/exact/circle_csg.h>
#include <geode/exact/circle_quantization.h>
#include <geode/exact/scope.h>
#include <geode/exact/Exact.h>
#include <geode/exact/ExactArcGraph.h>
#include <geode/exact/math.h>
#include <geode/exact/perturb.h>
#include <geode/geometry/BoxTree.h>
#include <geode/geometry/polygon.h>
#include <geode/geometry/traverse.h>
#include <geode/python/stl.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>
namespace geode {

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


template<Pb PS> static Field<int, FaceId> face_winding_depths(const ExactArcGraph<PS>& g) {
  auto edge_windings = Field<int, EdgeId>(g.n_edges(), uninit);
  for(const EdgeId eid : g.edge_ids()) {
    edge_windings[eid] = g.edges[eid].value.winding;
  }
  return compute_winding_numbers(g.graph, g.boundary_face(), edge_windings);
}

template<Pb PS> static Field<bool, FaceId> faces_greater_than(const ExactArcGraph<PS>& g, const int depth) {
  const auto depths = face_winding_depths(g);
  auto result = Field<bool, FaceId>(g.graph->n_faces(), uninit);
  for(const FaceId fid : g.graph->faces())
    result[fid] = (depths[fid] > depth);
  return result;
}

#if 0
// split_circle_arcs will eventually need to support other winding rules like this one
template<Pb PS> static Field<bool, FaceId> nonzero_faces(const ExactArcGraph<PS>& g) {
  const auto depths = face_winding_depths(g);
  auto result = Field<bool, FaceId>(g.graph->n_faces(), uninit);
  for(const FaceId fid : g.graph->faces())
    result[fid] = (depths[fid] != 0);
  return result;
}
#endif

Nested<CircleArc> split_circle_arcs(Nested<const CircleArc> arcs, const int depth) {
  IntervalScope scope;
  const auto PS = Pb::Implicit;
  auto q_and_graph = quantize_circle_arcs<PS>(arcs);
  auto& g = q_and_graph.y;
  g.split_edges();


  Field<bool, FaceId> interior_faces;
  // This would be a good place to switch on a splitting rule
  interior_faces = faces_greater_than(g, depth);

  const auto contour_edges = extract_region(g.graph, interior_faces);
  auto result = g.unquantize_circle_arcs(q_and_graph.x, contour_edges);
  return result;
}

ostream& operator<<(ostream& output, const CircleArc& arc) {
  return output << format("CircleArc([%g,%g],%g)",arc.x.x,arc.x.y,arc.q);
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
  Nested<CircleArc> new_polys(polys.sizes().subset(order).copy(),uninit);
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
}
ARRAY_CONVERSIONS(1,CircleArc)
NESTED_CONVERSIONS(CircleArc)

static void _set_circle_arc_dtypes(PyObject* inexact, PyObject* exact) {
  GEODE_ASSERT(PyArray_DescrCheck(inexact));
  GEODE_ASSERT(PyArray_DescrCheck(exact));
  GEODE_ASSERT(((PyArray_Descr*)inexact)->elsize==sizeof(CircleArc));
  Py_INCREF(inexact);
  Py_INCREF(  exact);
  NumpyDescr<     CircleArc  >::d = (PyArray_Descr*)inexact;
}

static Nested<CircleArc> circle_arc_quantize_test(Nested<const CircleArc> arcs) {
  IntervalScope scope;
  const auto quant = make_arc_quantizer(approximate_bounding_box(arcs));
  auto g = ExactArcGraph<Pb::Implicit>();
  auto edges = g.quantize_and_add_arcs(quant, arcs);
  return g.unquantize_circle_arcs(quant, edges);
}

static ExactArc<Pb::Implicit> make_full_circle_arc(const exact::Vec2 center, const Quantized r) {
  static constexpr auto PS = Pb::Implicit;
  const auto c = ExactCircle<PS>(center, r);
  const auto helper_c = ExactCircle<PS>(vec(center.x + r, center.y), 1);
  const auto helper_i = c.intersection_min(helper_c);
  return ExactArc<PS>({c, helper_i, helper_i});
}

static Tuple<Nested<CircleArc>,Nested<CircleArc>,Nested<CircleArc>> single_circle_handling_test(int seed, int count) {
  const auto test_center_range = Box<Vec2>(Vec2(0,0)).thickened(100);
  const real max_test_r = 100.;
  const auto test_bounds = test_center_range.thickened(max_test_r);
  const auto quant = make_arc_quantizer(test_bounds); // Get appropriate quantizer for test_bounds
  IntervalScope scope;

  auto rnd = new_<Random>(seed);
  auto graph = ExactArcGraph<Pb::Implicit>();

  Nested<HalfedgeId, false> input_contours;
  for(int i = 0; i < count; ++i) {
    const auto center = quant(rnd->uniform(test_center_range));
    const Quantized r = max(1, quant.quantize_length(rnd->uniform<real>(0, max_test_r)));
    const EdgeId added_edge = graph.add_arc(make_full_circle_arc(center, r), ExactArcGraph<Pb::Implicit>::EdgeValue(1,1));
    // Each circle becomes a single ccw halfedge
    input_contours.append_empty();
    input_contours.append_to_back(graph.graph->halfedge(added_edge, false));
  }

  const auto unquantized_input = graph.unquantize_circle_arcs(quant, input_contours.freeze());
  graph.split_edges();
  const auto unquantized_unions = graph.unquantize_circle_arcs(quant, extract_region(graph.graph, faces_greater_than(graph, 0)));
  const auto unquantized_overlaps = graph.unquantize_circle_arcs(quant, extract_region(graph.graph, faces_greater_than(graph, 1)));
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
  GEODE_FUNCTION(canonicalize_circle_arcs)
  GEODE_FUNCTION_2(circle_arc_area,static_cast<real(*)(Nested<const CircleArc>)>(circle_arc_area))
#ifdef GEODE_PYTHON
  GEODE_FUNCTION(_set_circle_arc_dtypes)
  GEODE_FUNCTION(circle_arc_quantize_test)
  GEODE_FUNCTION(random_circle_quantize_test)
  GEODE_FUNCTION(single_circle_handling_test)
#endif
}
