// Robust constructive solid geometry for circular arc polygons in the plane

#include <geode/array/convert.h>
#include <geode/array/Field.h>
#include <geode/array/NestedField.h>
#include <geode/array/sort.h>
#include <geode/exact/circle_csg.h>
#include <geode/exact/circle_offsets.h>
#include <geode/exact/circle_quantization.h>
#include <geode/exact/scope.h>
#include <geode/exact/Exact.h>
#include <geode/exact/math.h>
#include <geode/exact/perturb.h>
#include <geode/exact/PlanarArcGraph.h>
#include <geode/geometry/ArcSegment.h>
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

Nested<CircleArc> split_circle_arcs(Nested<const CircleArc> arcs, const int depth) {
  IntervalScope scope;
  const auto PS = Pb::Implicit;
  auto q_and_graph = quantize_circle_arcs<PS>(arcs);
  const PlanarArcGraph<PS>& g = *(q_and_graph.y);

  Field<bool, FaceId> interior_faces;
  // This would be a good place to switch on a splitting rule
  interior_faces = faces_greater_than(g, depth);
  const auto contour_edges = extract_region(g.topology, interior_faces);
  return g.unquantize_circle_arcs(q_and_graph.x, contour_edges);
}

Nested<CircleArc> split_arcs_by_parity(Nested<const CircleArc> arcs) {
  IntervalScope scope;
  const auto PS = Pb::Implicit;
  auto q_and_graph = quantize_circle_arcs<PS>(arcs);
  auto& g = *(q_and_graph.y);

  Field<bool, FaceId> interior_faces;
  // This would be a good place to switch on a splitting rule
  interior_faces = odd_faces(g);

  const auto contour_edges = extract_region(g.topology, interior_faces);
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
  return area;
}

real circle_arc_area(Nested<const CircleArc> polys) {
  real area = 0;
  for (const auto arcs : polys)
    area += circle_arc_area(arcs);
  return area;
}

real circle_arc_length(Nested<const CircleArc> arcs) {
  real total = 0;
  for(const auto contour : arcs) {
    for(int prev_i = contour.size()-1, curr_i = 0; curr_i < contour.size(); prev_i = curr_i++) {
      total += arc_length(contour[prev_i].x, contour[curr_i].x, contour[prev_i].q);
    }
  }
  return total;
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
  VertexSet<Pb::Implicit> verts;
  const auto contours = verts.quantize_circle_arcs(quant, arcs);
  return verts.unquantize_circle_arcs(quant, contours);
}


static Tuple<Nested<CircleArc>,Nested<CircleArc>,Nested<CircleArc>> single_circle_handling_test(int seed, int count) {
  const auto test_center_range = Box<Vec2>(Vec2(0,0)).thickened(100);
  const real max_test_r = 100.;
  const auto test_bounds = test_center_range.thickened(max_test_r);
  const auto quant = make_arc_quantizer(test_bounds); // Get appropriate quantizer for test_bounds
  IntervalScope scope;
  auto rnd = new_<Random>(seed);

  ArcAccumulator<Pb::Implicit> acc;

  for(int i = 0; i < count; ++i) {
    const auto center = quant(rnd->uniform(test_center_range));
    const Quantized r = max(1, quant.quantize_length(rnd->uniform<real>(0, max_test_r)));

    acc.add_full_circle(ExactCircle<Pb::Implicit>(center, r), ArcDirection::CCW);
    // Each circle becomes a single ccw halfedge
  }

  const auto unquantized_input = acc.vertices.unquantize_circle_arcs(quant, acc.contours);
  auto graph = new_<PlanarArcGraph<Pb::Implicit>>(acc.vertices, acc.contours);
  const auto unquantized_unions   = graph->unquantize_circle_arcs(quant, extract_region(graph->topology, faces_greater_than(*graph, 0)));
  const auto unquantized_overlaps = graph->unquantize_circle_arcs(quant, extract_region(graph->topology, faces_greater_than(*graph, 1)));
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
    // Take the union and make sure we don't hit any asserts
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
    // Take the union and make sure we don't hit any asserts
    auto unioned = circle_arc_union(arcs);

    // If range of sizes is very large, some arcs could be filtered out if they are smaller than quantization threshold...
    GEODE_ASSERT(unioned.size() <= arcs.size());
  }
}

static void circle_arc_area_test() {
  // These should return exactly zero with no rounding error
  GEODE_ASSERT(circle_arc_area(Array<CircleArc>()) == 0.);
  GEODE_ASSERT(circle_arc_area(Nested<CircleArc>()) == 0.);
  const auto degenerate = make_array({CircleArc(Vec2(2.,2.),1.)});
  // No csg functions should return an arc polygon with a single vertex, but circle_arc_area should be able to handle it
  GEODE_ASSERT(circle_arc_area(degenerate) == 0.);

  const auto area_matches = [](const real area, const real expected) -> bool {
    constexpr real area_tolerance = 1e-9; // Maximum expected delta due to floating point rounding
    return abs(area - expected) < area_tolerance;
  };

  // Check result for a square
  auto unit_square = make_array({CircleArc(Vec2(0,0),0), CircleArc(Vec2(1,0),0), CircleArc(Vec2(1,1),0), CircleArc(Vec2(0,1),0)});
  GEODE_ASSERT(area_matches(circle_arc_area(unit_square), 1.));
  reverse_arcs(unit_square); // Should negate area
  GEODE_ASSERT(area_matches(circle_arc_area(unit_square), -1.));

  // Check circle with radius 1
  const auto simple_circle = make_array({CircleArc(Vec2(1,0),1), CircleArc(Vec2(-1,0),1)});
  GEODE_ASSERT(area_matches(circle_arc_area(simple_circle), pi));
  // Offset shouldn't change area
  for(auto& a : simple_circle) a.x += Vec2(100.,200.);
  GEODE_ASSERT(area_matches(circle_arc_area(simple_circle), pi));
  reverse_arcs(simple_circle); // Should negate area
  GEODE_ASSERT(area_matches(circle_arc_area(simple_circle), -pi));

  const auto make_circle = [](const int n_segments, const real radius) -> Array<CircleArc> {
    const real segment_angle = 2.*pi / static_cast<real>(n_segments);
    const real q = tan(0.25*segment_angle);
    Array<CircleArc> arcs;
    for(const int i : range(0, n_segments)) {
      arcs.append(CircleArc(radius*polar(i*segment_angle),q));
    }
    return arcs;
  };

  // Check circle with different numbers of segments
  for(const int n_segments : range(3,9)) {
    const auto arcs = make_circle(n_segments, 1.);
    GEODE_ASSERT(area_matches(circle_arc_area(arcs), pi));
  }

  // Try annulus with outer radius 2 and inner radius 1
  Nested<CircleArc, false> nested_circles;
  nested_circles.append(make_circle(3,2.));
  nested_circles.append(make_circle(4,1.));
  reverse_arcs(nested_circles[1]);
  GEODE_ASSERT(area_matches(circle_arc_area(nested_circles),3.*pi));
  reverse_arcs(nested_circles); // Check reverse_arcs on Nested
  GEODE_ASSERT(area_matches(circle_arc_area(nested_circles),-3.*pi));
}

static real circle_arc_quantization_unit(const Nested<const CircleArc>& arcs) {
  IntervalScope scope;
  const auto quant = make_arc_quantizer(approximate_bounding_box(arcs));
  return quant.inverse.unquantize_length(1.);
}

// Theoretical worst case error for any point on polyarc after round trip quantization and back assuming maximum abs(q)<=1 
// Result is in multiples of circle_arc_quantization_unit
// This ignores floating point rounding errors (which should be small by comparison).
// WARNING: This is not a guarantee, only an educated guess (though it seems to hold up in testing)
static real circle_arc_max_quantization_error_guess() {
  // Moving ends of an arc segment moves midpoint by the same or smaller distance as long as abs(q) <= 1
  const auto endpoint_snap_error = sqrt(0.5); // Noise added when snapping to quantized point
  // During quantization, we compute a rational approximation of q which will differ by 1/exact::bounds from the expected value of q (assuming again that abs(q) <= 1)
  // This approximation of q can alter midpoint by 1/2*distance_between_endpoints*delta_q
  // The arc quantizer is chosen such that longest possible edge will be on the order of sqrt(exact::bound/8)
  // This means approximation of q adds 1/2*sqrt(exact::bound/8)/exact::bound which will be quite small
  const auto q_error = 1./sqrt(exact::bound/32.);
  // Computing the center of the circle is exactly rounded using quantized endpoints and rational approximation of q so it can be off by at most sqrt(0.5)
  const auto center_error = sqrt(0.5) + q_error;
  // Radius is computed via distance from endpoints to center so full error from rounding center can be carried over plus an additional sqrt(0.5) error from rounding radius
  const auto radius_error = center_error + sqrt(0.5);
  // Rounding center further from theoretical value of arc midpoint will also increase radius which will help to stabilize position of arc midpoint
  // This means center_error and radius_error will tend to cancel, however I can't think of a proof that rules out them adding in the worst case
  const auto midpoint_error = endpoint_snap_error + center_error + radius_error;
  // When going back from exact to inexact representation we add error from using the approximate intersections and from culling helper circles
  const auto unquant_error = ApproxIntersection::tolerance() + constructed_arc_endpoint_error_bound(); // Maximum distance to constructed intersection
  // Worst case error should be at midpoints of arcs (error at any endpoint_snap_error should be at most endpoint_snap_error + unquant_error)
  return (midpoint_error + unquant_error);
}

// Theoretical worst case error for offset starting with exact arcs
// Result is in multiples of circle_arc_quantization_unit
// Add circle_arc_max_quantization_error_guess() if starting with unquantized arcs
// WARNING: This is not a guarantee, only an educated guess
static real circle_arc_max_offset_error_guess() {
  // We have to round offset to a quantized value
  const auto offset_snap_error = 0.5;
  // Each arc in input gets expanded into a curved capsule
  // The capsule 'sides' will be exact, but endcaps require some approximate constructions
  // We construct endcaps centered at vertices, but we only have approximate value of intersection which we must round to a quantized value
  const auto endcap_center_error = ApproxIntersection::tolerance() + sqrt(0.5);
  // We have to pad radius of endcaps to ensure we get non-degenerate intersections with sides
  const auto endcap_safety_margin = 2.;
  // When endcaps aren't far apart enough too guarantee well behaved constructions padding gets doubled
  const auto degenerate_endcap_padding = 2.*endcap_safety_margin;
  // left_flags_safe is computed by comparing conservative interval distance between arc endpoints
  // Since it's a conservative comparison it doesn't actually guarantee arc is short, however current interval implementation should be tight enough
  // Worst case error would occur at endcaps
  return offset_snap_error + endcap_center_error + degenerate_endcap_padding;
}

static void check_circle_quantize(const Nested<const CircleArc>& arcs0) {
  const auto arcs1 = circle_arc_quantize_test(arcs0);
  // Any inserted helper arcs should be culled when unquantizing so we should have one-to-one correspondence
  // However this seems to have some failures. Maybe very small arcs in input are being culled? Needs more testing
  if(arcs0.offsets != arcs1.offsets) {
    // Can't directly compare arcs if offsets are different
    // Ideally we would find what the difference is and check check correspondence on remaining arcs, but for now we just abort
    GEODE_ASSERT(false);
  }
  const auto max_error = circle_arc_quantization_unit(arcs0)*circle_arc_max_quantization_error_guess();
  // Loop over pairs of arcs and compare
  for(const int i : range(arcs0.size())) {
    const auto end_j = arcs0[i].size();
    for(const int j0 : range(end_j)) {
      const int j1 = (j0+1)%end_j;
      const auto a0 = ArcSegment(arcs0[i][j0].x,arcs0[i][j1].x,arcs0[i][j0].q);
      const auto a1 = ArcSegment(arcs1[i][j0].x,arcs1[i][j1].x,arcs1[i][j0].q);
      // Check that start and end vertices are in expected error bound
      GEODE_ASSERT((a0.x0-a1.x0).magnitude() <= max_error);
      GEODE_ASSERT((a0.x1-a1.x1).magnitude() <= max_error);
      // Check that midpoint of arc is in expected error bound
      const auto midpoint_error = (a0.arc_mid()-a1.arc_mid()).magnitude();
      GEODE_ASSERT(midpoint_error <= max_error);
    }
  }
}
#endif
} // namespace geode
using namespace geode;

void wrap_circle_csg() {
  GEODE_FUNCTION(split_circle_arcs)
  GEODE_FUNCTION(split_arcs_by_parity)
  GEODE_FUNCTION(canonicalize_circle_arcs)
  GEODE_FUNCTION_2(circle_arc_area,static_cast<real(*)(Nested<const CircleArc>)>(circle_arc_area))
  GEODE_FUNCTION(circle_arc_length)
  GEODE_FUNCTION(offset_arcs)
  GEODE_FUNCTION(offset_open_arcs)
  GEODE_FUNCTION(offset_shells)
#ifdef GEODE_PYTHON
  GEODE_FUNCTION(_set_circle_arc_dtypes)
  GEODE_FUNCTION(circle_arc_quantize_test)
  GEODE_FUNCTION(random_circle_quantize_test)
  GEODE_FUNCTION(single_circle_handling_test)
  GEODE_FUNCTION(circle_arc_area_test)
  GEODE_FUNCTION(circle_arc_max_quantization_error_guess)
  GEODE_FUNCTION(circle_arc_max_offset_error_guess)
  GEODE_FUNCTION(circle_arc_quantization_unit)
  GEODE_FUNCTION(check_circle_quantize)
#endif
}
