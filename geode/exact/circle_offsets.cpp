#include <geode/exact/circle_csg.h>
#include <geode/exact/circle_offsets.h>
#include <geode/exact/circle_quantization.h>
#include <geode/exact/exact_circle_offsets.h>
#include <geode/exact/PlanarArcGraph.h>
#include <geode/exact/scope.h>

namespace geode {
static constexpr Pb PS = Pb::Implicit;

vector<Nested<CircleArc>> offset_shells(const Nested<const CircleArc> arcs, const real d, const int max_shells) {
  vector<Nested<CircleArc>> result;
  const auto approx_bounds = approximate_bounding_box(arcs).thickened(max(d*max_shells,0));
  const auto quant = make_arc_quantizer(approx_bounds);
  const auto exact_d = quantize_offset(quant,d);
  if(exact_d == 0) {
    const auto u = circle_arc_union(arcs);
    if(max_shells >= 0) for(int i = 0; i < max_shells; ++i) {
      result.push_back(u.copy());
    }
    return result;
  }
  IntervalScope scope;
  assert(exact_d < 0 || max_shells >= 0);

  VertexSet<PS> input_verts;
  auto input_arcs = input_verts.quantize_circle_arcs(quant, arcs);
  const auto input_g = new_<PlanarArcGraph<PS>>(input_verts, input_arcs);

  auto shell = tuple(input_g, extract_region(input_g->topology, faces_greater_than(*input_g, 0)));
  for(int i = 0; i < max_shells; ++i) {
    shell = offset_closed_exact_arcs(*shell.x, shell.y, exact_d);
    if(shell.y.empty())
      break;
    result.push_back(shell.x->unquantize_circle_arcs(quant, shell.y));
  }
  return result;
}

Nested<CircleArc> offset_arcs(const Nested<const CircleArc> arcs, const real d) {
  auto bounds = approximate_bounding_box(arcs);
  if(bounds.empty()) bounds = Box<Vec2>::unit_box(); // We generate a non-degenerate box in case input was empty
  bounds = bounds.thickened(max(d,0.));
  const Quantizer<real,2> quant = make_arc_quantizer(bounds);
  IntervalScope scope;
  const Quantized signed_offset = quantize_offset(quant, d);
  if(signed_offset == 0) {
    return circle_arc_union(arcs); // Since we normally have to union inputs before we can take the offset, we do that here in case caller was relying on that union
  }

  VertexSet<PS> input_verts;
  auto input_arcs = input_verts.quantize_circle_arcs(quant, arcs);
  const auto input_g = new_<PlanarArcGraph<PS>>(input_verts, input_arcs);

  Field<bool, FaceId> interior_faces = faces_greater_than(*input_g, 0);
  const auto input_contours = extract_region(input_g->topology, interior_faces);

  auto offset_g_and_edges = offset_closed_exact_arcs(*input_g, input_contours, signed_offset);
  auto result = offset_g_and_edges.x->unquantize_circle_arcs(quant, offset_g_and_edges.y);
  return result;
}

Nested<CircleArc> offset_open_arcs(const Nested<const CircleArc> arcs, const real d) {
  const auto bounds = approximate_bounding_box(arcs).thickened(max(d,0));
  const auto quant = make_arc_quantizer(bounds);
  const Quantized signed_offset = quantize_offset(quant, d);
  if(signed_offset == 0) {
    return Nested<CircleArc>(); // Union of 0 width shapes is empty
  }

  IntervalScope scope;
  ArcAccumulator<PS> minkowski_terms;

  for(const auto& c : arcs) {
    assert(c.size() > 0);
    for(const int i : range(c.size() - 1)) {
      add_capsule(minkowski_terms, quant(c[i].x), c[i].q, quant(c[i+1].x), signed_offset);
    }
  }

  const Tuple<Ref<PlanarArcGraph<PS>>,Nested<HalfedgeId>> exact_result = minkowski_terms.split_and_union();
  return exact_result.x->unquantize_circle_arcs(quant, exact_result.y);
}

} // namespace geode