#include <geode/exact/circle_offsets.h>
#include <geode/exact/circle_predicates.h>
#include <geode/geometry/polygon.h>
#include <geode/utility/stl.h>
#include <geode/python/wrap.h>
#include <geode/python/stl.h>

namespace geode {

typedef RawArray<const ExactCircleArc> Arcs;
typedef RawArray<const Vertex> Vertices;


static Array<ExactCircleArc> build_arc_capsule(Arcs arcs, const Vertex& start, const Vertex& end, ExactInt signed_offset, int& next_free_index) {
  assert(start.i1 == end.i0);

  const ExactCircleArc& curr_arc = arcs[start.i1];
  const ExactInt abs_offset = abs(signed_offset);
  const ExactInt outer_r = curr_arc.radius + abs_offset;
  const ExactInt inner_r = curr_arc.radius - abs_offset;

  assert(curr_arc.radius > 0 && outer_r > 0 && abs_offset > 0);

  const bool positive = signed_offset > 0;

  Array<ExactCircleArc> result;

  const auto delta = start.rounded - end.rounded;

  result.append(ExactCircleArc());
  result.back().center = curr_arc.positive ? start.rounded : end.rounded;
  result.back().radius = abs_offset;
  result.back().index = next_free_index++;
  result.back().positive = true;
  result.back().left = true;

  result.append(ExactCircleArc());
  result.back().center = curr_arc.center;
  result.back().radius = outer_r;
  result.back().index = next_free_index++;
  result.back().positive = true;
  result.back().left = true;

  if(delta.L1_Norm() != 0 || inner_r > 0) {
    result.append(ExactCircleArc());
    result.back().center = curr_arc.positive ? end.rounded : start.rounded;
    result.back().radius = abs_offset;
    result.back().index = next_free_index++;
    result.back().positive = true;
    result.back().left = false;
  }

  if(inner_r > 0) { // if inner_r < 0, then inside arc has completely eroded and endcaps will connect together
    result.append(ExactCircleArc());
    result.back().center = curr_arc.center;
    result.back().radius = inner_r;
    result.back().index = next_free_index++;
    result.back().positive = false;
    result.back().left = false;
    assert(result.size() == 4);
  }

  if(result.size() > 2) {
    // We will grow one end of each capsule very slightly
    // This helps avoid the cost of evaluting perturbed degenerate intersections where capsules for consecutive arcs share endpoints
    result[curr_arc.positive ? 0 : 2].radius += 5;
    result[curr_arc.positive ? 2 : 0].radius += 2;
  }

  tweak_arcs_to_intersect(result);

  if(!positive) {
    std::reverse(result.begin(), result.end());
    for(auto& arc : result) {
      arc.positive = !arc.positive;
      arc.left = !arc.left;
    }
  }

  return result;
}

Nested<ExactCircleArc> exact_offset_arcs(const Nested<const ExactCircleArc> nested_arcs, const ExactInt offset) {
  // Compute some preliminary info
  const RawArray<const ExactCircleArc> arcs = nested_arcs.flat;
  const Array<const int> next = closed_contours_next(nested_arcs);
  const Array<const Vertex> vertices = compute_vertices(arcs, next); // vertices[i] is the start of arcs[i]

  // offset shape is union of the origional shape and the thickened border
  // for a negative offset the thickened border will need to be negatively oriented

  // start with a copy of the input arcs
  auto minkowski_terms = Nested<ExactCircleArc, false>::copy(nested_arcs);

  // We will use consecutive indicies of newly constructed geometry starting from max(arc index)+1
  // TODO: For a large series of operations we might need to use some sort of free list to avoid overflow
  int next_free_index = 0;
  for(const ExactCircleArc& ea : minkowski_terms.flat) {
    if(ea.index >= next_free_index) next_free_index = ea.index+1;
  }

  // for each arc in the original shape, thicken by offset and add to back
  // capsule will have negitive winding if offset < 0
  for(int arc_i : range(arcs.size())) {
    minkowski_terms.append(build_arc_capsule(arcs, vertices[arc_i], vertices[next[arc_i]], offset, next_free_index));
    GEODE_ASSERT(next_free_index > 0, "Index overflow using consecutive indicies. Free index data structure needed!");
  }
  auto result = exact_split_circle_arcs(minkowski_terms, 0);
  return result;
}

Nested<CircleArc> offset_arcs(const Nested<const CircleArc> raw_arcs, const real offset_amount) {
  const auto min_bounds = approximate_bounding_box(raw_arcs).thickened(abs(offset_amount));
  const auto qu_ea = quantize_circle_arcs(raw_arcs, min_bounds);

  const Quantizer<real,2>& quantizer = qu_ea.x;
  const Nested<ExactCircleArc>& exact_arcs = qu_ea.y;

  const ExactInt exact_offset = ExactInt(floor(quantizer.scale*offset_amount));

  if(exact_offset == 0) {
    GEODE_WARNING("Arc offset amount was below numerical representation threshold! (this should be a few nm for geometry that fits in meter bounds)");
    return raw_arcs.copy();
  }

  auto result = unquantize_circle_arcs(quantizer, exact_offset_arcs(exact_arcs, exact_offset));
  return result;
}

std::vector<Nested<CircleArc>> offset_shells(const Nested<const CircleArc> raw_arcs, const real shell_thickness, const int num_shells) {
  GEODE_ASSERT(num_shells > 0);
  const auto min_bounds = approximate_bounding_box(raw_arcs).thickened(abs(shell_thickness)*num_shells);
  const auto qu_ea = quantize_circle_arcs(raw_arcs, min_bounds);

  const Quantizer<real,2>& quantizer = qu_ea.x;
  const Nested<ExactCircleArc>& exact_arcs = qu_ea.y;

  const ExactInt exact_shell = ExactInt(floor(quantizer.scale*shell_thickness));

  if(exact_shell == 0) {
    GEODE_WARNING("Arc offset amount was below numerical representation threshold! (this should be a few nm for geometry that fits in meter bounds)");
    return make_vector(raw_arcs.copy());
  }

  std::vector<Nested<CircleArc>> result;
  Nested<ExactCircleArc> temp_shell = exact_arcs;
  for(int i = 0; i< num_shells; ++i) {
    temp_shell = exact_offset_arcs(temp_shell, exact_shell);
    //temp_shell = remove_degenerate_arcs(temp_shell, 200);
    result.push_back(unquantize_circle_arcs(quantizer, temp_shell));
  }
  return result;
}

} // namespace geode
using namespace geode;
void wrap_circle_offsets() {
  GEODE_FUNCTION(offset_arcs)
  GEODE_FUNCTION(offset_shells)
}

