#include "clip_open_arcs.h"

#include <geode/array/ConstantMap.h>

#include <geode/exact/circle_quantization.h>
#include <geode/exact/PlanarArcGraph.h>
#include <geode/exact/scope.h>

namespace geode {

static constexpr auto PS = Pb::Implicit;

// If fn returns true, stops traversing contour
template<class Fn> static bool ordered_traverse_halfedges(const PlanarArcGraph<PS>& graph, const RawArcContour contour, Fn&& fn) {
  for(const SignedArcInfo signed_arc : contour) {
    // This is the next arc along the contour
    const auto ccw_arc = graph.vertices.ccw_arc(signed_arc);

    // Any edges for this arc will be on this circle
    const CircleId arc_circle = graph.vertices.reference_cid(ccw_arc.src);

    if(signed_arc.positive()) {
      // This iterator will let us traverse edges
      auto incident_iter = graph.incident_order.circulator(arc_circle, ccw_arc.src);
      do {
        const auto eid = graph.outgoing_edges[*incident_iter];
        assert(graph.topology->valid(eid));
        const auto hid = directed_edge(eid, signed_arc.direction());
        if(fn(hid))
          return true;
        ++incident_iter;
      } while(*incident_iter != ccw_arc.dst);
    }
    else {
      auto incident_iter = graph.incident_order.circulator(arc_circle, ccw_arc.dst);
      do {
        --incident_iter;
        const auto eid = graph.outgoing_edges[*incident_iter];
        assert(graph.topology->valid(eid));
        const auto hid = directed_edge(eid, signed_arc.direction());
        if(fn(hid))
          return true;
      } while(*incident_iter != ccw_arc.src);
    }
  }
  return false;
}

static void split_arc_paths(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs,
                           const ClippingMode mode, bool invert_inside,
                           Nested<CircleArc, false>& result, Array<int>* source_ids) {
  const auto bounds = Box<Vec2>::combine(approximate_bounding_box(closed_arcs),
                                         approximate_bounding_box(open_arcs));
  const auto quant = make_arc_quantizer(bounds);

  // Build exact representation of inputs
  GEODE_UNUSED IntervalScope scope;
  VertexSet<PS> input_verts;

  Array<int8_t> input_weights;
  ArcContours input_contours;

  // Quantize open arcs and set weights to 0
  input_verts.quantize_circle_arcs(quant, open_arcs, input_contours, true);
  input_weights.extend(ConstantMap<int8_t>{input_contours.size()-input_weights.size(), 0});

  // Save which input arcs are the open ones
  const int num_open_contours = input_contours.size();

  // Quantize closed arcs and set weights to +/-1
  input_verts.quantize_circle_arcs(quant, closed_arcs, input_contours);
  input_weights.extend(ConstantMap<int8_t>{input_contours.size()-input_weights.size(),1});

  // Compute planer embedding
  const auto graph = new_<PlanarArcGraph<PS>>(input_verts, input_contours, input_weights);

  // Find interior of closed faces
  auto interior_faces = faces_greater_than(*graph, 0);
  if(invert_inside) {
    for(auto& f : interior_faces.flat) f = !f;
  }

  ArcContours output_contours;

  if(source_ids) source_ids->clear();

  if(mode != ClippingMode::AllOrNothing) {
    const bool has_early_exit = (mode == ClippingMode::OnlyStartAtStart);
    for(const int i : range(num_open_contours)) {
      auto tail_vertex = VertexId{}; // Initialize to an invalid id

      // Walk over input arcs saving edges that weren't clipped
      ordered_traverse_halfedges(*graph, input_contours[i], [&graph, &tail_vertex, &output_contours, &interior_faces, &source_ids, i, has_early_exit](const HalfedgeId hid) {
        const auto faces = vec(graph->topology->face(hid), graph->topology->opp_face(hid));
        // Inputs can be exactly on boundary. We require edges to be strictly inside to be kept
        const bool is_inside = interior_faces[faces[0]] && interior_faces[faces[1]];
        if(is_inside) {
          if(!tail_vertex.valid())
            output_contours.start_contour();
          output_contours.append_to_back({graph->src(hid), arc_direction(hid)});
          tail_vertex = to_vid(graph->dst(hid));
        }
        else {
          if(tail_vertex.valid()) {
            output_contours.end_open_contour(tail_vertex);
            if(source_ids) source_ids->append(i);
            tail_vertex = VertexId{};
          }
        }
        return has_early_exit ? !is_inside // If we need to short circuit, indicate if we hit an outside point
                              : false; // If we don't short circuit, always return false
      });

      if(tail_vertex.valid()) {
        output_contours.end_open_contour(tail_vertex);
        if(source_ids) source_ids->append(i);
      }
    }
  }
  else {
    for(const int i : range(num_open_contours)) {
      // Walk over input arcs saving edges that weren't clipped
      const bool clipped = ordered_traverse_halfedges(*graph, input_contours[i], [&graph, &interior_faces](const HalfedgeId hid) {
        const auto faces = vec(graph->topology->face(hid), graph->topology->opp_face(hid));
        // Inputs can be exactly on boundary. We require edges to be strictly inside to be kept
        const bool is_inside = interior_faces[faces[0]] && interior_faces[faces[1]];
        return !is_inside;
      });

      if(!clipped) {
        result.append(open_arcs[i]); // Just use input path so we don't have to unquantize
        if(source_ids) source_ids->append(i);
      }
    }
    assert(!source_ids || source_ids->size() == result.size());
    return; // We just copied over input paths, so we don't have to bother unquantizing
  }

  // Remove redundant vertices in the middle of contours
  const auto simple_output_contours = graph->combine_concentric_arcs(output_contours);

  result.offsets.preallocate(simple_output_contours.size() + 1);
  result.flat.preallocate(simple_output_contours.store.total_size());

  const auto& verts = graph->vertices;

  // TODO: This loop is basically unquantize_open_arcs. It should probably be moved into some PlanarArcGraph class
  for(const auto contour : simple_output_contours) {
    bool can_cull = false;
    result.append_empty();
    for(const SignedArcInfo sa : contour) {
      if(can_cull && verts.reference(sa.head()).radius == constructed_arc_endpoint_error_bound()) {
        can_cull = false;
        continue;
      }
      can_cull = true;
      if(sa.is_full_circle()) {
        // Split arc at midpoint since inexact CircleArc can't represent a full circle
        const auto x0 = quant.inverse(verts.approx(sa.head()).guess());
        const auto circle_center = quant.inverse(verts.reference(sa.head()).center);
        // Reflect x0 across circle_center to get opposite point
        const auto opp_point = circle_center - (x0 - circle_center);
        // Arc start at x0 headed to opp_point
        result.append_to_back(CircleArc{x0, static_cast<double>(sign(sa.direction()))});
        // Arc starting at opp_point headed back to x0 (which should be next point)
        result.append_to_back(CircleArc{opp_point, static_cast<double>(sign(sa.direction()))});
      }
      else {
        const auto unsigned_arc = verts.arc(verts.ccw_arc(sa));
        const auto x0 = quant.inverse(verts.approx(sa.head()).guess());
        result.append_to_back(CircleArc{x0, sign(sa.direction()) * unsigned_arc.q()});
      }
    }

    assert(!contour.is_closed());
    // Add one last point to finish off the arc
    // q value should never be used
    const auto x1 = quant.inverse(verts.approx(iid_cl(contour.tail())).guess());
    result.append_to_back(CircleArc{x1, 0.});
  }
  assert(!source_ids || source_ids->size() == result.size());
}

Nested<CircleArc> arc_path_intersection(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs, ClippingMode mode) {
  Nested<CircleArc, false> result;
  split_arc_paths(closed_arcs, open_arcs, mode, false, result, nullptr);
  return result;
}

Nested<CircleArc> arc_path_difference(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs, ClippingMode mode) {
  Nested<CircleArc, false> result;
  split_arc_paths(closed_arcs, open_arcs, mode, true, result, nullptr);
  return result;
}

std::pair<Nested<CircleArc>,Array<int>> arc_path_intersection_with_coorespondence(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs, ClippingMode mode) {
  Nested<CircleArc, false> paths;
  Array<int> sources;
  split_arc_paths(closed_arcs, open_arcs, mode, false, paths, &sources);
  return std::make_pair(paths, sources);
}

std::pair<Nested<CircleArc>,Array<int>> arc_path_difference_with_coorespondence(const Nested<const CircleArc> closed_arcs, const Nested<const CircleArc> open_arcs, ClippingMode mode) {
  Nested<CircleArc, false> paths;
  Array<int> sources;
  split_arc_paths(closed_arcs, open_arcs, mode, true, paths, &sources);
  return std::make_pair(paths, sources);
}

} // geode namespace
