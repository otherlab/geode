#include <geode/mesh/ComponentData.h>

namespace geode {

ComponentData::ComponentData(const HalfedgeGraph& g)
 : border_to_component(g.n_borders())
{
  assert(g.has_all_border_data());

  for(const auto seed_b : g.borders()) {
    if(border_to_component[seed_b].valid())
      continue;
    Array<BorderId> queue;
    const ComponentId cid = components_.append(ComponentInfo(seed_b));
    queue.append(seed_b);
    border_to_component[seed_b] = cid;

    while(!queue.empty()) {
      const BorderId top = queue.pop();
      for(const HalfedgeId eid : g.border_edges(top)) {
        const BorderId opp_border = g.opp_border(eid);
        auto& opp_cid = border_to_component[opp_border];
        if(!opp_cid.valid()) {
          opp_cid = cid;
          queue.append(opp_border);
        }
        assert(opp_cid == cid);
      }
    }
  }
}

void initialize_path_faces(const RawArray<const HalfedgeId> path_from_infinity, FaceId& infinity_face, HalfedgeGraph& g, ComponentData& cd) {
  // If we haven't created the infinity_face, attempt to do so now
  if(!infinity_face.valid() && !path_from_infinity.empty()) {
    infinity_face = g.new_face_for_border(g.border(path_from_infinity.front()));
  }

  auto current_face = infinity_face;

  for(const HalfedgeId hit : path_from_infinity) {
    const BorderId hit_border = g.border(hit);

    auto& hit_component = cd[hit_border];
    if(!hit_component.exterior_face.valid())
      hit_component.exterior_face = current_face; // First hit on a component is always its exterior face

    if(!g.face(hit_border).valid())
      g.add_to_face(current_face, hit_border); // Add the border that we hit to the current face if it isn't already part

    assert(g.face(hit_border) == current_face); // Border should already be part of face or have just been added

    // Stepping across edge will switch to a new border...
    const BorderId opp_border = g.border(g.reverse(hit));
    current_face = g.face(opp_border); // ...and also a new face

    assert(cd.component(opp_border) == cd.component(hit_border)); // We should have stayed on the same component when crossing the edge

    if(!current_face.valid()) // If the new border didn't already have a face, we need to add one
      current_face = g.new_face_for_border(opp_border);
  }
}

} // namespace geode