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

} // namespace geode