#pragma once
// A data structure for handling connected components in a HalfedgeGraph

#include <geode/mesh/HalfedgeGraph.h>

namespace geode {

struct ComponentData
{
  struct ComponentInfo {
    BorderId seed_border;
    FaceId exterior_face;
    ComponentInfo() {}
    ComponentInfo(BorderId _seed_border) : seed_border(_seed_border) {}
  };
  Field<ComponentId, BorderId> border_to_component;
  Field<ComponentInfo, ComponentId> components_;
  ComponentData(const HalfedgeGraph& g);

  int n_components() const { return components_.size(); }
  Range<IdIter<ComponentId>> components() const { return Range<IdIter<ComponentId>>(ComponentId(0), ComponentId(n_components())); }

  BorderId border(const ComponentId cid) const { return components_[cid].seed_border; }
  ComponentId component(const BorderId bid) const { return border_to_component[bid]; }
  ComponentInfo& operator[](const BorderId bid) { return components_[component(bid)]; }
  ComponentInfo& operator[](const ComponentId cid) { return components_[cid]; }
};

} // namespace geode