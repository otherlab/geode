// Set colors on a TriMesh
#pragma once

#include <geode/openmesh/TriMesh.h>
#include <geode/utility/compose.h>

#ifdef GEODE_OPENMESH
namespace geode {

// Turn a mesh property into a function object

template<class PropHandle> struct PropFunction {
  typedef typename PropHandle::value_type result_type;

  const TriMesh& mesh;
  PropHandle prop;

  PropFunction(const TriMesh& mesh, PropHandle prop)
    : mesh(mesh), prop(prop) {}

  result_type operator()(typename PropToHandle<PropHandle>::type id) const {
    return mesh.property(prop,id);
  }
};

template<class PropHandle> static inline PropFunction<PropHandle> prop_function(const TriMesh& mesh, PropHandle prop) {
  return PropFunction<PropHandle>(mesh,prop);
}

// Visualize things on the vertices, edges, or faces
template<class Handle> GEODE_CORE_EXPORT void visualize(TriMesh& mesh, const function<TriMesh::Color(Handle)>& color);

// Visualize properties using a transfer function
template<class PropHandle,class Transfer> static inline void visualize(TriMesh& mesh, PropHandle prop, const Transfer& transfer) {
  return visualize<typename PropToHandle<PropHandle>::type>(mesh,compose(transfer,prop_function(mesh,prop)));
}

// Transfer functions for visualization
template<class T> struct TransferSmooth {
  typedef TriMesh::Color result_type;

  T vmin, vmax, vzero;
  TriMesh::Color color_min, color_max, color_zero;

  // create and set range to range of property
  TransferSmooth(RawArray<const T> data,
                 TriMesh::Color color_min = TriMesh::Color(255, 0, 0, 255),
                 TriMesh::Color color_max = TriMesh::Color(0, 255, 0, 255),
                 TriMesh::Color color_zero = TriMesh::Color(0, 0, 255, 255))
    : color_min(color_min)
    , color_max(color_max)
    , color_zero(color_zero)
  {
    if (!data.size())
      return;
    GEODE_ASSERT(data.size());
    vmin = data.min();
    vmax = data.max();
    vzero = vmin + vmax / 2;
  }

  // set range to fixed values
  TransferSmooth(T vmin = -1, T vmax = 1, T vzero = 0,
                 TriMesh::Color color_min = TriMesh::Color(255, 0, 0, 255),
                 TriMesh::Color color_max = TriMesh::Color(0, 255, 0, 255),
                 TriMesh::Color color_zero = TriMesh::Color(0, 0, 255, 255))
    : vmin(vmin)
    , vmax(vmax)
    , vzero(vzero)
    , color_min(color_min)
    , color_max(color_max)
    , color_zero(color_zero)
  {}

  TriMesh::Color operator()(T const &value) const {
    if (value < vzero)
      return Vector<unsigned char, 4>(lerp(value, vmin, vzero, Vector<double,4>(color_min), Vector<double, 4>(color_zero)));
    else
      return Vector<unsigned char, 4>(lerp(value, vzero, vmax, Vector<double,4>(color_zero), Vector<double, 4>(color_max)));
  }
};

// Transfer functions for visualization
template<class T> struct TransferDiscrete {
  typedef TriMesh::Color result_type;

  Hashtable<T,int> values;

  // create and set range to range of property
  TransferDiscrete(RawArray<const T> prop, int seed=0) {
    Array<T> data = prop.copy();
    new_<Random>(seed)->shuffle(data);
    for (int i=0;i<data.size();i++)
      if (!values.contains(data[i]))
        values.set(data[i],values.size());
  }

  TriMesh::Color operator()(T const &value) const {
    return to_byte_color(wheel_color(values.get_default(value,0),values.size()));
  }
};

// Transfer functions for visualization
template<class T> struct TransferMap {
  typedef TriMesh::Color result_type;

  int n_colors;
  typedef unordered_map<T,TriMesh::Color> ColorMap;
  ColorMap colors;

  TransferMap(const ColorMap& colors)
    : colors(colors) {}

  TriMesh::Color operator()(T const &value) const {
    typename ColorMap::const_iterator it = colors.find(value);
    if (it == colors.end())
      return TriMesh::Color(0, 0, 0, 255);
    else
      return it->second;
  }
};

}
#endif // GEODE_OPENMESH
