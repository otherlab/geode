//#####################################################################
// Class ParticleTree
//#####################################################################
#pragma once

#include <geode/geometry/forward.h>
#include <geode/geometry/BoxTree.h>
#include <geode/math/constants.h>

namespace geode {

template<class TV> class ParticleTree : public BoxTree<TV>
{
  typedef typename TV::Scalar T;
public:
  GEODE_NEW_FRIEND
  typedef BoxTree<TV> Base;
  using Base::leaves;using Base::prims;using Base::boxes;using Base::update_nonleaf_boxes;
  using Base::nodes;

  const Array<const TV> X;

protected:
  GEODE_CORE_EXPORT ParticleTree(Array<const TV> X, int leaf_size);
public:
  ~ParticleTree();

  GEODE_CORE_EXPORT void update(); // Call whenever X changes
  GEODE_CORE_EXPORT Array<int> remove_duplicates(T tolerance) const; // Returns map from point to component index

  template<class Shape>
  GEODE_CORE_EXPORT void intersection(const Shape& box, Array<int>& hits) const;

  GEODE_CORE_EXPORT TV closest_point(TV point, int& index, T max_distance=inf, int ignore = -1) const; // simplex=-1 if nothing is found
  GEODE_CORE_EXPORT TV closest_point(TV point, T max_distance=inf) const; // return value is infinity if nothing is found
  GEODE_CORE_EXPORT Tuple<TV,int> closest_point_py(TV point, T max_distance=inf) const;
};

}
