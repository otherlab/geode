//#####################################################################
// Class ParticleTree
//#####################################################################
#pragma once

#include <other/core/geometry/forward.h>
#include <other/core/geometry/BoxTree.h>
#include <other/core/math/constants.h>

namespace other{

template<class TV> class ParticleTree : public BoxTree<TV>
{
  typedef typename TV::Scalar T;
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef BoxTree<TV> Base;
  using Base::leaves;using Base::prims;using Base::boxes;using Base::update_nonleaf_boxes;
  using Base::nodes;

  const Array<const TV> X;

protected:
  OTHER_CORE_EXPORT ParticleTree(Array<const TV> X, int leaf_size);
public:
  ~ParticleTree();

  OTHER_CORE_EXPORT void update(); // Call whenever X changes
  OTHER_CORE_EXPORT Array<int> remove_duplicates(T tolerance) const; // Returns map from point to component index

  template<class Shape>
  OTHER_CORE_EXPORT void intersection(const Shape& box, Array<int>& hits) const;

  OTHER_CORE_EXPORT TV closest_point(TV point, int& index, T max_distance=inf) const; // simplex=-1 if nothing is found
  OTHER_CORE_EXPORT TV closest_point(TV point, T max_distance=inf) const; // return value is infinity if nothing is found
};

}
