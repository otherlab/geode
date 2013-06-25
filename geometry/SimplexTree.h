//#####################################################################
// Class SimplexTree
//#####################################################################
#pragma once

#include <other/core/geometry/forward.h>
#include <other/core/geometry/BoxTree.h>
#include <other/core/mesh/SegmentMesh.h>
#include <other/core/mesh/TriangleMesh.h>
#include <other/core/math/constants.h>
#include <vector>
namespace other{

template<class TV,int d> class SimplexTree : public BoxTree<TV> {
  typedef typename TV::Scalar T;
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef BoxTree<TV> Base;
  typedef typename mpl::if_c<d==1,SegmentMesh,TriangleMesh>::type Mesh;
  typedef typename mpl::if_c<d==1,Segment<TV>,Triangle<TV>>::type Simplex;
  typedef typename mpl::if_c<d==1,T,Vector<T,3>>::type Weights;
  using Base::leaves;using Base::prims;using Base::boxes;using Base::update_nonleaf_boxes;
  using Base::bounding_box;using Base::nodes;

  const Ref<const Mesh> mesh;
  const Array<const TV> X;
  const Array<Simplex> simplices;

protected:
OTHER_CORE_EXPORT SimplexTree(const Mesh& mesh, Array<const TV> X, int leaf_size);
public:
  ~SimplexTree();

  OTHER_CORE_EXPORT void update(); // Call whenever X changes
  OTHER_CORE_EXPORT bool intersection(Ray<TV>& ray, const T thickness_over_two) const;
  OTHER_CORE_EXPORT Array<Ray<TV> > intersections(const Ray<TV>& ray, const T thickness_over_two) const;
  OTHER_CORE_EXPORT void intersection(const Sphere<TV>& sphere, Array<int>& hits) const;
  OTHER_CORE_EXPORT void intersections(const Plane<T>& plane, Array<Segment<TV>>& result) const;
  OTHER_CORE_EXPORT bool inside(TV point) const;
  OTHER_CORE_EXPORT bool inside_given_closest_point(TV point, int simplex, Weights weights) const;
  OTHER_CORE_EXPORT T distance(TV point, T max_distance=inf) const; // return value is infinity if nothing is found

  // Returns closest_point,simplex,weights.  If nothing is found, simplex = -1 and closet_point = inf.
  OTHER_CORE_EXPORT Tuple<TV,int,Weights> closest_point(const TV point, const T max_distance=inf) const;
};

}
