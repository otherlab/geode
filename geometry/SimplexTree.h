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
  typedef real T;
public:
  OTHER_DECLARE_TYPE
  typedef BoxTree<TV> Base;
  typedef typename mpl::if_c<d==1,SegmentMesh,TriangleMesh>::type Mesh;
  typedef typename mpl::if_c<d==1,Segment<TV>,Triangle<TV>>::type Simplex;
  using Base::leaves;using Base::prims;using Base::boxes;using Base::update_nonleaf_boxes;
  using Base::bounding_box;using Base::nodes;

  const Ref<const Mesh> mesh;
  const Array<const TV> X;
  const Array<Simplex> simplices;

protected:
  SimplexTree(const Mesh& mesh, Array<const TV> X, int leaf_size) OTHER_EXPORT;
public:
  ~SimplexTree();

  void update() OTHER_EXPORT; // Call whenever X changes
  bool intersection(Ray<TV>& ray, T thickness_over_two) const OTHER_EXPORT;
  std::vector<Ray<TV> > intersections(Ray<TV>& ray,T thickness_over_two) const OTHER_EXPORT;
  void intersection(const Sphere<TV>& sphere, Array<int>& hits) const OTHER_EXPORT;
  bool inside(TV point) const OTHER_EXPORT;
  bool inside_given_closest_point(TV point, int simplex, Vector<T,d+1> weights) const OTHER_EXPORT;
  TV closest_point(TV point, int& simplex, Vector<T,d+1>& weights, T max_distance=inf) const OTHER_EXPORT; // simplex=-1 if nothing is found
  TV closest_point(TV point, T max_distance=inf) const OTHER_EXPORT; // return value is infinity if nothing is found
  T distance (TV point, T max_distance=inf) const OTHER_EXPORT; // return value is infinity if nothing is found
};

}
