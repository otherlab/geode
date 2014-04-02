//#####################################################################
// Class SimplexTree
//#####################################################################
#pragma once

#include <geode/utility/config.h>
#include <geode/geometry/forward.h>
#include <geode/geometry/BoxTree.h>
#include <geode/mesh/SegmentSoup.h>
#include <geode/mesh/TriangleSoup.h>
#include <geode/math/constants.h>
#include <vector>
namespace geode {

template<class TV,int d_> class SimplexTree : public BoxTree<TV> {
  typedef typename TV::Scalar T;
public:
  GEODE_NEW_FRIEND
  typedef BoxTree<TV> Base;
  static const int d = d_;
  typedef typename mpl::if_c<d==1,SegmentSoup,TriangleSoup>::type Mesh;
  typedef typename mpl::if_c<d==1,Segment<TV>,Triangle<TV>>::type Simplex;
  typedef typename mpl::if_c<d==1,T,Vector<T,3>>::type Weights;
  using Base::leaves;using Base::prims;using Base::boxes;using Base::update_nonleaf_boxes;
  using Base::bounding_box;using Base::nodes;

  const Ref<const Mesh> mesh;
  const Array<const TV> X;
  const Array<Simplex> simplices;

protected:
  GEODE_CORE_EXPORT SimplexTree(const Mesh& mesh, Array<const TV> X, int leaf_size);
  GEODE_CORE_EXPORT SimplexTree(const SimplexTree& other, Array<const TV> X); // Shares ownership for topology (mesh, tree structure, etc.) but not geometry (X,boxes,simplices)
public:
  ~SimplexTree();

  GEODE_CORE_EXPORT void update(); // Call whenever X changes
  GEODE_CORE_EXPORT bool intersection(Ray<TV>& ray, const T thickness_over_two) const;
  GEODE_CORE_EXPORT Array<Ray<TV> > intersections(const Ray<TV>& ray, const T thickness_over_two) const;
  GEODE_CORE_EXPORT void intersection(const Sphere<TV>& sphere, Array<int>& hits) const;
  GEODE_CORE_EXPORT void intersections(const Plane<T>& plane, Array<Segment<TV>>& result) const;
  GEODE_CORE_EXPORT bool inside(TV point) const;
  GEODE_CORE_EXPORT bool inside_given_closest_point(TV point, int simplex, Weights weights) const;
  GEODE_CORE_EXPORT T distance(TV point, T max_distance=inf) const; // return value is infinity if nothing is found

  // Returns closest_point,simplex,weights.  If nothing is found, simplex = -1 and closet_point = inf.
  GEODE_CORE_EXPORT Tuple<TV,int,Weights> closest_point(const TV point, const T max_distance=inf) const;
};

// For testing purposes
template<class T,int d> GEODE_CORE_EXPORT int ray_traversal_test(const SimplexTree<Vector<T,d>,d-1>& tree,
                                                                 const int rays, const T half_thickness);

}
