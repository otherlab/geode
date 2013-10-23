//#####################################################################
// Class TriangleSubdivision
//#####################################################################
//
// Topological and linear subdivision for triangle meshes.
//
//#####################################################################
#pragma once

#include <geode/mesh/TriangleMesh.h>
#include <geode/python/Object.h>
#include <geode/python/Ptr.h>
#include <geode/python/Ref.h>
#include <geode/vector/Vector.h>
namespace geode {

class TriangleSubdivision : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;

  typedef real T;

  Ref<const TriangleMesh> coarse_mesh;
  Ref<TriangleMesh> fine_mesh;
  Array<const int> corners; // Change only before subdivision functions are called
protected:
  mutable Ptr<SparseMatrix> loop_matrix_;

  GEODE_CORE_EXPORT TriangleSubdivision(const TriangleMesh& coarse_mesh);
public:
  ~TriangleSubdivision();

  template<class TV,int d> Array<typename boost::remove_const<TV>::type,d> linear_subdivide(const Array<TV,d>& X) const {
    return linear_subdivide(RawArray<typename boost::add_const<TV>::type,d>(X));
  }

  template<class TV,int d> Array<typename boost::remove_const<TV>::type,d> loop_subdivide(const Array<TV,d>& X) const {
    return loop_subdivide(RawArray<typename boost::add_const<TV>::type,d>(X));
  }

  template<class TV> GEODE_CORE_EXPORT Array<TV> linear_subdivide(RawArray<const TV> X) const;
  template<class TV> GEODE_CORE_EXPORT Array<TV> loop_subdivide(RawArray<const TV> X) const;
  GEODE_CORE_EXPORT Array<T,2> linear_subdivide(RawArray<const T,2> X) const;
  NdArray<T> linear_subdivide_python(NdArray<const T> X) const;
  NdArray<T> loop_subdivide_python(NdArray<const T> X) const;
  Ref<SparseMatrix> loop_matrix() const;
};

}
