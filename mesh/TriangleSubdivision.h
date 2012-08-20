//#####################################################################
// Class TriangleSubdivision
//#####################################################################
//
// Topological and linear subdivision for triangle meshes.
//
//#####################################################################
#pragma once

#include <other/core/mesh/TriangleMesh.h>
#include <other/core/python/Object.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/Ref.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/vector/Vector.h>
namespace other {

class TriangleSubdivision:public Object
{
public:
    OTHER_DECLARE_TYPE
    typedef Object Base;

    typedef real T;

    Ref<const TriangleMesh> coarse_mesh;
    Ref<TriangleMesh> fine_mesh;
    Hashtable<int> corners;
protected:
    mutable Ptr<SparseMatrix> loop_matrix_;

    TriangleSubdivision(const TriangleMesh& coarse_mesh) OTHER_EXPORT;
public:
    ~TriangleSubdivision();

    template<class TV,int d> Array<typename boost::remove_const<TV>::type,d> linear_subdivide(const Array<TV,d>& X) const {
      return linear_subdivide(RawArray<typename boost::add_const<TV>::type,d>(X));
    }

    template<class TV,int d> Array<typename boost::remove_const<TV>::type,d> loop_subdivide(const Array<TV,d>& X) const {
      return loop_subdivide(RawArray<typename boost::add_const<TV>::type,d>(X));
    }

    template<class TV> OTHER_EXPORT Array<TV> linear_subdivide(RawArray<const TV> X) const;
    template<class TV> Array<TV> loop_subdivide(RawArray<const TV> X) const OTHER_EXPORT;
    Array<T,2> linear_subdivide(RawArray<const T,2> X) const OTHER_EXPORT;
    NdArray<T> linear_subdivide_python(NdArray<const T> X) const;
    NdArray<T> loop_subdivide_python(NdArray<const T> X) const;
    Ref<SparseMatrix> loop_matrix() const;
};
}
