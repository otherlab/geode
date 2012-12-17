//#####################################################################
// Header Vectors/Forward
//#####################################################################
#pragma once

#include <other/core/utility/config.h>
#include <other/core/utility/forward.h>
#include <boost/mpl/and.hpp>
namespace other {

namespace mpl=boost::mpl;

template<class T,int d> class OTHER_CORE_CLASS_EXPORT Vector;

struct Zero;
template<class T> class Quaternion;
template<class TV> class Rotation;
template<class T> class TetrahedralGroup;
template<class TV> class Frame;
template<class TV> class Twist;

template<class TVector,class TNew> struct Rebind;
template<class T,int d,class TNew> struct Rebind<Vector<T,d>,TNew>{typedef Vector<TNew,d> type;};

template<class T> struct IsVector:public mpl::false_{};
template<class T,int d> struct IsVector<Vector<T,d> >:public mpl::true_{};

template<class T> struct IsScalarBlock;
template<class T> struct IsScalarVectorSpace;
template<class T> struct is_packed_pod;
template<class T,int d> struct IsScalarBlock<Vector<T,d> >:public mpl::and_<mpl::bool_<(d>0)>,IsScalarBlock<T> >{};
template<class T,int d> struct IsScalarVectorSpace<Vector<T,d> >:public mpl::and_<mpl::bool_<(d>0)>,IsScalarVectorSpace<T> >{};
template<class T,int d> struct is_packed_pod<Vector<T,d> >:public mpl::and_<mpl::bool_<(d>0)>,is_packed_pod<T> >{};

template<class T,int m_,int n_=m_> class Matrix;
template<class T,int d> class DiagonalMatrix;
template<class T,int d> class SymmetricMatrix;
template<class T,int d> class UpperTriangularMatrix;
class SparseMatrix;

template<class T,int m,int n> struct IsScalarBlock<Matrix<T,m,n> >:public mpl::and_<mpl::bool_<(m>0 && n>0)>,IsScalarBlock<T> >{};
template<class T,int m,int n> struct IsScalarVectorSpace<Matrix<T,m,n> >:public mpl::and_<mpl::bool_<(m>0 && n>0)>,IsScalarVectorSpace<T> >{};
template<class T,int m,int n> struct is_packed_pod<Matrix<T,m,n> >:public mpl::and_<mpl::bool_<(m>0 && n>0)>,is_packed_pod<T> >{};

class SolidMatrixStructure;
template<class TV> class SolidMatrix;
template<class TV> class SolidDiagonalMatrix;

// Declare vector conversions.  See vector/convert.h for the matching OTHER_DEFINE_VECTOR_CONVERSIONS.
#ifdef OTHER_PYTHON
#define OTHER_DECLARE_VECTOR_CONVERSIONS(EXPORT,d,...) \
  EXPORT PyObject* to_python(const Vector<__VA_ARGS__,d>& v); \
  template<> struct FromPython<Vector<__VA_ARGS__,d>> { EXPORT static Vector<__VA_ARGS__,d> convert(PyObject* o); };
#else
#define OTHER_DECLARE_VECTOR_CONVERSIONS(...) // non-python stub
#endif

}
