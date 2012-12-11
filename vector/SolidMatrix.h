//#####################################################################
// Class SolidMatrix
//#####################################################################
//
// A symmetric, mostly sparse, dxd block matrix intended for cloth simulation.
// Entry (i,j) is stored only if i <= j, since the other entry can be obtained by symmetry.
//
// In order to incorporate pressure forces and other outer product-like dense terms, the
// matrix is represented as
//
//     A = S + sum_i U_i B_i U_i^T
//
// where S is sparse and each U_i is tall and thin.
//
//#####################################################################
#pragma once

#include <other/core/array/NestedArray.h>
#include <other/core/array/Array2d.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/python/Object.h>
#include <other/core/vector/Matrix.h>
#include <other/core/vector/Vector.h>
#include <other/core/structure/Triple.h>
namespace other {

class SolidMatrixStructure:public Object
{
public:
  OTHER_DECLARE_TYPE

private:
  template<class TV> friend class SolidMatrix;

  int n;
  Hashtable<Vector<int,2> > sparse;
  std::vector<Tuple<int,Array<const int> > > outers;

  SolidMatrixStructure(int size) OTHER_CORE_EXPORT;
  SolidMatrixStructure(const SolidMatrixStructure& src) OTHER_CORE_EXPORT;
public:
  ~SolidMatrixStructure();

  int size() const
  {return n;}

  // Swaps i,j if necessary to get i<=j
  void add_entry(int i,int j) OTHER_CORE_EXPORT;

  void add_outer(int m,Array<const int> nodes) OTHER_CORE_EXPORT;

  Ref<SolidMatrixStructure> copy() const OTHER_CORE_EXPORT;
};

template<class TV>
class SolidMatrixBase:public Object
{
public:
  OTHER_DECLARE_TYPE
  typedef Object Base;
  typedef typename TV::Scalar T;

protected:
  const int n;

  SolidMatrixBase(int n) OTHER_CORE_EXPORT;
public:
  ~SolidMatrixBase() OTHER_CORE_EXPORT;

  int size() const {
    return n;
  }

  virtual void multiply(RawArray<const TV> x,RawArray<TV> y) const = 0;
};

template<class TV>
class SolidMatrix:public SolidMatrixBase<TV>
{
    typedef typename TV::Scalar T;
    enum {d=TV::m};
    typedef Matrix<T,d> TMatrix;
    struct Unusable {};
public:
    OTHER_DECLARE_TYPE
    typedef SolidMatrixBase<TV> Base;
    using Base::size;

    const NestedArray<const int> sparse_j;
    const NestedArray<TMatrix> sparse_A;
    const std::vector<Tuple<Array<const int>,T,Array<TV> > > outers; // restricted to m==1 for now
private:
    int next_outer;

    SolidMatrix(const SolidMatrixStructure& structure) OTHER_CORE_EXPORT;
    SolidMatrix(const SolidMatrix& A) OTHER_CORE_EXPORT;
public:
    ~SolidMatrix();

    Ref<SolidMatrix> copy() const OTHER_CORE_EXPORT;

    bool valid() const OTHER_CORE_EXPORT;

    // Reset sparse entries to zero and clear outers
    void zero() OTHER_CORE_EXPORT;

    // Swaps i,j and transposes a if necessary to get i<=j
    void add_entry(int i,int j,const Matrix<T,d>& a) OTHER_CORE_EXPORT;
    void add_entry(int i,int j,T a) OTHER_CORE_EXPORT;

    void add_entry(int i,const SymmetricMatrix<T,d>& a)
    {sparse_A(i,0) += a;}

    void add_entry(int i,T a)
    {sparse_A(i,0) += a;}

    // Must be called in exactly the same order as on the structure
    void add_outer(T B,RawArray<const TV> U) OTHER_CORE_EXPORT;

    void scale(T s) OTHER_CORE_EXPORT;
    void add_scalar(T s) OTHER_CORE_EXPORT;
    void add_diagonal_scalars(RawArray<const T> s) OTHER_CORE_EXPORT;
    void add_partial_scalar(RawArray<const int> nodes, T s) OTHER_CORE_EXPORT;

    TMatrix entry(int i,int j) const OTHER_CORE_EXPORT;
    Tuple<Array<int>,Array<int>,Array<T> > entries() const OTHER_CORE_EXPORT;
    void multiply(RawArray<const TV> x,RawArray<TV> y) const OTHER_CORE_EXPORT;
    T inner_product(RawArray<const TV> x,RawArray<const TV> y) const OTHER_CORE_EXPORT;
    Ref<SolidDiagonalMatrix<TV> > inverse_block_diagonal() const;
    Box<T> diagonal_range() const OTHER_CORE_EXPORT;
    Array<T,2> dense() const;
private:
    int find_entry(int i,int j) const;
};

template<class TV>
class SolidDiagonalMatrix:public SolidMatrixBase<TV> {
  typedef typename TV::Scalar T;
  enum {d=TV::m};
  typedef SymmetricMatrix<T,d> SM;
public:
  OTHER_DECLARE_TYPE
  typedef SolidMatrixBase<TV> Base;

  const Array<SM> A;

protected:
  SolidDiagonalMatrix(int size,bool initialize=false) OTHER_CORE_EXPORT;
public:
  ~SolidDiagonalMatrix() OTHER_CORE_EXPORT;

  void multiply(RawArray<const TV> x,RawArray<TV> y) const OTHER_CORE_EXPORT;
  T inner_product(RawArray<const TV> x,RawArray<const TV> y) const OTHER_CORE_EXPORT;
};

}
