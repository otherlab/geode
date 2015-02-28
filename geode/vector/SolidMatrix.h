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

#include <geode/array/Nested.h>
#include <geode/array/Array2d.h>
#include <geode/structure/Hashtable.h>
#include <geode/python/Object.h>
#include <geode/vector/Matrix.h>
#include <geode/vector/Vector.h>
#include <geode/structure/Triple.h>
namespace geode {

class SolidMatrixStructure : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)

private:
  template<class TV> friend class SolidMatrix;

  int n;
  Hashtable<Vector<int,2> > sparse;
  std::vector<Tuple<int,Array<const int> > > outers;

  GEODE_CORE_EXPORT SolidMatrixStructure(int size);
  GEODE_CORE_EXPORT SolidMatrixStructure(const SolidMatrixStructure& src);
public:
  ~SolidMatrixStructure();

  int size() const
  {return n;}

  // Swaps i,j if necessary to get i<=j
  GEODE_CORE_EXPORT void add_entry(int i,int j);

  GEODE_CORE_EXPORT void add_outer(int m,Array<const int> nodes);

  GEODE_CORE_EXPORT Ref<SolidMatrixStructure> copy() const ;
};

template<class TV> class SolidMatrixBase : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;
  typedef typename TV::Scalar T;

protected:
  const int n;

  GEODE_CORE_EXPORT SolidMatrixBase(int n);
public:
  GEODE_CORE_EXPORT ~SolidMatrixBase();

  int size() const {
    return n;
  }

  virtual void multiply(RawArray<const TV> x,RawArray<TV> y) const = 0;
};

template<class TV> class SolidMatrix : public SolidMatrixBase<TV> {
  typedef typename TV::Scalar T;
  enum {d=TV::m};
  typedef Matrix<T,d> TMatrix;
  struct Unusable {};
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef SolidMatrixBase<TV> Base;

  const Nested<const int> sparse_j;
  const Nested<TMatrix> sparse_A;
  const std::vector<Tuple<Array<const int>,T,Array<TV> > > outers; // restricted to m==1 for now
private:
  int next_outer;

  GEODE_CORE_EXPORT SolidMatrix(const SolidMatrixStructure& structure);
  GEODE_CORE_EXPORT SolidMatrix(const SolidMatrix& A);
public:
  ~SolidMatrix();

  GEODE_CORE_EXPORT Ref<SolidMatrix> copy() const ;

  GEODE_CORE_EXPORT bool valid() const ;

  // Reset sparse entries to zero and clear outers
  GEODE_CORE_EXPORT void zero();

  // Swaps i,j and transposes a if necessary to get i<=j
  GEODE_CORE_EXPORT void add_entry(int i,int j,const Matrix<T,d>& a);
  GEODE_CORE_EXPORT void add_entry(int i,int j,T a);

  void add_entry(int i,const SymmetricMatrix<T,d>& a)
  {sparse_A(i,0) += a;}

  void add_entry(int i,T a)
  {sparse_A(i,0) += a;}

  // Must be called in exactly the same order as on the structure
  GEODE_CORE_EXPORT void add_outer(T B,RawArray<const TV> U);

  GEODE_CORE_EXPORT void scale(T s);
  GEODE_CORE_EXPORT void add_scalar(T s);
  GEODE_CORE_EXPORT void add_diagonal_scalars(RawArray<const T> s);
  GEODE_CORE_EXPORT void add_partial_scalar(RawArray<const int> nodes, T s);

  GEODE_CORE_EXPORT TMatrix entry(int i,int j) const ;
  GEODE_CORE_EXPORT Tuple<Array<int>,Array<int>,Array<T> > entries() const ;
  GEODE_CORE_EXPORT void multiply(RawArray<const TV> x,RawArray<TV> y) const ;
  GEODE_CORE_EXPORT T inner_product(RawArray<const TV> x,RawArray<const TV> y) const ;
  Ref<SolidDiagonalMatrix<TV> > inverse_block_diagonal() const;
  GEODE_CORE_EXPORT Box<T> diagonal_range() const ;
  Array<T,2> dense() const;

  GEODE_CORE_EXPORT void add_multiply_outers(RawArray<const TV> x, RawArray<TV> y) const; // y += A_outers x
private:
  int find_entry(int i,int j) const;
};

template<class TV> class SolidDiagonalMatrix : public SolidMatrixBase<TV> {
  typedef typename TV::Scalar T;
  enum {d=TV::m};
  typedef SymmetricMatrix<T,d> SM;
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef SolidMatrixBase<TV> Base;

  const Array<SM> A;

protected:
  GEODE_CORE_EXPORT SolidDiagonalMatrix(const int size);
  GEODE_CORE_EXPORT SolidDiagonalMatrix(const int size, Uninit);
public:
  GEODE_CORE_EXPORT ~SolidDiagonalMatrix();

  GEODE_CORE_EXPORT void multiply(RawArray<const TV> x,RawArray<TV> y) const ;
  GEODE_CORE_EXPORT T inner_product(RawArray<const TV> x,RawArray<const TV> y) const ;
};

}
