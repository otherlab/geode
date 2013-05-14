//#####################################################################
// Class SolidMatrix
//#####################################################################
#include <other/core/vector/SolidMatrix.h>
#include <other/core/array/sort.h>
#include <other/core/array/view.h>
#include <other/core/structure/Tuple.h>
#include <other/core/python/Class.h>
#include <other/core/vector/DiagonalMatrix.h>
#include <other/core/vector/SymmetricMatrix.h>
#include <other/core/geometry/Box.h>
#include <other/core/utility/const_cast.h>
namespace other{

typedef real T;

OTHER_DEFINE_TYPE(SolidMatrixStructure)
template<> OTHER_DEFINE_TYPE(SolidMatrixBase<Vector<T,2>>)
template<> OTHER_DEFINE_TYPE(SolidMatrixBase<Vector<T,3>>)
template<> OTHER_DEFINE_TYPE(SolidMatrix<Vector<T,2>>)
template<> OTHER_DEFINE_TYPE(SolidMatrix<Vector<T,3>>)
template<> OTHER_DEFINE_TYPE(SolidDiagonalMatrix<Vector<T,2>>)
template<> OTHER_DEFINE_TYPE(SolidDiagonalMatrix<Vector<T,3>>)

SolidMatrixStructure::
SolidMatrixStructure(int n)
  : n(n) {
  OTHER_ASSERT(n>=0);
}

SolidMatrixStructure::
SolidMatrixStructure(const SolidMatrixStructure& src)
  : n(src.n), sparse(src.sparse), outers(src.outers) {}

SolidMatrixStructure::
~SolidMatrixStructure()
{}

Ref<SolidMatrixStructure> SolidMatrixStructure::
copy() const {
  return new_<SolidMatrixStructure>(*this);
}

void SolidMatrixStructure::
add_entry(int i,int j) {
  OTHER_ASSERT(unsigned(i)<unsigned(n) && unsigned(j)<unsigned(n));
  if (i!=j)
    sparse.set(vec(i,j).sorted());
}

void SolidMatrixStructure::
add_outer(int m,Array<const int> nodes) {
  OTHER_ASSERT(m>=1);
  if (nodes.size())
    OTHER_ASSERT(0<=nodes.min() && nodes.max()<n);
  outers.push_back(tuple(m,nodes));
}

template<class TV> SolidMatrixBase<TV>::
SolidMatrixBase(int n)
  : n(n) {}

template<class TV> SolidMatrixBase<TV>::
~SolidMatrixBase() {}

template<class TV> SolidMatrix<TV>::
SolidMatrix(const SolidMatrixStructure& structure)
  : Base(structure.n), next_outer(0) {
  // Allocate sparse component
  Array<int> lengths(structure.n,false);
  lengths.fill(1); // account for diagonal
  for (auto& s : structure.sparse)
    lengths[s.x]++;
  Nested<int> sparse_j(lengths);
  for (int i=0;i<structure.n;i++)
    sparse_j(i,--lengths[i]) = i;
  for (auto& s : structure.sparse) {
    int i,j;s.get(i,j);
    sparse_j(i,--lengths[i]) = j;
  }
  const_cast_(sparse_A) = Nested<Matrix<T,d>>::zeros_like(sparse_j);
  for (int i=0;i<sparse_j.size();i++) {
    sort(sparse_j[i]);
    OTHER_ASSERT(sparse_j.size(i) && sparse_j(i,0)==i); // ensure diagonal exists
  }
  const_cast_(this->sparse_j) = sparse_j;

  // Allocate outers
  const_cast_(outers).resize(structure.outers.size());
  for (int o=0;o<(int)outers.size();o++) {
    OTHER_ASSERT(structure.outers[o].x==1);
    const_cast_(outers[o].x) = structure.outers[o].y;
    const_cast_(outers[o].z).resize(outers[o].x.size(),false);
  }
}

template<class TV> SolidMatrix<TV>::
SolidMatrix(const SolidMatrix& A)
  : Base(A.size())
  , sparse_j(A.sparse_j)
  , sparse_A(A.sparse_A.copy())
  , next_outer(A.next_outer) {
  for (const auto& outer : A.outers)
    const_cast_(outers).push_back(tuple(outer.x,outer.y,outer.z.copy()));
}

template<class TV> SolidMatrix<TV>::
~SolidMatrix()
{}

template<class TV> Ref<SolidMatrix<TV>> SolidMatrix<TV>::
copy() const {
  return new_<SolidMatrix>(*this);
}

template<class TV> bool SolidMatrix<TV>::
valid() const {
  return next_outer==(int)outers.size();
}

template<class TV> void SolidMatrix<TV>::
zero() {
  sparse_A.flat.zero();
  next_outer = 0;
}

template<class TV> inline int SolidMatrix<TV>::
find_entry(int i,int j) const {
  assert(i<=j);
  RawArray<const int> row_j = sparse_j[i];
  for (int k=0;k<row_j.size();k++)
    if (row_j[k]==j)
      return k;
  throw KeyError(format("SolidMatrix::find_entry: index (%d,%d) doesn't exist",i,j));
}

template<class TV> void SolidMatrix<TV>::
add_entry(int i,int j,const Matrix<T,d>& a) {
  if (i<=j)
    sparse_A(i,find_entry(i,j)) += a;
  else
    sparse_A(j,find_entry(j,i)) += a.transposed();
}

template<class TV> void SolidMatrix<TV>::
add_entry(int i,int j,T a) {
  if (i>j) swap(i,j);
  sparse_A(i,find_entry(i,j)) += a;
}

template<class TV> void SolidMatrix<TV>::
add_outer(T B,RawArray<const TV> U) {
  int o = next_outer++;
  OTHER_ASSERT(o<(int)outers.size());
  OTHER_ASSERT(outers[o].x.size()==U.size());
  const_cast_(outers[o].y) = B;
  if (B)
    outers[o].z.copy(U);
}

template<class TV> void SolidMatrix<TV>::
scale(T s) {
  OTHER_ASSERT(valid());
  if (s==1)
    return;
  else if (s==-1) {
    sparse_A.flat.negate();
    for (int o=0;o<(int)outers.size();o++)
        const_cast_(outers[o].y) = -outers[o].y;
  } else {
    sparse_A.flat *= s;
    for (int o=0;o<(int)outers.size();o++)
        const_cast_(outers[o].y) *= s;
  }
}

template<class TV> void SolidMatrix<TV>::
add_scalar(T s) {
  OTHER_ASSERT(valid());
  if (s)
    for(int i=0;i<sparse_j.size();i++)
      sparse_A(i,0) += s;
}

template<class TV> void SolidMatrix<TV>::
add_diagonal_scalars(RawArray<const T> s) {
  OTHER_ASSERT(valid());
  OTHER_ASSERT(s.size()==sparse_j.size());
  for (int i=0;i<sparse_j.size();i++)
    sparse_A(i,0) += s[i];
}

template<class TV> void SolidMatrix<TV>::
add_partial_scalar(RawArray<const int> nodes, T s) {
  OTHER_ASSERT(valid());
  const int n = sparse_j.size();
  if (s)
    for (int i : nodes) {
      OTHER_ASSERT((unsigned)i<=(unsigned)n);
      sparse_A(i,0) += s;
    }
}

template<class TV> typename SolidMatrix<TV>::TMatrix SolidMatrix<TV>::
entry(int i,int j) const {
  OTHER_ASSERT(valid() && !outers.size());
  return i<=j?sparse_A(i,find_entry(i,j))
             :sparse_A(j,find_entry(j,i)).transposed();
}

template<class TV> Tuple<Array<int>,Array<int>,Array<typename TV::Scalar>> SolidMatrix<TV>::
entries() const {
  OTHER_ASSERT(valid() && !outers.size());
  Array<int> I,J;
  Array<T> C;
  for (int i=0;i<sparse_j.size();i++) for(int k=0;k<sparse_j.size(i);k++){
    int j = sparse_j(i,k);
    const Matrix<T,d>& A = sparse_A(i,k);
    for (int ii=0;ii<d;ii++) for(int jj=0;jj<d;jj++){
      I.append(d*i+ii);
      J.append(d*j+jj);
      C.append(A(ii,jj));
    }
  }
  return tuple(J,I,C);
}

template<class TV> void SolidMatrix<TV>::
add_multiply_outers(RawArray<const TV> x, RawArray<TV> y) const {
  OTHER_ASSERT(valid() && x.size()==size() && y.size()==size());
  for (int o=0;o<(int)outers.size();o++) {
    RawArray<const int> nodes = outers[o].x;
    T B = outers[o].y;
    if (!B)
      continue;
    RawArray<const TV> U = outers[o].z;
    T sum = 0;
    for (int a=0;a<nodes.size();a++)
      sum += dot(U[a],x[nodes[a]]);
    sum *= B;
    for (int a=0;a<nodes.size();a++)
      y[nodes[a]] += sum*U[a];
  }
}

template<class TV> void SolidMatrix<TV>::
multiply(RawArray<const TV> x, RawArray<TV> y) const {
  y.zero();
  add_multiply_outers(x,y);
  for (int i=0;i<sparse_j.size();i++) {
    y[i] += assume_symmetric(sparse_A(i,0))*x[i];
    for (int k=1;k<sparse_j.size(i);k++) {
      int j = sparse_j(i,k);
      const Matrix<T,d>& A = sparse_A(i,k);
      y[i] += A*x[j];
      y[j] += A.transpose_times(x[i]);
    }
  }
}

template<class TV> typename TV::Scalar SolidMatrix<TV>::
inner_product(RawArray<const TV> x, RawArray<const TV> y) const {
  OTHER_ASSERT(valid() && x.size()==size() && y.size()==size());
  T sum = 0;
  for (int i=0;i<sparse_j.size();i++) {
    sum += dot(x[i],assume_symmetric(sparse_A(i,0))*y[i]);
    for (int k=1;k<sparse_j.size(i);k++){
      int j = sparse_j(i,k);
      const Matrix<T,d>& A = sparse_A(i,k);
      sum += dot(x[i],A*y[j])+dot(y[i],A*x[j]);
    }
  }
  for (int o=0;o<(int)outers.size();o++) {
    RawArray<const int> nodes = outers[o].x;
    T B = outers[o].y;
    if (!B)
      continue;
    RawArray<const TV> U = outers[o].z;
    T left = 0, right = 0;
    for (int a=0;a<nodes.size();a++) {
      left  += dot(U[a],x[nodes[a]]);
      right += dot(U[a],y[nodes[a]]);
    }
    sum += B*dot(left,right);
  }
  return sum;
}

template<class TV> Box<typename TV::Scalar> SolidMatrix<TV>::
diagonal_range() const {
  OTHER_ASSERT(!outers.size());
  Box<T> range = Box<T>::empty_box();
  for(int i=0;i<size();i++) {
    DiagonalMatrix<T,d> D = assume_symmetric(sparse_A(i,0)).diagonal_part();
    range.enlarge(Box<T>(D.min(),D.max()));
  }
  return range;
}

template<class TV> Array<typename TV::Scalar,2> SolidMatrix<TV>::
dense() const {
  OTHER_ASSERT(valid());
  const int n = sparse_j.size();
  Array<TMatrix,2> dense(n,n);
  for (int i=0;i<n;i++){
    dense(i,i) = sparse_A(i,0);
    for (int k=1;k<sparse_j.size(i);k++) {
      int j = sparse_j(i,k);
      const Matrix<T,d>& A = sparse_A(i,k);
      dense(i,j) = A;
      dense(j,i) = A.transposed();
    }
  }
  for (int o=0;o<(int)outers.size();o++) {
    RawArray<const int> nodes = outers[o].x;
    T B = outers[o].y;
    if (!B)
      continue;
    RawArray<const TV> U = outers[o].z;
    for (int a=0;a<nodes.size();a++) for(int b=0;b<nodes.size();b++)
      dense(nodes[a],nodes[b]) += outer_product(B*U[a],U[b]);
  }
  return scalar_view_own(dense.flat).reshape_own(3*n,3*n);
}

template<class TV> Ref<SolidDiagonalMatrix<TV>> SolidMatrix<TV>::
inverse_block_diagonal() const {
  OTHER_ASSERT(!outers.size());
  Ref<SolidDiagonalMatrix<TV>> diagonal = new_<SolidDiagonalMatrix<TV>>(size(),false);
  for(int i=0;i<size();i++)
    diagonal->A[i] = assume_symmetric(sparse_A(i,0)).inverse();
  return diagonal;
}

template<class TV> SolidDiagonalMatrix<TV>::
SolidDiagonalMatrix(int size,bool initialize)
  : Base(size), A(size,initialize) {}

template<class TV> SolidDiagonalMatrix<TV>::
~SolidDiagonalMatrix() {}

template<class TV> void SolidDiagonalMatrix<TV>::
multiply(RawArray<const TV> x,RawArray<TV> y) const {
  for (int i=0;i<A.size();i++)
    y[i] = A[i]*x[i];
}

template<class TV> typename TV::Scalar SolidDiagonalMatrix<TV>::
inner_product(RawArray<const TV> x,RawArray<const TV> y) const {
  T sum = 0;
  for (int i=0;i<A.size();i++)
    sum += dot(x[i],A[i]*y[i]);
  return sum;
}

template class SolidMatrixBase<Vector<T,2>>;
template class SolidMatrixBase<Vector<T,3>>;
template class SolidMatrix<Vector<T,2>>;
template class SolidMatrix<Vector<T,3>>;
template class SolidDiagonalMatrix<Vector<T,2>>;
template class SolidDiagonalMatrix<Vector<T,3>>;

}
using namespace other;

template<int d> static void wrap_helper() {
  {typedef SolidMatrixBase<Vector<T,d>> Self;
  Class<Self>(d==2?"SolidMatrixBase2d":"SolidMatrixBase3d")
    .OTHER_METHOD(multiply)
    ;}

  {typedef SolidMatrix<Vector<T,d>> Self;
  Class<Self>(d==2?"SolidMatrix2d":"SolidMatrix3d")
    .OTHER_INIT(const SolidMatrixStructure&)
    .OTHER_METHOD(copy)
    .OTHER_METHOD(size)
    .OTHER_METHOD(zero)
    .OTHER_METHOD(scale)
    .method("add_entry",static_cast<void(Self::*)(int,int,const Matrix<T,d>&)>(&Self::add_entry))
    .OTHER_METHOD(add_scalar)
    .OTHER_METHOD(add_diagonal_scalars)
    .OTHER_METHOD(add_partial_scalar)
    .OTHER_METHOD(add_outer)
    .OTHER_METHOD(entries)
    .OTHER_METHOD(inverse_block_diagonal)
    .OTHER_METHOD(inner_product)
    .OTHER_METHOD(diagonal_range)
    .OTHER_METHOD(dense)
    ;}

  {typedef SolidDiagonalMatrix<Vector<T,d>> Self;
  Class<Self>(d==2?"SolidDiagonalMatrix2d":"SolidDiagonalMatrix3d")
    .OTHER_METHOD(inner_product)
    ;}
}

void wrap_solid_matrix() {
  {typedef SolidMatrixStructure Self;
  Class<Self>("SolidMatrixStructure")
    .OTHER_INIT(int)
    .OTHER_METHOD(copy)
    .OTHER_METHOD(add_entry)
    .OTHER_METHOD(add_outer)
    ;}
  wrap_helper<2>();
  wrap_helper<3>();
}
