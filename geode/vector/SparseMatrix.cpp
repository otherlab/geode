//#####################################################################
// Class SparseMatrix
//#####################################################################
#include <geode/vector/SparseMatrix.h>
#include <geode/array/NdArray.h>
#include <geode/array/ProjectedArray.h>
#include <geode/array/view.h>
#include <geode/vector/Vector.h>
#include <geode/structure/Hashtable.h>
#include <geode/utility/Log.h>
#include <geode/utility/const_cast.h>
namespace geode {

typedef real T;

static void sort_rows(SparseMatrix& self) {
  // Use insertion sort since we expect rows to be small
  for (int i=0;i<self.rows();i++) {
    RawArray<int> J = self.J[i].const_cast_();
    RawArray<T> A = self.A[i];
    for (int a=1;a<J.size();a++) {
      const int ja = J[a];
      const T aa = A[a];
      int b = a-1;
      for (;b>=0 && J[b]>ja;b--) {
        J[b+1] = J[b];
        A[b+1] = A[b];
      }
      J[b+1] = ja;
      A[b+1] = aa;
    }
  }
}

SparseMatrix::SparseMatrix(Nested<int> J, Array<T> A)
  : J(J), A(Nested<T>::reshape_like(A,J)), cholesky(false) {
  GEODE_ASSERT(!J.flat.size() || J.flat.min()>=0);
  columns_ = J.size()?J.flat.max()+1:0;

  // Run an insertion sort on each row
  sort_rows(*this);
}

SparseMatrix::SparseMatrix(const Hashtable<Vector<int,2>,T>& entries, const Vector<int,2>& sizes)
  : cholesky(false) {
  // Count rows and columns
  int rows = 0;
  for (auto& k : entries)
    rows = max(rows,k.x[0]+1);
  if (sizes.x>=0) {
    GEODE_ASSERT(rows<=sizes.x);
    rows = sizes.x;
  }

  // Compute row lengths
  Array<int> lengths(rows);
  for (auto& k : entries)
    lengths[k.x[0]]++;

  // Compute offsets
  const_cast_(J) = Nested<int>(lengths);
  const_cast_(A) = Nested<T>::zeros_like(J);

  // Fill in entries
  for (auto& k : entries) {
    int i,j;k.x.get(i,j);
    const int index = J.offsets[i]+--lengths[i];
    const_cast_(J.flat[index]) = j;
    A.flat[index] = k.y;
  }

  // Finish up
  columns_ = J.flat.size()?J.flat.max()+1:0;
  if (sizes.y>=0) {
    GEODE_ASSERT(columns_<=sizes.y);
    columns_ = sizes.y;
  }
  sort_rows(*this);
}

SparseMatrix::SparseMatrix(Nested<const int> J, Nested<T> A, Array<const int> diagonal_index, const bool cholesky, Private)
  : J(J), A(A), cholesky(cholesky), diagonal_index(diagonal_index) {}

SparseMatrix::~SparseMatrix() {}

int SparseMatrix::
find_entry(const int i,const int j) const
{
    if(i<0 || i>=rows()) return -1;
    return J[i].find(j);
}

bool SparseMatrix::
contains_entry(const int i,const int j) const
{
    if(i<0 || i>=rows()) return false;
    return J[i].contains(j);
}

T SparseMatrix::
operator()(const int i,const int j) const
{
    int index = find_entry(i,j);
    if(index>=0) return A(i,index);
    else throw IndexError("invalid sparse matrix index");
}

template<class TV> void SparseMatrix::
multiply_helper(RawArray<const TV> x,RawArray<TV> result) const
{
    const int rows = this->rows();
    GEODE_ASSERT(columns()<=x.size() && rows<=result.size());
    RawArray<const int> offsets = J.offsets;
    RawArray<const int> J_flat = J.flat;
    RawArray<const T> A_flat = A.flat;
    for(int i=0;i<rows;i++){
        int end=offsets[i+1];TV sum=TV();
        for(int index=offsets[i];index<end;index++) sum+=A_flat[index]*x[J_flat[index]];
        result[i]=sum;}
    result.slice(rows,result.size()).zero();
}

template void SparseMatrix::multiply_helper(RawArray<const T>,RawArray<T>) const;
template void SparseMatrix::multiply_helper(RawArray<const Vector<T,2> >,RawArray<Vector<T,2> >) const;
template void SparseMatrix::multiply_helper(RawArray<const Vector<T,3> >,RawArray<Vector<T,3> >) const;

void SparseMatrix::
multiply_python(NdArray<const T> x,NdArray<T> result) const {
  GEODE_ASSERT(x.shape==result.shape);
  if(x.rank()==1)
    multiply_helper(RawArray<const T>(x),RawArray<T>(result));
  else if(x.rank()==2) {
    switch(x.shape[1]) {
      case 1: return multiply_helper(RawArray<const T>(x),RawArray<T>(result));
      case 2: return multiply_helper(vector_view<2>(x.flat),vector_view<2>(result.flat));
      case 3: return multiply_helper(vector_view<3>(x.flat),vector_view<3>(result.flat));
      default: GEODE_NOT_IMPLEMENTED("general size vectors");
    }
  } else
    GEODE_FATAL_ERROR("expected rank 1 or 2");
}

bool SparseMatrix::
symmetric(const T tolerance) const
{
    if(rows()!=columns()) return false;
    for(int i=0;i<rows();i++) for(int a=0;a<J.size(i);a++) {
        int j = J(i,a);
        int other = find_entry(j,i);
        if(other<0 || abs(A(i,a)-A(j,other))>tolerance) return false;}
    return true;
}

bool SparseMatrix::
positive_diagonal_and_nonnegative_row_sum(const T tolerance) const
{
    bool result=true;
    for(int i=0;i<rows();i++){
        if(int d=find_entry(i,i)){
            if(A(i,d)<=0){Log::cout<<"diagonal entry "<<i<<" contains nonpositive element "<<A(i,d)<<std::endl;result=false;}}
        else{
            Log::cout<<"missing diagonal entry "<<i<<std::endl;result=false;}
        T sum = A[i].sum();
        if(sum<-tolerance){Log::cout<<"sum of row "<<i<<" is negative: "<<sum<<std::endl;result=false;}}
    return result;
}

void SparseMatrix::
solve_forward_substitution(RawArray<const T> b,RawArray<T> x) const
{
    GEODE_ASSERT(cholesky && rows()<=x.size() && rows()<=b.size());
    // The result of Incomplete_Cholesky_Factorization has unit diagonals in the lower triangle.
    for(int i=0;i<rows();i++){
        T sum=0;
        for(int index=J.offsets[i];index<diagonal_index[i];index++)
            sum+=A.flat[index]*x[J.flat[index]];
        x[i]=b[i]-sum;}
}

void SparseMatrix::
solve_backward_substitution(RawArray<const T> b,RawArray<T> x) const
{
    GEODE_ASSERT(cholesky && rows()<=x.size() && rows()<=b.size());
    // The result of Incomplete_Cholesky_Factorization has an inverted diagonal for the upper triangle.
    for(int i=rows()-1;i>=0;i--){
        T sum=0;
        for(int index=diagonal_index[i]+1;index<J.offsets[i+1];index++)
            sum+=A.flat[index]*x[J.flat[index]];
        x[i]=(b[i]-sum)*A.flat[diagonal_index[i]];}
}

void SparseMatrix::
initialize_diagonal_index() const
{
    if(diagonal_index.size()) return;
    Array<int> diagonal(min(rows(),columns()),uninit);
    for(int i=0;i<diagonal.size();i++){
        diagonal[i]=find_entry(i,i);
        GEODE_ASSERT(diagonal[i]>=0);
        diagonal[i]+=A.offsets[i];}
    diagonal_index=diagonal;
}

// actually an LU saving square roots, with an inverted diagonal saving divides
Ref<SparseMatrix > SparseMatrix::
incomplete_cholesky_factorization(const T modified_coefficient,const T zero_tolerance) const
{
    GEODE_ASSERT(rows()==columns());
    initialize_diagonal_index();
    Array<T> C(A.flat.copy());
    for(int i=0;i<rows();i++){ // for each row
        int row_diagonal_index=diagonal_index[i],row_end=J.offsets[i+1];T sum=0;
        for(int k_bar=J.offsets[i];k_bar<row_diagonal_index;k_bar++){ // for all the entries before the diagonal element
            int k=J.flat[k_bar];int row2_diagonal_index=diagonal_index[k],row2_end=J.offsets[k+1];
            C[k_bar]*=C[row2_diagonal_index]; // divide by the diagonal element (which has already been inverted)
            int j_bar=k_bar+1; // start with the next element in the row, when subtracting the dot product
            for(int i_bar=row2_diagonal_index+1;i_bar<row2_end;i_bar++){ // run through the rest of the elements in the row2
                int i=J.flat[i_bar];T dot_product_term=C[k_bar]*C[i_bar];
                while(j_bar<row_end-1 && J.flat[j_bar]<i) j_bar++; // gets j_bar such that j_bar>=i
                if(J.flat[j_bar]==i) C[j_bar]-=dot_product_term;
                else sum+=dot_product_term;}}
        T denominator=C[row_diagonal_index]-modified_coefficient*sum;
        if(i==rows()-1 && denominator<=zero_tolerance) denominator=zero_tolerance; // ensure last diagonal element is not zero
        C[row_diagonal_index]=1/denominator;} // finally, store the diagonal element in inverted form

    return new_<SparseMatrix>(J,Nested<T>::reshape_like(C,J),diagonal_index,true,Private());
}

void SparseMatrix::
gauss_seidel_solve(RawArray<T> x,RawArray<const T> b,const T tolerance,const int max_iterations) const
{
    GEODE_ASSERT(rows()==columns() && x.size()==rows() && b.size()==rows());
    const T sqr_tolerance=sqr(tolerance);
    for(int iteration=0;iteration<max_iterations;iteration++){
        T sqr_residual=0;
        for(int i=0;i<rows();i++){
            T rho=0;T diagonal_entry=0;
            for(int index=J.offsets[i];index<J.offsets[i+1];index++){
                if(J.flat[index]==i) diagonal_entry=A.flat[index];
                else rho+=A.flat[index]*x[J.flat[index]];}
            T new_x=x[i]=(b[i]-rho)/diagonal_entry;
            sqr_residual+=sqr(new_x-x[i]);
            x[i]=new_x;}
        if(sqr_residual <= sqr_tolerance) break;}
}

std::ostream&
operator<<(std::ostream& output,const SparseMatrix& A)
{
    for(int i=0;i<A.rows();i++){
        for(int j=0;j<A.columns();j++) output<<(A.contains_entry(i,j)?A(i,j):0)<<' ';
        output<<std::endl;}
    return output;
}

}
