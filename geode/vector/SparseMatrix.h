//#####################################################################
// Class SparseMatrix
//#####################################################################
//
// A sparse matrix class using flat storage.
//
//#####################################################################
#pragma once

#include <geode/array/Nested.h>
#include <geode/python/Object.h>
#include <geode/vector/Vector.h>
#include <geode/structure/Hashtable.h>

namespace geode {

class SparseMatrix:public Object
{
public:
    GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
    typedef real T;

    const Nested<const int> J;
    const Nested<T> A;
private:
    int columns_;
    bool cholesky;
    mutable Array<const int> diagonal_index;
    struct Private{};

    GEODE_CORE_EXPORT SparseMatrix(Nested<int> J,Array<T> A); // entries in each row will be sorted
    GEODE_CORE_EXPORT SparseMatrix(const Hashtable<Vector<int,2>,T>& entries, const Vector<int,2>& sizes = (Vector<int,2>(-1,-1)));
    SparseMatrix(Nested<const int> J, Nested<T> A, Array<const int> diagonal_index, const bool cholesky, Private);
public:
    ~SparseMatrix();

    Vector<int,2> sizes() const
    {return Vector<int,2>(rows(),columns());}

    int rows() const
    {return J.size();}

    int columns() const
    {return columns_;}

    template<class TX,class TY> void multiply(const TX& x,const TY& result) const {
      return multiply_helper<typename TX::Element>(x,result);
    }

    int find_entry(const int i,const int j) const;
    bool contains_entry(const int i,const int j) const;
    T operator()(const int i,const int j) const;
    template<class TV> void multiply_helper(RawArray<const TV> x,RawArray<TV> result) const;
    void multiply_python(NdArray<const T> x,NdArray<T> result) const;
    bool symmetric(const T tolerance=1e-7) const;
    bool positive_diagonal_and_nonnegative_row_sum(const T tolerance=1e-7) const;
    void solve_forward_substitution(RawArray<const T> b,RawArray<T> x) const;
    void solve_backward_substitution(RawArray<const T> b,RawArray<T> x) const;
    Ref<SparseMatrix> incomplete_cholesky_factorization(const T modified_coefficient=.97,const T zero_tolerance=1e-8) const;
    void gauss_seidel_solve(RawArray<T> x,RawArray<const T> b,const T tolerance=1e-12,const int max_iterations=1000000) const;
private:
    void initialize_diagonal_index() const;
};

std::ostream& operator<<(std::ostream& output,const SparseMatrix& A);

}
