//#####################################################################
// Class Matrix<T,0>
//#####################################################################
#pragma once

#include <geode/vector/Vector.h>
#include <cfloat>
namespace geode {

template<class T>
class Matrix<T,0>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=0,n=0};
    static const bool is_const=false;

    Matrix()
    {}

    template<class T2>
    explicit Matrix(const Matrix<T2,0>& matrix)
    {}

    explicit Matrix(RawArray<const T,2> A)
    {
        assert(A.m==0 && A.n==0);
    }

    Matrix& operator=(const Matrix& matrix)
    {return *this;}

    void copy(const Matrix& A)
    {}

    Vector<int,2> sizes() const
    {return Vector<int,2>(0,0);}

    int rows() const
    {return 0;}

    int columns() const
    {return 0;}

    T& operator()(const int i,const int j=0)
    {GEODE_FATAL_ERROR();}

    const T& operator()(const int i,const int j=0) const
    {GEODE_FATAL_ERROR();}

    const Vector<T,0> column(const int i) const
    {GEODE_FATAL_ERROR();}

    bool valid_index(const int i,const int j) const
    {return false;}

    bool operator==(const Matrix& A) const
    {return true;}

    bool operator!=(const Matrix& A) const
    {return false;}

    Matrix transposed() const
    {return *this;}

    void transpose()
    {}

    Matrix inverse() const
    {return *this;}

    Vector<T,0> solve_linear_system(const Vector<T,0>& b) const
    {return b;}

    Vector<T,0> robust_solve_linear_system(const Vector<T,0>& b) const
    {return b;}

    Matrix cofactor_matrix() const
    {return *this;}

    Matrix normal_equations_matrix() const
    {return *this;}

    Matrix operator-() const
    {return *this;}

    Matrix operator+(const T a) const
    {return *this;}

    Matrix operator+(const Matrix& A) const
    {return *this;}

    Matrix operator*(const T a) const
    {return *this;}

    Matrix operator/(const T a) const
    {return *this;}

    Vector<T,0> operator*(const Vector<T,0>& v) const
    {return v;}

    Matrix operator*(const Matrix& A) const
    {return *this;}

    Matrix& operator+=(const T a)
    {return *this;}

    Matrix& operator+=(const Matrix& A)
    {return *this;}

    Matrix operator-(const T a) const
    {return *this;}

    Matrix operator-(const Matrix& A) const
    {return *this;}

    Matrix& operator-=(const Matrix& A)
    {return *this;}

    Matrix& operator-=(const T& a)
    {return *this;}

    Matrix& operator*=(const Matrix& A)
    {return *this;}

    Matrix& operator*=(const T a)
    {return *this;}

    Matrix& operator/=(const T a)
    {return *this;}

    template<class RW>
    void read(std::istream& input)
    {}

    template<class RW>
    void write(std::ostream& output) const
    {}

    T max() const
    {return -FLT_MAX;}

    T min() const
    {return FLT_MAX;}

    T frobenius_norm() const
    {return 0;}

    T sqr_frobenius_norm() const
    {return 0;}

    T inner_product(const Vector<T,0>& u,const Vector<T,0>& v) const
    {return 0;}

    bool positive_definite() const
    {return true;}

    bool positive_semidefinite() const
    {return true;}

    Matrix positive_definite_part() const
    {return *this;}

    Matrix diagonal_part() const
    {return *this;}

    Matrix sqrt() const
    {return *this;}

    Matrix clamp_min(const T a) const
    {return *this;}

    Matrix clamp_max(const T a) const
    {return *this;}

    Matrix abs() const
    {return *this;}

    Matrix sign() const
    {return *this;}

    static Matrix identity_matrix()
    {return Matrix();}

    T trace() const
    {return 0;}

    T determinant() const
    {return 1;}

    void fast_solve_eigenproblem(Matrix&,Matrix&) const
    {}

    void fast_singular_value_decomposition(Matrix&,Matrix&,Matrix&) const
    {}

//#####################################################################
};

template<class T>
inline Matrix<T,0> operator*(const T a,const Matrix<T,0>& A)
{return A;}

template<class T>
inline Matrix<T,0> operator+(const T a,const Matrix<T,0>& A)
{return A;}

template<class T>
inline Matrix<T,0> operator-(const T a,const Matrix<T,0>& A)
{return A;}

template<class T>
inline Matrix<T,0> log(const Matrix<T,0>& A)
{return Matrix<T,0>();}

template<class T>
inline Matrix<T,0> exp(const Matrix<T,0>& A)
{return Matrix<T,0>();}

template<class T> inline Matrix<T,0>
outer_product(const Vector<T,0>& v)
{return Matrix<T,0>();}

template<class T>
inline std::istream& operator>>(std::istream& input,Matrix<T,0>& A)
{return input;}
}
