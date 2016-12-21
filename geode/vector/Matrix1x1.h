//#####################################################################
// Class Matrix<T,1>
//#####################################################################
#pragma once

#include <geode/vector/ArithmeticPolicy.h>
#include <geode/vector/Vector1d.h>
#include <geode/math/robust.h>
namespace geode {

using std::log;
using std::exp;
using std::sqrt;

template<class T>
class Matrix<T,1>
{
    struct Unusable{};
public:
    typedef T Scalar;
    enum Workaround1 {m=1,n=1};
    static const bool is_const=false;

    T x00;

    Matrix()
        :x00()
    {}

    template<class T2>
    explicit Matrix(const Matrix<T2,1>& matrix)
        :x00(matrix.x00)
    {}

    explicit Matrix(RawArray<const T,2> A)
        :x00(A(0,0))
    {
        assert(A.m==1 && A.n==1);
    }

    explicit Matrix(const T x00)
        :x00(x00)
    {}

    explicit Matrix(const Vector<T,1>& v)
        :x00(v.x)
    {}

    Matrix& operator=(const Matrix& matrix)
    {x00=matrix.x00;return *this;}

    void copy(const Matrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(1,1);}

    int rows() const
    {return 1;}

    int columns() const
    {return 1;}

    T& operator()(const int i,const int j=1)
    {assert(i==0 && j==0);return x00;}

    const T& operator()(const int i,const int j=1) const
    {assert(i==0 && j==0);return x00;}

    bool valid_index(const int i,const int j) const
    {return i==0 && j==0;}

    const Vector<T,1> row(const int i) const
    {assert(i==0);return Vector<T,1>(x00);}

    const Vector<T,1> column(const int j) const
    {assert(j==0);return Vector<T,1>(x00);}

    void set_column(const int j,const Vector<T,1>& c)
    {assert(j==0);x00=c.x;}

    bool operator==(const Matrix& A) const
    {return x00==A.x00;}

    bool operator!=(const Matrix& A) const
    {return !(*this==A);}

    Vector<T,1> column_sum() const
    {return column(0);}

    Vector<T,1> column_magnitudes() const
    {return Vector<T,1>(column(0).magnitude());}

    Matrix inverse() const
    {assert(x00);return Matrix(1/x00);}

    Vector<T,1> solve_linear_system(const Vector<T,1>& b) const
    {return Vector<T,1>(b.x/x00);}

    Vector<T,1> robust_solve_linear_system(const Vector<T,1>& b) const
    {return Vector<T,1>(robust_divide(b.x,x00));}

    Matrix cofactor_matrix() const
    {return Matrix(1);}

    Matrix normal_equations_matrix() const // 1 mult
    {return Matrix(sqr(x00));}

    Matrix operator-() const
    {return Matrix(-x00);}

    Matrix operator+(const T a) const
    {return Matrix(x00+a);}

    Matrix operator+(const Matrix& A) const
    {return Matrix(x00+A.x00);}

    Matrix operator-(const T a) const
    {return Matrix(x00-a);}

    Matrix operator-(const Matrix& A) const
    {return Matrix(x00-A.x00);}

    Matrix operator*(const Matrix& A) const
    {return Matrix(x00*A.x00);}

    template<int p>
    Matrix<T,1,p> operator*(const Matrix<T,1,p>& A) const
    {return A*x00;}

    Matrix operator*(const T a) const
    {return Matrix(a*x00);}

    Matrix operator/(const T a) const
    {return Matrix(x00/a);}

    Vector<T,1> operator*(const Vector<T,1>& v) const
    {return Vector<T,1>(x00*v.x);}

    Matrix& operator+=(const Matrix& A)
    {x00+=A.x00;return *this;}

    Matrix& operator-=(const Matrix& A)
    {x00-=A.x00;return *this;}

    Matrix& operator+=(const T& a)
    {x00+=a;return *this;}

    Matrix& operator-=(const T& a)
    {x00-=a;return *this;}

    Matrix& operator*=(const T a)
    {x00*=a;return *this;}

    Matrix& operator*=(const Matrix& A)
    {x00*=A.x00;return *this;}

    Matrix& operator/=(const T a)
    {x00/=a;return *this;}

    Vector<T,1> transpose_times(const Vector<T,1>& v) const
    {return Vector<T,1>(x00*v.x);}

    template<class TMatrix>
    TMatrix transpose_times(const TMatrix& A) const
    {return *this*A;}

    template<class TMatrix>
    typename Transpose<TMatrix>::type times_transpose(const TMatrix& A) const
    {return *this*A.transposed();}

    void transpose()
    {}

    Matrix transposed() const
    {return *this;}

    T trace() const
    {return x00;}

    T determinant() const
    {return x00;}

    Matrix<T,1> fast_eigenvalues() const
    {return *this;}

    T max() const
    {return x00;}

    T min() const
    {return x00;}

    T frobenius_norm() const
    {return abs(x00);}

    T sqr_frobenius_norm() const
    {return sqr(x00);}

    SymmetricMatrix<T,2> conjugate_with_cross_product_matrix(const Vector<T,2>& v) const
    {return x00*SymmetricMatrix<T,2>(sqr(v.y),-v.x*v.y,sqr(v.x));}

    T inner_product(const Vector<T,1>& u,const Vector<T,1>& v) const
    {return x00*u.x*v.x;}

    T inverse_inner_product(const Vector<T,1>& u,const Vector<T,1>& v) const
    {return u.x*v.x/x00;}

    Matrix<T,1,2> times_cross_product_matrix(Vector<T,2> v) const
    {return x00*cross_product_matrix(v);}

    bool positive_definite() const
    {return x00>0;}

    bool positive_semidefinite() const
    {return x00>=0;}

    Matrix positive_definite_part() const
    {return clamp_min(0);}

    Matrix diagonal_part() const
    {return *this;}

    Matrix sqrt() const
    {return Matrix(sqrt(x00));}

    Matrix clamp_min(const T a) const
    {return Matrix(clamp_min(x00,a));}

    Matrix clamp_max(const T a) const
    {return Matrix(clamp_max(x00,a));}

    Matrix abs() const
    {return Matrix(abs(x00));}

    Matrix sign() const
    {return Matrix(sign(x00));}

    static Matrix identity_matrix()
    {return Matrix((T)1);}

    Vector<T,1> to_vector() const
    {return Vector<T,1>(x00);}

    void solve_eigenproblem(Matrix& D,Matrix& V) const
    {fast_solve_eigenproblem(D,V);}

    void fast_solve_eigenproblem(Matrix& D,Matrix& V) const
    {V.x00=1;D=*this;}

    void fast_singular_value_decomposition(Matrix& U,Matrix& D,Matrix& V) const
    {U.x00=V.x00=1;D=*this;}

//#####################################################################
};

template<class T>
inline Matrix<T,1> operator*(const T a,const Matrix<T,1>& A)
{return A*a;}

template<class T>
inline Matrix<T,1> operator+(const T a,const Matrix<T,1>& A)
{return A+a;}

template<class T>
inline Matrix<T,1> operator-(const T a,const Matrix<T,1>& A)
{return Matrix<T,1>(a-A.x00);}

template<class T>
inline Matrix<T,1> clamp(const Matrix<T,1>& x,const Matrix<T,1>& xmin,const Matrix<T,1>& xmax)
{return Matrix<T,1>(clamp(x.x00,xmin.x00,xmax.x00));}

template<class T>
inline Matrix<T,1> clamp_min(const Matrix<T,1>& x,const Matrix<T,1>& xmin)
{return Matrix<T,1>(clamp_min(x.x00,xmin.x00));}

template<class T>
inline Matrix<T,1> clamp_max(const Matrix<T,1>& x,const Matrix<T,1>& xmax)
{return Matrix<T,1>(clamp_max(x.x00,xmax.x00));}

template<class T>
inline Matrix<T,1> log(const Matrix<T,1>& A)
{return Matrix<T,1>(log(A.x00));}

template<class T>
inline Matrix<T,1> exp(const Matrix<T,1>& A)
{return Matrix<T,1>(exp(A.x00));}

template<class T> inline Matrix<T,1>
outer_product(const Vector<T,1>& u)
{return Matrix<T,1>(u.x*u.x);}

template<class T> inline Matrix<T,1>
outer_product(const Vector<T,1>& u,const Vector<T,1>& v)
{return Matrix<T,1>(u.x*v.x);}

template<class T> inline Matrix<T,1> conjugate(const Matrix<T,1>& A,const Matrix<T,1>& B) {
  return A*B*A;
}

template<class T>
inline std::istream& operator>>(std::istream& input,Matrix<T,1>& A)
{return input>>A.x00;}

}
