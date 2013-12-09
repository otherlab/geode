//#####################################################################
// Class DiagonalMatrix2x2
//#####################################################################
#pragma once

#include <geode/vector/Matrix.h>
#include <geode/vector/Vector2d.h>
#include <geode/math/minabs.h>
#include <geode/math/sign.h>
#include <geode/math/robust.h>
namespace geode {

using ::std::log;

template<class T> struct IsScalarBlock<DiagonalMatrix<T,2> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<DiagonalMatrix<T,2> >:public IsScalarVectorSpace<T>{};

template<class T>
class DiagonalMatrix<T,2>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=2};
    enum Workaround2 {n=2};

    T x00,x11;

    DiagonalMatrix()
        :x00(),x11()
    {
        BOOST_STATIC_ASSERT(sizeof(DiagonalMatrix)==2*sizeof(T));
    }

    template<class T2> explicit
    DiagonalMatrix(const DiagonalMatrix<T2,2>& matrix)
        :x00((T)matrix.x00),x11((T)matrix.x11)
    {}

    DiagonalMatrix(const T y00,const T y11)
        :x00(y00),x11(y11)
    {}

    explicit DiagonalMatrix(const Vector<T,2>& v)
        :x00(v.x),x11(v.y)
    {}

    void copy(const DiagonalMatrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(2,2);}

    int rows() const
    {return 2;}

    int columns() const
    {return 2;}

    T& operator()(const int i)
    {assert(unsigned(i)<2);return ((T*)this)[i];}

    const T& operator()(const int i) const
    {assert(unsigned(i)<2);return ((T*)this)[i];}

    T first() const
    {return x00;}

    T last() const
    {return x11;}

    bool operator==(const DiagonalMatrix& A) const
    {return x00==A.x00 && x11==A.x11;}

    bool operator!=(const DiagonalMatrix& A) const
    {return x00!=A.x00 || x11!=A.x11;}

    DiagonalMatrix operator-() const
    {return DiagonalMatrix(-x00,-x11);}

    DiagonalMatrix& operator+=(const DiagonalMatrix& A)
    {x00+=A.x00;x11+=A.x11;return *this;}

    DiagonalMatrix& operator+=(const T& a)
    {x00+=a;x11+=a;return *this;}

    DiagonalMatrix& operator-=(const DiagonalMatrix& A)
    {x00-=A.x00;x11-=A.x11;return *this;}

    DiagonalMatrix& operator-=(const T& a)
    {x00-=a;x11-=a;return *this;}

    DiagonalMatrix& operator*=(const T a)
    {x00*=a;x11*=a;return *this;}

    DiagonalMatrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;x00*=s;x11*=s;return *this;}

    DiagonalMatrix operator+(const DiagonalMatrix& A) const
    {return DiagonalMatrix(x00+A.x00,x11+A.x11);}

    Matrix<T,2> operator+(const Matrix<T,2>& A) const
    {return Matrix<T,2>(x00+A.x[0],A.x[1],A.x[2],x11+A.x[3]);}

    Matrix<T,2> operator-(const Matrix<T,2>& A) const
    {return Matrix<T,2>(x00-A.x[0],-A.x[1],-A.x[2],x11-A.x[3]);}

    DiagonalMatrix operator+(const T a) const
    {return DiagonalMatrix(x00+a,x11+a);}

    DiagonalMatrix operator-(const DiagonalMatrix& A) const
    {return DiagonalMatrix(x00-A.x00,x11-A.x11);}

    DiagonalMatrix operator-(const T a) const
    {return DiagonalMatrix(x00-a,x11-a);}

    DiagonalMatrix operator*(const T a) const
    {return DiagonalMatrix(a*x00,a*x11);}

    DiagonalMatrix operator/(const T a) const
    {assert(a!=0);return *this*(1/a);}

    Vector<T,2> operator*(const Vector<T,2>& v) const
    {return Vector<T,2>(x00*v.x,x11*v.y);}

    DiagonalMatrix operator*(const DiagonalMatrix& A) const
    {return DiagonalMatrix(x00*A.x00,x11*A.x11);}

    DiagonalMatrix& operator*=(const DiagonalMatrix& A)
    {return *this=*this*A;}

    DiagonalMatrix operator/(const DiagonalMatrix& A) const
    {return DiagonalMatrix(x00/A.x00,x11/A.x11);}

    T determinant() const
    {return x00*x11;}

    DiagonalMatrix inverse() const
    {assert(x00!=0 && x11!=0);return DiagonalMatrix(1/x00,1/x11);}

    Vector<T,2> solve_linear_system(const Vector<T,2>& v) const
    {assert(x00!=0 && x11!=0);return Vector<T,2>(v.x/x00,v.y/x11);}

    Vector<T,2> robust_solve_linear_system(const Vector<T,2>& v) const
    {return Vector<T,2>(robust_divide(v.x,x00),robust_divide(v.y,x11));}

    DiagonalMatrix transposed() const
    {return *this;}

    void transpose()
    {}

    DiagonalMatrix cofactor_matrix() const
    {return DiagonalMatrix(x11,x00);}

    T trace() const
    {return x00+x11;}

    T dilational() const
    {return (T).5*trace();}

    T min() const
    {return geode::min(x00,x11);}

    T max() const
    {return geode::max(x00,x11);}

    T minabs() const
    {return geode::minabs(x00,x11);}

    T maxabs() const
    {return geode::maxabs(x00,x11);}

    T inner_product(const Vector<T,2>& a,const Vector<T,2>& b) const // inner product with respect to this matrix
    {return a.x*x00*b.x+a.y*x11*b.y;}

    T inverse_inner_product(const Vector<T,2>& a,const Vector<T,2>& b) const // inner product with respect to the inverse of this matrix
    {assert(x00!=0 && x11!=0);return a.x/x00*b.x+a.y/x11*b.y;}

    T sqr_frobenius_norm() const
    {return sqr(x00)+sqr(x11);}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    bool positive_definite() const
    {return x00>0 && x11>0;}

    bool positive_semidefinite() const
    {return x00>=0 && x11>=0;}

    DiagonalMatrix positive_definite_part() const
    {return clamp_min(0);}

    DiagonalMatrix sqrt() const
    {return DiagonalMatrix(geode::sqrt(x00),geode::sqrt(x11));}

    DiagonalMatrix clamp_min(const T a) const
    {return DiagonalMatrix(geode::clamp_min(x00,a),geode::clamp_min(x11,a));}

    DiagonalMatrix clamp_max(const T a) const
    {return DiagonalMatrix(geode::clamp_max(x00,a),geode::clamp_max(x11,a));}

    DiagonalMatrix abs() const
    {return DiagonalMatrix(geode::abs(x00),geode::abs(x11));}

    DiagonalMatrix sign() const
    {return DiagonalMatrix(geode::sign(x00),geode::sign(x11));}

    static DiagonalMatrix identity_matrix()
    {return DiagonalMatrix(1,1);}

    Vector<T,2> to_vector() const
    {return Vector<T,2>(x00,x11);}

    template<class TMatrix>
    typename Product<DiagonalMatrix,TMatrix>::type transpose_times(const TMatrix& M) const
    {return *this*M;}

    template<class TMatrix>
    typename ProductTranspose<DiagonalMatrix,TMatrix>::type
    times_transpose(const TMatrix& B) const
    {return (B**this).transposed();}

    DiagonalMatrix times_transpose(const DiagonalMatrix& M) const
    {return *this*M;}

//#####################################################################
    Matrix<T,2> times_transpose(const Matrix<T,2>& A) const;
    Matrix<T,2,3> times_transpose(const Matrix<T,3,2>& A) const;
//#####################################################################
};

template<class T>
inline DiagonalMatrix<T,2> operator*(const T a,const DiagonalMatrix<T,2>& A)
{return A*a;}

template<class T>
inline DiagonalMatrix<T,2> operator+(const T a,const DiagonalMatrix<T,2>& A)
{return A+a;}

template<class T>
inline DiagonalMatrix<T,2> operator-(const T a,const DiagonalMatrix<T,2>& A)
{return -A+a;}

template<class T>
inline Matrix<T,2> operator+(const Matrix<T,2>& A,const DiagonalMatrix<T,2>& B)
{return B+A;}

template<class T>
inline Matrix<T,2> operator-(const Matrix<T,2>& A,const DiagonalMatrix<T,2>& B)
{return -B+A;}

template<class T>
inline std::ostream& operator<<(std::ostream& output_stream,const DiagonalMatrix<T,2>& A)
{return output_stream<<A.x00<<" 0\n0 "<<A.x11<<" 0\n";}

template<class T>
inline DiagonalMatrix<T,2> log(const DiagonalMatrix<T,2>& A)
{return DiagonalMatrix<T,2>(log(A.x00),log(A.x11));}

template<class T>
inline DiagonalMatrix<T,2> exp(const DiagonalMatrix<T,2>& A)
{return DiagonalMatrix<T,2>(exp(A.x00),exp(A.x11));}

template<class T> inline Matrix<T,2,3> DiagonalMatrix<T,2>::
times_transpose(const Matrix<T,3,2>& A) const
{
    return Matrix<T,2,3>::column_major(x00*A(0,0),x11*A(0,1),x00*A(1,0),x11*A(1,1),x00*A(2,0),x11*A(2,1));
}

template<class T> inline Matrix<T,2> DiagonalMatrix<T,2>::
times_transpose(const Matrix<T,2>& A) const
{
    return Matrix<T,2>(x00*A(0,0),x11*A(0,1),x00*A(1,0),x11*A(1,1));
}

template<class T> inline T inner_product(const DiagonalMatrix<T,2>& A,const DiagonalMatrix<T,2>& B) {
  return A.x00*B.x00+A.x11*B.x11;
}

template<class T> inline T
inner_product_conjugate(const DiagonalMatrix<T,2>& A,const Matrix<T,2>& Q,const DiagonalMatrix<T,2> B) {
  Matrix<T,2> BQ=B*Q.transposed();
  return A.x00*(Q.x[0]*BQ.x[0]+Q.x[2]*BQ.x[1])+A.x11*(Q.x[1]*BQ.x[2]+Q.x[3]*BQ.x[3]);
}

}
