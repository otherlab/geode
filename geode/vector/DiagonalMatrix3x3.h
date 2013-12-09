//#####################################################################
// Class DiagonalMatrix3x3
//#####################################################################
#pragma once

#include <geode/vector/Matrix.h>
#include <geode/vector/Vector3d.h>
#include <geode/math/min.h>
#include <geode/math/minabs.h>
namespace geode {

using ::std::log;
using ::std::exp;

template<class T> struct IsScalarBlock<DiagonalMatrix<T,3> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<DiagonalMatrix<T,3> >:public IsScalarVectorSpace<T>{};

template<class T>
class DiagonalMatrix<T,3>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=3};
    enum Workaround2 {n=3};

    T x00,x11,x22;

    DiagonalMatrix()
        :x00(),x11(),x22()
    {
        BOOST_STATIC_ASSERT(sizeof(DiagonalMatrix)==3*sizeof(T));
    }

    template<class T2> explicit
    DiagonalMatrix(const DiagonalMatrix<T2,3>& matrix)
        :x00((T)matrix.x00),x11((T)matrix.x11),x22((T)matrix.x22)
    {}

    DiagonalMatrix(const T y00,const T y11,const T y22)
        :x00(y00),x11(y11),x22(y22)
    {}

    explicit DiagonalMatrix(const Vector<T,3>& v)
        :x00(v.x),x11(v.y),x22(v.z)
    {}

    void copy(const DiagonalMatrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(3,3);}

    int rows() const
    {return 3;}

    int columns() const
    {return 3;}

    T& operator()(const int i)
    {assert(unsigned(i)<3);return ((T*)this)[i];}

    const T& operator()(const int i) const
    {assert(unsigned(i)<3);return ((T*)this)[i];}

    T first() const
    {return x00;}

    T last() const
    {return x22;}

    bool operator==(const DiagonalMatrix& A) const
    {return x00==A.x00 && x11==A.x11 && x22==A.x22;}

    bool operator!=(const DiagonalMatrix& A) const
    {return x00!=A.x00 || x11!=A.x11 || x22!=A.x22;}

    DiagonalMatrix operator-() const
    {return DiagonalMatrix(-x00,-x11,-x22);}

    DiagonalMatrix& operator+=(const DiagonalMatrix& A)
    {x00+=A.x00;x11+=A.x11;x22+=A.x22;return *this;}

    DiagonalMatrix& operator+=(const T& a)
    {x00+=a;x11+=a;x22+=a;return *this;}

    DiagonalMatrix& operator-=(const DiagonalMatrix& A)
    {x00-=A.x00;x11-=A.x11;x22-=A.x22;return *this;}

    DiagonalMatrix& operator-=(const T& a)
    {x00-=a;x11-=a;x22-=a;return *this;}

    DiagonalMatrix& operator*=(const T a)
    {x00*=a;x11*=a;x22*=a;return *this;}

    DiagonalMatrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;x00*=s;x11*=s;x22*=s;return *this;}

    DiagonalMatrix operator+(const DiagonalMatrix& A) const
    {return DiagonalMatrix(x00+A.x00,x11+A.x11,x22+A.x22);}

    Matrix<T,3> operator+(const Matrix<T,3>& A) const
    {return Matrix<T,3>(x00+A.x[0],A.x[1],A.x[2],A.x[3],x11+A.x[4],A.x[5],A.x[6],A.x[7],x22+A.x[8]);}

    Matrix<T,3> operator-(const Matrix<T,3>& A) const
    {return Matrix<T,3>(x00-A.x[0],-A.x[1],-A.x[2],-A.x[3],x11-A.x[4],-A.x[5],-A.x[6],-A.x[7],x22-A.x[8]);}

    DiagonalMatrix operator+(const T a) const
    {return DiagonalMatrix(x00+a,x11+a,x22+a);}

    DiagonalMatrix operator-(const DiagonalMatrix& A) const
    {return DiagonalMatrix(x00-A.x00,x11-A.x11,x22-A.x22);}

    DiagonalMatrix operator-(const T a) const
    {return DiagonalMatrix(x00-a,x11-a,x22-a);}

    DiagonalMatrix operator*(const T a) const
    {return DiagonalMatrix(a*x00,a*x11,a*x22);}

    DiagonalMatrix operator/(const T a) const
    {assert(a!=0);return *this*(1/a);}

    Vector<T,3> operator*(const Vector<T,3>& v) const
    {return Vector<T,3>(x00*v.x,x11*v.y,x22*v.z);}

    DiagonalMatrix operator*(const DiagonalMatrix& A) const
    {return DiagonalMatrix(x00*A.x00,x11*A.x11,x22*A.x22);}

    DiagonalMatrix& operator*=(const DiagonalMatrix& A)
    {return *this=*this*A;}

    DiagonalMatrix operator/(const DiagonalMatrix& A) const
    {return DiagonalMatrix(x00/A.x00,x11/A.x11,x22/A.x22);}

    T determinant() const
    {return x00*x11*x22;}

    DiagonalMatrix inverse() const
    {assert(x00!=0 && x11!=0 && x22!=0);return DiagonalMatrix(1/x00,1/x11,1/x22);}

    Vector<T,3> solve_linear_system(const Vector<T,3>& v) const
    {assert(x00!=0 && x11!=0 && x22!=0);return Vector<T,3>(v.x/x00,v.y/x11,v.z/x22);}

    Vector<T,3> robust_solve_linear_system(const Vector<T,3>& v) const
    {return Vector<T,3>(robust_divide(v.x,x00),robust_divide(v.y,x11),robust_divide(v.z,x22));}

    template<class TMatrix>
    typename ProductTranspose<DiagonalMatrix,TMatrix>::type times_transpose(const TMatrix& B) const
    {return (B**this).transposed();}

    DiagonalMatrix times_transpose(const DiagonalMatrix& M) const
    {return *this*M;}

    const DiagonalMatrix& transposed() const
    {return *this;}

    void transpose()
    {}

    template<class TMatrix>
    typename Product<DiagonalMatrix,TMatrix>::type transpose_times(const TMatrix& M) const
    {return *this*M;}

    SymmetricMatrix<T,3> conjugate_with_cross_product_matrix(const Vector<T,3>& v) const
    {T yy=sqr(v.y),zz=sqr(v.z),bx=x11*v.x,cx=x22*v.x;return SymmetricMatrix<T,3>(x11*zz+x22*yy,-cx*v.y,-bx*v.z,x00*zz+cx*v.x,-v.y*v.z*x00,x00*yy+bx*v.x);}

    DiagonalMatrix cofactor_matrix() const
    {return DiagonalMatrix(x11*x22,x00*x22,x00*x11);}

    T trace() const
    {return x00+x11+x22;}

    T dilational() const
    {return T(1./3)*trace();}

    T min() const
    {return geode::min(x00,x11,x22);}

    T max() const
    {return geode::max(x00,x11,x22);}

    T minabs() const
    {return geode::minabs(x00,x11,x22);}

    T maxabs() const
    {return geode::maxabs(x00,x11,x22);}

    T inner_product(const Vector<T,3>& a,const Vector<T,3>& b) const // inner product with respect to this matrix
    {return a.x*x00*b.x+a.y*x11*b.y+a.z*x22*b.z;}

    T inverse_inner_product(const Vector<T,3>& a,const Vector<T,3>& b) const // inner product with respect to the inverse of this matrix
    {assert(x00!=0 && x11!=0 && x22!=0);return a.x/x00*b.x+a.y/x11*b.y+a.z/x22*b.z;}

    T sqr_frobenius_norm() const
    {return sqr(x00)+sqr(x11)+sqr(x22);}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    bool positive_definite() const
    {return x00>0 && x11>0 && x22>0;}

    bool positive_semidefinite() const
    {return x00>=0 && x11>=0 && x22>=0;}

    DiagonalMatrix positive_definite_part() const
    {return clamp_min(0);}

    DiagonalMatrix sqrt() const
    {return DiagonalMatrix(geode::sqrt(x00),geode::sqrt(x11),geode::sqrt(x22));}

    DiagonalMatrix clamp_min(const T a) const
    {return DiagonalMatrix(geode::clamp_min(x00,a),geode::clamp_min(x11,a),geode::clamp_min(x22,a));}

    DiagonalMatrix clamp_max(const T a) const
    {return DiagonalMatrix(geode::clamp_max(x00,a),geode::clamp_max(x11,a),geode::clamp_max(x22,a));}

    DiagonalMatrix abs() const
    {return DiagonalMatrix(geode::abs(x00),geode::abs(x11),geode::abs(x22));}

    DiagonalMatrix sign() const
    {return DiagonalMatrix(geode::sign(x00),geode::sign(x11),geode::sign(x22));}

    static DiagonalMatrix identity_matrix()
    {return DiagonalMatrix(1,1,1);}

    Vector<T,3> to_vector() const
    {return Vector<T,3>(x00,x11,x22);}

//#####################################################################
    Matrix<T,3> times_transpose(const Matrix<T,3>& A) const;
//#####################################################################
};
// global functions
template<class T>
inline DiagonalMatrix<T,3> operator*(const T a,const DiagonalMatrix<T,3>& A)
{return A*a;}

template<class T>
inline DiagonalMatrix<T,3> operator+(const T a,const DiagonalMatrix<T,3>& A)
{return A+a;}

template<class T>
inline DiagonalMatrix<T,3> operator-(const T a,const DiagonalMatrix<T,3>& A)
{return -A+a;}

template<class T>
inline Matrix<T,3> operator+(const Matrix<T,3>& A,const DiagonalMatrix<T,3>& B)
{return B+A;}

template<class T>
inline Matrix<T,3> operator-(const Matrix<T,3>& A,const DiagonalMatrix<T,3>& B)
{return -B+A;}

template<class T>
inline std::istream& operator>>(std::istream& input,DiagonalMatrix<T,3>& A)
{return input>>A.x00>>A.x11>>A.x22;}

template<class T>
inline std::ostream& operator<<(std::ostream& output,const DiagonalMatrix<T,3>& A)
{return output<<A.x00<<" "<<A.x11<<" "<<A.x22;}

template<class T>
inline DiagonalMatrix<T,3> log(const DiagonalMatrix<T,3>& A)
{return DiagonalMatrix<T,3>(log(A.x00),log(A.x11),log(A.x22));}

template<class T>
inline DiagonalMatrix<T,3> exp(const DiagonalMatrix<T,3>& A)
{return DiagonalMatrix<T,3>(exp(A.x00),exp(A.x11),exp(A.x22));}

template<class T> inline Matrix<T,3> DiagonalMatrix<T,3>::
times_transpose(const Matrix<T,3>& A) const
{
    return Matrix<T,3>(x00*A.x[0][0],x11*A.x[0][1],x22*A.x[0][2],x00*A.x[1][0],x11*A.x[1][1],x22*A.x[1][2],x00*A.x[2][0],x11*A.x[2][1],x22*A.x[2][2]);
}

template<class T> inline T
inner_product(const DiagonalMatrix<T,3>& A,const DiagonalMatrix<T,3>& B) {
  return A.x00*B.x00+A.x11*B.x11+A.x22*B.x22;
}

template<class T> inline T
inner_product_conjugate(const DiagonalMatrix<T,3>& A,const Matrix<T,3>& Q,const DiagonalMatrix<T,3> B) {
  Matrix<T,3> BQ=B*Q.transposed();
  return A.x00*(Q.x[0]*BQ.x[0]+Q.x[3]*BQ.x[1]+Q.x[6]*BQ.x[2])+A.x11*(Q.x[1]*BQ.x[3]+Q.x[4]*BQ.x[4]+Q.x[7]*BQ.x[5])+A.x22*(Q.x[2]*BQ.x[6]+Q.x[5]*BQ.x[7]+Q.x[8]*BQ.x[8]);
}

}
