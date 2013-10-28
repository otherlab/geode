//#####################################################################
// Class UpperTriangularMatrix3x3
//#####################################################################
#pragma once

#include <geode/vector/forward.h>
#include <ostream>
namespace geode {

template<class T> struct IsScalarBlock<UpperTriangularMatrix<T,3> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<UpperTriangularMatrix<T,3> >:public IsScalarVectorSpace<T>{};

template<class T>
class UpperTriangularMatrix<T,3>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=3};
    enum Workaround2 {n=3};

    T x00,x01,x11,x02,x12,x22;

    UpperTriangularMatrix()
        :x00(),x01(),x11(),x02(),x12(),x22()
    {
        BOOST_STATIC_ASSERT(sizeof(UpperTriangularMatrix)==6*sizeof(T));
    }

    template<class T2> explicit
    UpperTriangularMatrix(const UpperTriangularMatrix<T2,3>& matrix)
        :x00(matrix.x00),x01(matrix.x01),x11(matrix.x11),x02(matrix.x02),x12(matrix.x12),x22(matrix.x22)
    {}

    UpperTriangularMatrix(const T x00,const T x01,const T x11,const T x02,const T x12,const T x22)
        :x00(x00),x01(x01),x11(x11),x02(x02),x12(x12),x22(x22)
    {}

    Vector<int,2> sizes() const
    {return Vector<int,2>(3,3);}

    int rows() const
    {return 3;}

    int columns() const
    {return 3;}

    T& operator()(const int i,const int j)
    {assert(unsigned(i)<unsigned(j) && unsigned(j)<3);return ((T*)this)[((j*(j+1))>>1)+i];}

    const T& operator()(const int i,const int j) const
    {assert(unsigned(i)<unsigned(j) && unsigned(j)<3);return ((const T*)this)[((j*(j+1))>>1)+i];}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<unsigned(j) && unsigned(j)<3;}

    bool operator==(const UpperTriangularMatrix& A) const
    {return x00==A.x00 && x01==A.x01 && x11==A.x11 && x02==A.x02 && x12==A.x12 && x22==A.x22;}

    bool operator!=(const UpperTriangularMatrix& A) const
    {return !(*this==A);}

    UpperTriangularMatrix operator-() const
    {return UpperTriangularMatrix(-x00,-x01,-x11,-x02,-x12,-x22);}

    UpperTriangularMatrix& operator+=(const UpperTriangularMatrix& A)
    {x00+=A.x00;x01+=A.x01;x11+=A.x11;x02+=A.x02;x12+=A.x12;x22+=A.x22;return *this;}

    UpperTriangularMatrix& operator+=(const T& a)
    {x00+=a;x11+=a;x22+=a;return *this;}

    UpperTriangularMatrix& operator-=(const UpperTriangularMatrix& A)
    {x00-=A.x00;x01-=A.x01;x11-=A.x11;x02-=A.x02;x12-=A.x12;x22-=A.x22;return *this;}

    UpperTriangularMatrix& operator-=(const T& a)
    {x00-=a;x11-=a;x22-=a;return *this;}

    UpperTriangularMatrix operator+(const UpperTriangularMatrix& A) const
    {return UpperTriangularMatrix(x00+A.x00,x01+A.x01,x11+A.x11,x02+A.x02,x12+A.x12,x22+A.x22);}

    UpperTriangularMatrix operator+(const DiagonalMatrix<T,3>& A) const
    {return UpperTriangularMatrix(x00+A.x00,x01,x11+A.x11,x02,x12,x22+A.x22);}

    UpperTriangularMatrix operator+(const T a) const
    {return UpperTriangularMatrix(x00+a,x01,x11+a,x02,x12,x22+a);}

    UpperTriangularMatrix operator-(const UpperTriangularMatrix& A) const
    {return UpperTriangularMatrix(x00-A.x00,x01-A.x01,x11-A.x11,x02-A.x02,x12-A.x12,x22-A.x22);}

    UpperTriangularMatrix operator-(const DiagonalMatrix<T,3>& A) const
    {return UpperTriangularMatrix(x00-A.x00,x01,x11-A.x11,x02,x12,x22-A.x22);}

    UpperTriangularMatrix operator-(const T a) const
    {return UpperTriangularMatrix(x00-a,x01,x11-a,x02,x12,x22-a);}

    UpperTriangularMatrix& operator*=(const UpperTriangularMatrix& A)
    {return *this=*this*A;}

    UpperTriangularMatrix& operator*=(const T a)
    {x00*=a;x01*=a;x11*=a;x02*=a;x12*=a;x22*=a;return *this;}

    UpperTriangularMatrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;x00*=s;x01*=s;x11*=s;x02*=s;x12*=s;x22*=s;return *this;}

    UpperTriangularMatrix operator*(const T a) const
    {return UpperTriangularMatrix(a*x00,a*x01,a*x11,a*x02,a*x12,a*x22);}

    UpperTriangularMatrix operator/(const T a) const
    {assert(a!=0);T s=1/a;return UpperTriangularMatrix(s*x00,s*x01,s*x11,s*x02,s*x12,s*x22);}

    Vector<T,3> operator*(const Vector<T,3>& v) const // 6 mults, 3 adds
    {return Vector<T,3>(x00*v.x+x01*v.y+x02*v.z,x11*v.y+x12*v.z,x22*v.z);}

    UpperTriangularMatrix operator*(const UpperTriangularMatrix& A) const
    {return UpperTriangularMatrix(x00*A.x00,x00*A.x01+x01*A.x11,x11*A.x11,x00*A.x02+x01*A.x12+x02*A.x22,x11*A.x12+x12*A.x22,x22*A.x22);}

    UpperTriangularMatrix operator*(const DiagonalMatrix<T,3>& A) const
    {return UpperTriangularMatrix(x00*A.x00,x01*A.x11,x11*A.x11,x02*A.x22,x12*A.x22,x22*A.x22);}

    T determinant() const
    {return x00*x11*x22;}

    T trace() const
    {return x00+x11+x22;}

    UpperTriangularMatrix inverse() const
    {T determinant=x00*x11*x22;assert(determinant!=0);T s=1/determinant;
    return s*UpperTriangularMatrix(x11*x22,-x01*x22,x00*x22,x01*x12-x11*x02,-x00*x12,x00*x11);}

    Vector<T,3> solve_linear_system(const Vector<T,3>& b) const
    {return cofactor_matrix()*(b/determinant());}

    UpperTriangularMatrix cofactor_matrix() const
    {return UpperTriangularMatrix(x11*x22,-x01*x22,x00*x22,x01*x12-x11*x02,-x00*x12,x00*x11);}

    static UpperTriangularMatrix identity_matrix()
    {return UpperTriangularMatrix(1,0,1,0,0,1);}

    T maxabs() const
    {return maxabs(x00,x01,x11,x02,x12,x22);}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    T sqr_frobenius_norm() const
    {return sqr(x00)+sqr(x01)+sqr(x11)+sqr(x02)+sqr(x12)+sqr(x22);}

    T simplex_minimum_altitude() const
    {return Matrix<T,3>(*this).simplex_minimum_altitude();}

//#####################################################################
};
// global functions
template<class T>
inline UpperTriangularMatrix<T,3> operator*(const T a,const UpperTriangularMatrix<T,3>& A)
{return A*a;}

template<class T>
inline UpperTriangularMatrix<T,3> operator+(const T a,const UpperTriangularMatrix<T,3>& A)
{return A+a;}

template<class T>
inline UpperTriangularMatrix<T,3> operator-(const T a,const UpperTriangularMatrix<T,3>& A)
{return -A+a;}

template<class T>
inline UpperTriangularMatrix<T,3> operator*(const DiagonalMatrix<T,3>& A,const UpperTriangularMatrix<T,3>& B)
{return UpperTriangularMatrix<T,3>(A.x00*B.x00,A.x00*B.x01,A.x11*B.x11,A.x00*B.x02,A.x11*B.x12,A.x22*B.x22);}

template<class T>
inline UpperTriangularMatrix<T,3> operator+(const DiagonalMatrix<T,3>& A,const UpperTriangularMatrix<T,3>& B)
{return B+A;}

template<class T>
inline UpperTriangularMatrix<T,3> operator-(const DiagonalMatrix<T,3>& A,const UpperTriangularMatrix<T,3>& B)
{return -B+A;}

template<class T>
inline std::ostream& operator<<(std::ostream& output,const UpperTriangularMatrix<T,3>& A)
{return output<<A.x00<<" "<<A.x01<<" "<<A.x02<<"\n0 "<<A.x11<<" "<<A.x12<<"\n0 0 "<<A.x22<<"\n";}

}
