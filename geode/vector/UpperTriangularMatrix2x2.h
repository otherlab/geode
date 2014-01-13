//#####################################################################
// Class UpperTriangularMatrix2x2
//#####################################################################
#pragma once

#include <geode/vector/forward.h>
#include <cmath>
#include <ostream>
namespace geode {

using ::std::sqrt;

template<class T> struct IsScalarBlock<UpperTriangularMatrix<T,2> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<UpperTriangularMatrix<T,2> >:public IsScalarVectorSpace<T>{};

template<class T>
class UpperTriangularMatrix<T,2>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=2};
    enum Workaround2 {n=2};

    T x00,x01,x11;

    UpperTriangularMatrix()
        :x00(),x01(),x11()
    {
        static_assert(sizeof(UpperTriangularMatrix)==3*sizeof(T),"");
    }

    template<class T2> explicit
    UpperTriangularMatrix(const UpperTriangularMatrix<T2,2>& matrix)
        :x00(matrix.x00),x01(matrix.x01),x11(matrix.x11)
    {}

    UpperTriangularMatrix(const T x00,const T x01,const T x11)
        :x00(x00),x01(x01),x11(x11)
    {}

    Vector<int,2> sizes() const
    {return Vector<int,2>(2,2);}

    int rows() const
    {return 2;}

    int columns() const
    {return 2;}

    T& operator()(const int i,const int j)
    {assert(unsigned(i)<unsigned(j) && unsigned(j)<2);return ((T*)this)[((j*(j+1))>>1)+i];}

    const T& operator()(const int i,const int j) const
    {assert(unsigned(i)<unsigned(j) && unsigned(j)<2);return ((const T*)this)[((j*(j+1))>>1)+i];}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<unsigned(j) && unsigned(j)<2;}

    bool operator==(const UpperTriangularMatrix& A) const
    {return x00==A.x00 && x01==A.x01 && x11==A.x11;}

    bool operator!=(const UpperTriangularMatrix& A) const
    {return !(*this==A);}

    UpperTriangularMatrix operator-() const
    {return UpperTriangularMatrix(-x00,-x01,-x11);}

    UpperTriangularMatrix& operator+=(const UpperTriangularMatrix& A)
    {x00+=A.x00;x01+=A.x01;x11+=A.x11;return *this;}

    UpperTriangularMatrix& operator+=(const T& a)
    {x00+=a;x11+=a;return *this;}

    UpperTriangularMatrix& operator-=(const UpperTriangularMatrix& A)
    {x00-=A.x00;x01-=A.x01;x11-=A.x11;return *this;}

    UpperTriangularMatrix& operator-=(const T& a)
    {x00-=a;x11-=a;return *this;}

    UpperTriangularMatrix operator+(const UpperTriangularMatrix& A) const
    {return UpperTriangularMatrix(x00+A.x00,x01+A.x01,x11+A.x11);}

    UpperTriangularMatrix operator+(const DiagonalMatrix<T,2>& A) const
    {return UpperTriangularMatrix(x00+A.x00,x01,x11+A.x11);}

    UpperTriangularMatrix operator+(const T a) const
    {return UpperTriangularMatrix(x00+a,x01,x11+a);}

    UpperTriangularMatrix operator-(const UpperTriangularMatrix& A) const
    {return UpperTriangularMatrix(x00-A.x00,x01-A.x01,x11-A.x11);}

    UpperTriangularMatrix operator-(const DiagonalMatrix<T,2>& A) const
    {return UpperTriangularMatrix(x00-A.x00,x01,x11-A.x11);}

    UpperTriangularMatrix operator-(const T a) const
    {return UpperTriangularMatrix(x00-a,x01,x11-a);}

    UpperTriangularMatrix& operator*=(const UpperTriangularMatrix& A)
    {return *this=*this*A;}

    UpperTriangularMatrix& operator*=(const T a)
    {x00*=a;x01*=a;x11*=a;return *this;}

    UpperTriangularMatrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;x00*=s;x01*=s;x11*=s;return *this;}

    UpperTriangularMatrix operator*(const T a) const
    {return UpperTriangularMatrix(a*x00,a*x01,a*x11);}

    UpperTriangularMatrix operator/(const T a) const
    {assert(a!=0);return *this*(1/a);}

    Vector<T,2> operator*(const Vector<T,2>& v) const
    {return Vector<T,2>(x00*v.x+x01*v.y,x11*v.y);}

    UpperTriangularMatrix operator*(const UpperTriangularMatrix& A) const // 4 mults, 1 add
    {return UpperTriangularMatrix(x00*A.x00,x00*A.x01+x01*A.x11,x11*A.x11);}

    UpperTriangularMatrix operator*(const DiagonalMatrix<T,2>& A) const // 3 mults
    {return UpperTriangularMatrix(x00*A.x00,x01*A.x11,x11*A.x11);}

    Matrix<T,2,3> times_transpose(const Matrix<T,3,2>& A) const
    {return Matrix<T,2,3>(x00*A.x[0]+x01*A.x[3],x11*A.x[3],x00*A.x[1]+x01*A.x[4],x11*A.x[4],x00*A.x[2]+x01*A.x[5],x11*A.x[5]);}

    SymmetricMatrix<T,2> outer_product_matrix() const // 4 mults, 1 add
    {return SymmetricMatrix<T,2>(x00*x00+x01*x01,x01*x11,x11*x11);}

    T determinant() const
    {return x00*x11;}

    T trace() const
    {return x00+x11;}

    UpperTriangularMatrix inverse() const
    {assert(x00!=0 && x11!=0);T one_over_x00=1/x00,one_over_x11=1/x11;
    return UpperTriangularMatrix(one_over_x00,-x01*one_over_x00*one_over_x11,one_over_x11);}

    Vector<T,2> solve_linear_system(const Vector<T,2>& b) const
    {return inverse()*b;}

    UpperTriangularMatrix cofactor_matrix() const
    {return UpperTriangularMatrix(x11,-x01,x00);}

    static UpperTriangularMatrix identity_matrix()
    {return UpperTriangularMatrix(1,0,1);}

    T maxabs() const
    {return maxabs(x00,x01,x11);}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    T sqr_frobenius_norm() const
    {return sqr(x00)+sqr(x01)+sqr(x11);}

    T simplex_minimum_altitude() const
    {return determinant()/max(abs(x00),sqrt(sqr(x11)+max(sqr(x01),sqr(x01-x00))));}

//#####################################################################
};
// global functions
template<class T>
inline UpperTriangularMatrix<T,2> operator*(const T a,const UpperTriangularMatrix<T,2>& A) // 3 mults
{return A*a;}

template<class T>
inline UpperTriangularMatrix<T,2> operator+(const T a,const UpperTriangularMatrix<T,2>& A)
{return A+a;}

template<class T>
inline UpperTriangularMatrix<T,2> operator-(const T a,const UpperTriangularMatrix<T,2>& A)
{return -A+a;}

template<class T>
inline UpperTriangularMatrix<T,2> operator*(const DiagonalMatrix<T,2>& A,const UpperTriangularMatrix<T,2>& B) // 3 mults
{return UpperTriangularMatrix<T,2>(A.x00*B.x00,A.x00*B.x01,A.x11*B.x11);}

template<class T>
inline UpperTriangularMatrix<T,2> operator+(const DiagonalMatrix<T,2>& A,const UpperTriangularMatrix<T,2>& B)
{return B+A;}

template<class T>
inline UpperTriangularMatrix<T,2> operator-(const DiagonalMatrix<T,2>& A,const UpperTriangularMatrix<T,2>& B)
{return -B+A;}

template<class T>
inline std::ostream& operator<<(std::ostream& output_stream,const UpperTriangularMatrix<T,2>& A)
{return output_stream<<A.x00<<" "<<A.x01<<"\n0 "<<A.x11<<"\n";}

}
