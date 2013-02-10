//#####################################################################
// Class Matrix<T,3,2>
//#####################################################################
#pragma once

#include <other/core/vector/Vector.h>
#include <other/core/math/sqr.h>
#include <other/core/math/max.h>
#include <other/core/vector/SymmetricMatrix2x2.h>
namespace other {

template<class T>
class Matrix<T,3,2>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=3,n=2};
    static const bool is_const=false;

    T x[3][2];

    Matrix()
    {
        for(int i=0;i<3;i++) for(int j=0;j<2;j++) x[i][j]=0;
    }

    template<class T2> explicit
    Matrix(const Matrix<T2,3,2>& matrix)
    {
        for(int i=0;i<3;i++) for(int j=0;j<2;j++) x[i][j]=(T)matrix.x[i][j];
    }

    Matrix(const T x00,const T x10,const T x20,const T x01,const T x11,const T x21)
    {
        x[0][0]=x00;x[1][0]=x10;x[2][0]=x20;x[0][1]=x01;x[1][1]=x11;x[2][1]=x21;
    }

    explicit Matrix(RawArray<const T,2> A)
    {
        assert(A.m==3 && A.n==2);
        for(int i=0;i<3;i++) for(int j=0;j<2;j++) x[i][j]=A(i,j);
    }

    Matrix(const Vector<T,3>& column0,const Vector<T,3>& column1)
    {
        x[0][0]=column0.x;x[1][0]=column0.y;x[2][0]=column0.z;x[0][1]=column1.x;x[1][1]=column1.y;x[2][1]=column1.z;
    }

    static Matrix row_major(T x00,T x01,T x10,T x11,T x20,T x21) // TODO: revisit
    {
        Matrix m;
        m.x[0][0]=x00;m.x[0][1]=x01;
        m.x[1][0]=x10;m.x[1][1]=x11;
        m.x[2][0]=x20;m.x[2][1]=x21;
        return m;
    }

    static Matrix column_major(T x00,T x10,T x20,T x01,T x11,T x21) // TODO: revisit
    {
        Matrix m;
        m.x[0][0]=x00;m.x[0][1]=x01;
        m.x[1][0]=x10;m.x[1][1]=x11;
        m.x[2][0]=x20;m.x[2][1]=x21;
        return m;
    }

    static Matrix column_major(const Vector<T,3>& c0,const Vector<T,3>& c1) { // TODO: revisit
      return Matrix(c0,c1);
    }

    void copy(const Matrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(3,2);}

    int rows() const
    {return 3;}

    int columns() const
    {return 2;}

    T& operator()(const int i,const int j)
    {assert(unsigned(i)<3 && unsigned(j)<2);return x[i][j];}

    const T& operator()(const int i,const int j) const
    {assert(unsigned(i)<3 && unsigned(j)<2);return x[i][j];}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<3 && unsigned(j)<2;}

    const Vector<T,3> column(const int j) const
    {assert(unsigned(j)<2);return Vector<T,3>(x[0][j],x[1][j],x[2][j]);}

    void set_column(const int j,const Vector<T,3>& c)
    {assert(unsigned(j)<2);x[0][j]=c[0];x[1][j]=c[1];x[2][j]=c[2];}

    bool operator==(const Matrix& A) const
    {for(int i=0;i<3;i++) for(int j=0;j<2;j++) if(x[i][j]!=A.x[i][j]) return false;return true;}

    bool operator!=(const Matrix& A) const
    {return !(*this==A);}

    Vector<T,3> column_sum() const
    {return Vector<T,3>(x[0][0]+x[0][1],x[1][0]+x[1][1],x[2][0]+x[2][1]);}

    Matrix operator-() const
    {return Matrix::column_major(-x[0][0],-x[1][0],-x[2][0],-x[0][1],-x[1][1],-x[2][1]);}

    Matrix& operator+=(const Matrix& A)
    {for(int i=0;i<3;i++) for(int j=0;j<2;j++) x[i][j]+=A.x[i][j];return *this;}

    Matrix& operator-=(const Matrix& A)
    {for(int i=0;i<3;i++) for(int j=0;j<2;j++) x[i][j]-=A.x[i][j];return *this;}

    Matrix& operator*=(const Matrix<T,2>& A)
    {return *this=*this*A;}

    Matrix& operator*=(const T a)
    {for(int i=0;i<3;i++) for(int j=0;j<2;j++) x[i][j]*=a;return *this;}

    Matrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;for(int i=0;i<3;i++) for(int j=0;j<2;j++) x[i][j]*=s;return *this;}

    Matrix operator+(const Matrix& A) const // 6 adds
    {return Matrix::column_major(x[0][0]+A(0,0),x[1][0]+A(1,0),x[2][0]+A(2,0),x[0][1]+A(0,1),x[1][1]+A(1,1),x[2][1]+A(2,1));}

    Matrix operator-(const Matrix& A) const // 6 adds
    {return Matrix::column_major(x[0][0]-A(0,0),x[1][0]-A(1,0),x[2][0]-A(2,0),x[0][1]-A(0,1),x[1][1]-A(1,1),x[2][1]-A(2,1));}

    template<int p>
    Matrix<T,3,p> operator*(const Matrix<T,2,p>& A) const
    {Matrix<T,3,p> result;
    for(int i=0;i<3;i++) for(int j=0;j<p;j++) for(int k=0;k<2;k++) result(i,j)+=x[i][k]*A(k,j);
    return result;}

    Matrix times_transpose(const UpperTriangularMatrix<T,2>& A) const // 9 mults, 3 adds
    {return Matrix::column_major(x[0][0]*A.x00+x[0][1]*A.x01,x[1][0]*A.x00+x[1][1]*A.x01,x[2][0]*A.x00+x[2][1]*A.x01,x[0][1]*A.x11,x[1][1]*A.x11,x[2][1]*A.x11);}

    template<class TMatrix>
    typename ProductTranspose<Matrix,TMatrix>::type
    times_transpose(const TMatrix& A) const
    {return (A*transposed()).transposed();}

    Matrix<T,3> times_transpose(const Matrix& A) const // 18 mults, 9 adds
    {return Matrix<T,3>(x[0][0]*A(0,0)+x[0][1]*A(0,1),x[1][0]*A(0,0)+x[1][1]*A(0,1),x[2][0]*A(0,0)+x[2][1]*A(0,1),
                        x[0][0]*A(1,0)+x[0][1]*A(1,1),x[1][0]*A(1,0)+x[1][1]*A(1,1),x[2][0]*A(1,0)+x[2][1]*A(1,1),
                        x[0][0]*A(2,0)+x[0][1]*A(2,1),x[1][0]*A(2,0)+x[1][1]*A(2,1),x[2][0]*A(2,0)+x[2][1]*A(2,1));}

    Matrix operator*(const T a) const // 6 mults
    {return Matrix::column_major(a*x[0][0],a*x[1][0],a*x[2][0],a*x[0][1],a*x[1][1],a*x[2][1]);}

    Matrix operator/(const T a) const // 6 mults, 1 div
    {assert(a!=0);T s=1/a;return Matrix::column_major(s*x[0][0],s*x[1][0],s*x[2][0],s*x[0][1],s*x[1][1],s*x[2][1]);}

    Vector<T,3> operator*(const Vector<T,2>& v) const // 6 mults, 3 adds
    {return Vector<T,3>(x[0][0]*v.x+x[0][1]*v.y,x[1][0]*v.x+x[1][1]*v.y,x[2][0]*v.x+x[2][1]*v.y);}

    UpperTriangularMatrix<T,2> R_from_QR_factorization() const // Gram Schmidt
    {T x_dot_x=column(0).sqr_magnitude(),x_dot_y=dot(column(0),column(1)),y_dot_y=column(1).sqr_magnitude();
    T r11=sqrt(x_dot_x),r12=r11?x_dot_y/r11:0,r22=sqrt(max((T)0,y_dot_y-r12*r12));
    return UpperTriangularMatrix<T,2>(r11,r12,r22);}

    SymmetricMatrix<T,2> normal_equations_matrix() const // 9 mults, 6 adds
    {return SymmetricMatrix<T,2>(x[0][0]*x[0][0]+x[1][0]*x[1][0]+x[2][0]*x[2][0],x[0][1]*x[0][0]+x[1][1]*x[1][0]+x[2][1]*x[2][0],x[0][1]*x[0][1]+x[1][1]*x[1][1]+x[2][1]*x[2][1]);}

    Matrix<T,2> transpose_times(const Matrix& A) const // 12 mults, 8 adds
    {return Matrix<T,2>(x[0][0]*A(0,0)+x[1][0]*A(1,0)+x[2][0]*A(2,0),x[0][1]*A(0,0)+x[1][1]*A(1,0)+x[2][1]*A(2,0),x[0][0]*A(0,1)+x[1][0]*A(1,1)+x[2][0]*A(2,1),x[0][1]*A(0,1)+x[1][1]*A(1,1)+x[2][1]*A(2,1));}

    Vector<T,2> transpose_times(const Vector<T,3>& v) const // 6 mults, 4 adds
    {return Vector<T,2>(x[0][0]*v.x+x[1][0]*v.y+x[2][0]*v.z,x[0][1]*v.x+x[1][1]*v.y+x[2][1]*v.z);}

    template<class TMatrix>
    typename TransposeProduct<Matrix,TMatrix>::type
    transpose_times(const TMatrix& A) const
    {return transposed()*A;}

    T maxabs() const
    {return other::maxabs(x[0][0],x[1][0],x[2][0],x[0][1],x[1][1],x[2][1]);}

    T sqr_frobenius_norm() const
    {return sqr(x[0][0])+sqr(x[1][0])+sqr(x[2][0])+sqr(x[0][1])+sqr(x[1][1])+sqr(x[2][1]);}

    Matrix operator*(const UpperTriangularMatrix<T,2>& A) const // 9 mults, 3 adds
    {return Matrix::column_major(x[0][0]*A.x00,x[1][0]*A.x00,x[2][0]*A.x00,x[0][0]*A.x01+x[0][1]*A.x11,x[1][0]*A.x01+x[1][1]*A.x11,x[2][0]*A.x01+x[2][1]*A.x11);}

    Matrix operator*(const SymmetricMatrix<T,2>& A) const // 12 mults, 6 adds
    {return Matrix::column_major(x[0][0]*A.x00+x[0][1]*A.x10,x[1][0]*A.x00+x[1][1]*A.x10,x[2][0]*A.x00+x[2][1]*A.x10,x[0][0]*A.x10+x[0][1]*A.x11,x[1][0]*A.x10+x[1][1]*A.x11,x[2][0]*A.x10+x[2][1]*A.x11);}

    Matrix operator*(const DiagonalMatrix<T,2>& A) const // 6 mults
    {return Matrix::column_major(x[0][0]*A.x00,x[1][0]*A.x00,x[2][0]*A.x00,x[0][1]*A.x11,x[1][1]*A.x11,x[2][1]*A.x11);}

    Vector<T,3> weighted_normal() const
    {return cross(column(0),column(1));}

    Matrix cofactor_matrix() const
    {Vector<T,3> normal=weighted_normal().normalized();
    return Matrix::column_major(cross(column(1),normal),cross(normal,column(0)));}

    T parallelepiped_measure() const
    {return weighted_normal().magnitude();}

    Matrix<T,2,3> transposed() const
    {return Matrix<T,2,3>::column_major(x[0][0],x[0][1],x[1][0],x[1][1],x[2][0],x[2][1]);}

    void fast_singular_value_decomposition(Matrix<T,3>& U,DiagonalMatrix<T,2>& singular_values,Matrix<T,2>& V) const
    {Matrix<T,3,2> U_;fast_singular_value_decomposition(U_,singular_values,V);
    U=Matrix<T,3>(U_.column(0),U_.column(1),cross(U_.column(0),U_.column(1)));}

//#####################################################################
    OTHER_CORE_EXPORT void fast_singular_value_decomposition(Matrix& U,DiagonalMatrix<T,2>& singular_values,Matrix<T,2>& V) const ;
    OTHER_CORE_EXPORT void fast_indefinite_polar_decomposition(Matrix<T,3,2>& Q,SymmetricMatrix<T,2>& S) const ;
//#####################################################################
};

template<class T>
inline Matrix<T,3,2>
HStack(const Vector<T,3>& a,const Vector<T,3>& b)
{return Matrix<T,3,2>::row_major(a[0],b[0],a[1],b[1],a[2],b[2]);}

// global functions
template<class T>
inline Matrix<T,3,2> operator*(const T a,const Matrix<T,3,2>& A) // 6 mults
{return Matrix<T,3,2>::column_major(a*A(0,0),a*A(1,0),a*A(2,0),a*A(0,1),a*A(1,1),a*A(2,1));}

template<class T>
inline Vector<T,3> operator*(const Vector<T,2>& v,const Matrix<T,3,2>& A) // 6 mults, 3 adds
{return Vector<T,3>::column_major(v.x*A(0,0)+v.y*A(0,1),v.x*A(1,0)+v.y*A(1,1),v.x*A(2,0)+v.y*A(2,1));}

template<class T>
inline Matrix<T,3,2> operator*(const SymmetricMatrix<T,3>& A,const Matrix<T,3,2>& B) // 18 mults, 12 adds
{return Matrix<T,3,2>::column_major(A.x00*B.x[0][0]+A.x10*B.x[1][0]+A.x20*B.x[2][0],A.x10*B.x[0][0]+A.x11*B.x[1][0]+A.x21*B.x[2][0],A.x20*B.x[0][0]+A.x21*B.x[1][0]+A.x22*B.x[2][0],
    A.x00*B.x[0][1]+A.x10*B.x[1][1]+A.x20*B.x[2][1],A.x10*B.x[0][1]+A.x11*B.x[1][1]+A.x21*B.x[2][1],A.x20*B.x[0][1]+A.x21*B.x[1][1]+A.x22*B.x[2][1]);}

template<class T>
inline Matrix<T,3,2> operator*(const UpperTriangularMatrix<T,3>& A,const Matrix<T,3,2>& B)
{return Matrix<T,3,2>::column_major(A.x00*B.x[0][0]+A.x01*B.x[1][0]+A.x02*B.x[2][0],A.x11*B.x[1][0]+A.x12*B.x[2][0],A.x22*B.x[2][0],
    A.x00*B.x[0][1]+A.x01*B.x[1][1]+A.x02*B.x[2][1],A.x11*B.x[1][1]+A.x12*B.x[2][1],A.x22*B.x[2][1]);}

template<class T> inline T inner_product(const Matrix<T,3,2>& A,const Matrix<T,3,2>& B) { // 6 mults, 5 adds
  return A(0,0)*B.x[0][0]+A(1,0)*B.x[1][0]+A(2,0)*B.x[2][0]+A(0,1)*B.x[0][1]+A(1,1)*B.x[1][1]+A(2,1)*B.x[2][1];
}

template<class T>
inline std::istream& operator>>(std::istream& input,Matrix<T,3,2>& A)
{for(int i=0;i<3;i++) for(int j=0;j<2;j++) input>>A.x[i][j];return input;}

}
