//#####################################################################
// Class Matrix<T,2>
//#####################################################################
#pragma once

#include <geode/vector/ArithmeticPolicy.h>
#include <geode/vector/Vector2d.h>
#include <geode/math/maxabs.h>
#include <geode/math/sqr.h>
namespace geode {

template<class T>
class Matrix<T,2>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=2};
    enum Workaround2 {n=2};
    static const bool is_const=false;

    T x[2][2];

    Matrix()
    {
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) x[i][j]=0;
    }

    Matrix(const Matrix& matrix)
    {
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) x[i][j]=matrix.x[i][j];
    }

    template<class T2> explicit
    Matrix(const Matrix<T2,2>& matrix)
    {
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) x[i][j]=(T)matrix.x[i][j];
    }

    Matrix(const SymmetricMatrix<T,2>& matrix)
    {
        x[0][0]=matrix.x00;x[1][0]=x[0][1]=matrix.x10;x[1][1]=matrix.x11;
    }

    Matrix(const UpperTriangularMatrix<T,2>& matrix)
    {
        x[0][0]=matrix.x00;x[0][1]=matrix.x01;x[1][1]=matrix.x11;x[1][0]=0;
    }

    Matrix(const T x00,const T x10,const T x01,const T x11)
    {
        x[0][0]=x00;x[1][0]=x10;x[0][1]=x01;x[1][1]=x11;
    }

    Matrix(const Vector<T,2>& column1,const Vector<T,2>& column2)
    {
        x[0][0]=column1.x;x[1][0]=column1.y;x[0][1]=column2.x;x[1][1]=column2.y;
    }

    explicit Matrix(RawArray<const T,2> A)
    {
        assert(A.m==2 && A.n==2);for(int j=0;j<2;j++) for(int i=0;i<2;i++) (*this)(i,j)=A(i,j);
    }

    Matrix& operator=(const Matrix& matrix_input)
    {
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) x[i][j]=matrix_input.x[i][j];return *this;
    }

    void copy(const Matrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(2,2);}

    int rows() const
    {return 2;}

    int columns() const
    {return 2;}

    T& operator()(const int i,const int j)
    {assert(unsigned(i)<2 && unsigned(j)<2);return x[i][j];}

    const T& operator()(const int i,const int j) const
    {assert(unsigned(i)<2 && unsigned(j)<2);return x[i][j];}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<2 && unsigned(j)<2;}

    const Vector<T,2> column(const int j) const
    {assert(unsigned(j)<2);return Vector<T,2>(x[0][j],x[1][j]);}

    void set_column(const int j,const Vector<T,2>& c)
    {assert(unsigned(j)<2);x[0][j]=c.x;x[1][j]=c.y;}

    bool operator==(const Matrix& A) const
    {for(int i=0;i<2;i++) for(int j=0;j<2;j++) if(x[i][j]!=A.x[i][j]) return false;return true;}

    bool operator!=(const Matrix& A) const
    {return !(*this==A);}

    Vector<T,2> column_sum() const
    {return Vector<T,2>(x[0][0]+x[0][1],x[1][0]+x[1][1]);}

    Vector<T,2> column_magnitudes() const
    {return Vector<T,2>(column(0).magnitude(),column(1).magnitude());}

    Matrix operator-() const
    {return Matrix(-x[0][0],-x[1][0],-x[0][1],-x[1][1]);}

    Matrix& operator+=(const Matrix& A)
    {for(int i=0;i<2;i++) for(int j=0;j<2;j++) x[i][j]+=A.x[i][j];return *this;}

    Matrix& operator+=(const T a)
    {x[0][0]+=a;x[1][1]+=a;return *this;}

    Matrix& operator-=(const Matrix& A)
    {for(int i=0;i<2;i++) for(int j=0;j<2;j++) x[i][j]-=A.x[i][j];return *this;}

    Matrix& operator-=(const T a)
    {x[0][0]-=a;x[1][1]-=a;return *this;}

    Matrix& operator*=(const Matrix& A)
    {return *this=*this*A;}

    Matrix& operator*=(const T a)
    {for(int i=0;i<2;i++) for(int j=0;j<2;j++) x[i][j]*=a;return *this;}

    Matrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;for(int i=0;i<2;i++) for(int j=0;j<2;j++) x[i][j]*=s;return *this;}

    Matrix operator+(const Matrix& A) const
    {return Matrix(x[0][0]+A.x[0][0],x[1][0]+A.x[1][0],x[0][1]+A.x[0][1],x[1][1]+A.x[1][1]);}

    Matrix operator+(const T a) const
    {return Matrix(x[0][0]+a,x[1][0],x[0][1],x[1][1]+a);}

    Matrix operator-(const Matrix& A) const
    {return Matrix(x[0][0]-A.x[0][0],x[1][0]-A.x[1][0],x[0][1]-A.x[0][1],x[1][1]-A.x[1][1]);}

    Matrix operator-(const T a) const
    {return Matrix(x[0][0]-a,x[1][0],x[0][1],x[1][1]-a);}

    Matrix operator*(const Matrix& A) const
    {return Matrix(x[0][0]*A.x[0][0]+x[0][1]*A.x[1][0],x[1][0]*A.x[0][0]+x[1][1]*A.x[1][0],x[0][0]*A.x[0][1]+x[0][1]*A.x[1][1],x[1][0]*A.x[0][1]+x[1][1]*A.x[1][1]);}

    template<int p>
    Matrix<T,2,p> operator*(const Matrix<T,2,p>& A) const
    {Matrix<T,2,p> result;
    for(int i=0;i<2;i++) for(int j=0;j<p;j++) for(int k=0;k<2;k++) result(i,j)+=x[i][k]*A(k,j);
    return result;}

    Matrix operator*(const T a) const
    {return Matrix(a*x[0][0],a*x[1][0],a*x[0][1],a*x[1][1]);}

    Matrix operator/(const T a) const
    {assert(a!=0);T s=1/a;return Matrix(s*x[0][0],s*x[1][0],s*x[0][1],s*x[1][1]);}

    Vector<T,2> operator*(const Vector<T,2>& v) const
    {return Vector<T,2>(x[0][0]*v.x+x[0][1]*v.y,x[1][0]*v.x+x[1][1]*v.y);}

    T determinant() const
    {return x[0][0]*x[1][1]-x[1][0]*x[0][1];}

    T parallelepiped_measure() const
    {return determinant();}

    void invert()
    {*this=inverse();}

    Matrix inverse() const
    {T one_over_determinant=1/(x[0][0]*x[1][1]-x[1][0]*x[0][1]);
    return Matrix(one_over_determinant*x[1][1],-one_over_determinant*x[1][0],-one_over_determinant*x[0][1],one_over_determinant*x[0][0]);}

    Matrix inverse_transposed() const
    {return inverse().transposed();}

    Vector<T,2> solve_linear_system(const Vector<T,2>& b) const
    {T one_over_determinant=1/(x[0][0]*x[1][1]-x[1][0]*x[0][1]);
    return one_over_determinant*Vector<T,2>(x[1][1]*b.x-x[0][1]*b.y,x[0][0]*b.y-x[1][0]*b.x);}

    Vector<T,2> robust_solve_linear_system(const Vector<T,2>& b) const
    {T determinant=this->determinant();
    Vector<T,2> unscaled_result=Vector<T,2>(x[1][1]*b.x-x[0][1]*b.y,x[0][0]*b.y-x[1][0]*b.x);
    T relative_tolerance=(T)FLT_MIN*unscaled_result.maxabs();
    if(abs(determinant)<=relative_tolerance){relative_tolerance=max(relative_tolerance,(T)FLT_MIN);determinant=determinant>=0?relative_tolerance:-relative_tolerance;}
    return unscaled_result/determinant;}

    void transpose()
    {swap(x[1][0],x[0][1]);}

    Matrix transposed() const
    {return Matrix(x[0][0],x[0][1],x[1][0],x[1][1]);}

    T trace() const
    {return x[0][0]+x[1][1];}

    Matrix cofactor_matrix() const // cheap
    {return Matrix(x[1][1],-x[0][1],-x[1][0],x[0][0]);}

    SymmetricMatrix<T,2> normal_equations_matrix() const // 6 mults, 3 adds
    {return SymmetricMatrix<T,2>(x[0][0]*x[0][0]+x[1][0]*x[1][0],x[0][0]*x[0][1]+x[1][0]*x[1][1],x[0][1]*x[0][1]+x[1][1]*x[1][1]);}

    SymmetricMatrix<T,2> outer_product_matrix() const // 6 mults, 3 adds
    {return SymmetricMatrix<T,2>(x[0][0]*x[0][0]+x[0][1]*x[0][1],x[0][0]*x[1][0]+x[0][1]*x[1][1],x[1][0]*x[1][0]+x[1][1]*x[1][1]);}

    static Matrix identity_matrix()
    {return Matrix(1,0,0,1);}

    static Matrix rotation_matrix(const T radians)
    {T c=cos(radians),s=sin(radians);return Matrix(c,s,-s,c);}

    static Matrix derivative_rotation_matrix(const T radians)
    {T c=cos(radians),s=sin(radians);return Matrix(-s,c,-c,-s);}

    T sqr_frobenius_norm() const
    {return sqr(x[0][0])+sqr(x[1][0])+sqr(x[0][1])+sqr(x[1][1]);}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    T maxabs() const
    {return geode::maxabs(x[0][0],x[1][0],x[0][1],x[1][1]);}

    Matrix operator*(const DiagonalMatrix<T,2>& A) const // 4 mults
    {return Matrix(x[0][0]*A.x00,x[1][0]*A.x00,x[0][1]*A.x11,x[1][1]*A.x11);}

    Matrix operator*(const UpperTriangularMatrix<T,2>& A) const // 6 mults, 2 adds
    {return Matrix(x[0][0]*A.x00,x[1][0]*A.x00,x[0][0]*A.x01+x[0][1]*A.x11,x[1][0]*A.x01+x[1][1]*A.x11);}

    Matrix operator*(const SymmetricMatrix<T,2>& A) const // 8 mults, 4 adds
    {return Matrix(x[0][0]*A.x00+x[0][1]*A.x10,x[1][0]*A.x00+x[1][1]*A.x10,x[0][0]*A.x10+x[0][1]*A.x11,x[1][0]*A.x10+x[1][1]*A.x11);}

    Matrix times_transpose(const UpperTriangularMatrix<T,2>& A) const // 6 mults, 2 adds
    {return Matrix(x[0][0]*A.x00+x[0][1]*A.x01,x[1][0]*A.x00+x[1][1]*A.x01,x[0][1]*A.x11,x[1][1]*A.x11);}

    Matrix times_transpose(const Matrix& A) const // 8 mults, 4 adds
    {return Matrix(x[0][0]*A.x[0][0]+x[0][1]*A.x[0][1],x[1][0]*A.x[0][0]+x[1][1]*A.x[0][1],x[0][0]*A.x[1][0]+x[0][1]*A.x[1][1],x[1][0]*A.x[1][0]+x[1][1]*A.x[1][1]);}

    template<class TMatrix>
    typename ProductTranspose<Matrix,TMatrix>::type
    times_transpose(const TMatrix& A) const
    {return (A*transposed()).transposed();}

    Matrix transpose_times(const Matrix& A) const // 8 mults, 4 adds
    {return Matrix(x[0][0]*A.x[0][0]+x[1][0]*A.x[1][0],x[0][1]*A.x[0][0]+x[1][1]*A.x[1][0],x[0][0]*A.x[0][1]+x[1][0]*A.x[1][1],x[0][1]*A.x[0][1]+x[1][1]*A.x[1][1]);}

    template<class TMatrix>
    typename TransposeProduct<Matrix,TMatrix>::type
    transpose_times(const TMatrix& A) const
    {return transposed()*A;}

    UpperTriangularMatrix<T,2> R_from_QR_factorization() const
    {if(x[1][0]==0) return UpperTriangularMatrix<T,2>(x[0][0],x[0][1],x[1][1]);T c,s;
    if(abs(x[1][0])>abs(x[0][0])){T t=-x[0][0]/x[1][0];s=1/sqrt(1+t*t);c=s*t;}
    else{T t=-x[1][0]/x[0][0];c=1/sqrt(1+t*t);s=c*t;}
    return UpperTriangularMatrix<T,2>(c*x[0][0]-s*x[1][0],c*x[0][1]-s*x[1][1],s*x[0][1]+c*x[1][1]);}

    static T determinant_differential(const Matrix& A,const Matrix& dA)
    {return dA.x[0][0]*A.x[1][1]+A.x[0][0]*dA.x[1][1]-dA.x[1][0]*A.x[0][1]-A.x[1][0]*dA.x[0][1];}

    static Matrix cofactor_differential(const Matrix& dA)
    {return dA.cofactor_matrix();}

    T simplex_minimum_altitude() const
    {return determinant()/sqrt(max(column(0).sqr_magnitude(),column(1).sqr_magnitude(),(column(0)-column(1)).sqr_magnitude()));}

    GEODE_CORE_EXPORT void indefinite_polar_decomposition(Matrix& Q,SymmetricMatrix<T,2>& S) const ;
    GEODE_CORE_EXPORT void fast_singular_value_decomposition(Matrix& U,DiagonalMatrix<T,2>& singular_values,Matrix& V) const ;
};

// global functions
template<class T>
inline Matrix<T,2> operator+(const T a,const Matrix<T,2>& A)
{return A+a;}

template<class T>
inline Matrix<T,2> operator-(const T a,const Matrix<T,2>& A)
{return -A+a;}

template<class T>
inline Matrix<T,2> operator+(const SymmetricMatrix<T,2>& A,const Matrix<T,2>& B)
{return Matrix<T,2>(A.x00+B.x[0][0],A.x10+B.x[1][0],A.x10+B.x[0][1],A.x11+B.x[1][1]);}

template<class T>
inline Matrix<T,2> operator-(const SymmetricMatrix<T,2>& A,const Matrix<T,2>& B)
{return Matrix<T,2>(A.x00-B.x[0][0],A.x10-B.x[1][0],A.x10-B.x[0][1],A.x11-B.x[1][1]);}

template<class T>
inline Matrix<T,2> operator+(const UpperTriangularMatrix<T,2>& A,const Matrix<T,2>& B)
{return Matrix<T,2>(A.x00+B.x[0][0],B.x[1][0],A.x01+B.x[0][1],A.x11+B.x[1][1]);}

template<class T>
inline Matrix<T,2> operator-(const UpperTriangularMatrix<T,2>& A,const Matrix<T,2>& B)
{return Matrix<T,2>(A.x00-B.x[0][0],-B.x[1][0],A.x01-B.x[0][1],A.x11-B.x[1][1]);}

template<class T>
inline Matrix<T,2> operator*(const T a,const Matrix<T,2>& A)
{return Matrix<T,2>(a*A.x[0][0],a*A.x[1][0],a*A.x[0][1],a*A.x[1][1]);}

template<class T>
inline Vector<T,2> operator*(const Vector<T,2>& v,const Matrix<T,2>& A)
{return Vector<T,2> (v.x*A.x[0][0]+v.y*A.x[0][1],v.x*A.x[1][0]+v.y*A.x[1][1]);}

template<class T>
inline Matrix<T,2> operator*(const DiagonalMatrix<T,2>& A,const Matrix<T,2>& B)
{return Matrix<T,2>(A.x00*B.x[0][0],A.x11*B.x[1][0],A.x00*B.x[0][1],A.x11*B.x[1][1]);}

template<class T>
inline Matrix<T,2> operator*(const UpperTriangularMatrix<T,2>& A,const Matrix<T,2>& B)
{return Matrix<T,2>(A.x00*B.x[0][0]+A.x01*B.x[1][0],A.x11*B.x[1][0],A.x00*B.x[0][1]+A.x01*B.x[1][1],A.x11*B.x[1][1]);}

template<class T> inline Matrix<T,2>
outer_product(const Vector<T,2>& u,const Vector<T,2>& v)
{return Matrix<T,2>(u.x*v.x,u.y*v.x,u.x*v.y,u.y*v.y);}

template<class T> inline T inner_product(const Matrix<T,2>& A,const Matrix<T,2>& B) {
  return A.x[0][0]*B.x[0][0]+A.x[1][0]*B.x[1][0]+A.x[0][1]*B.x[0][1]+A.x[1][1]*B.x[1][1];
}

template<class T> inline SymmetricMatrix<T,2> symmetric_part(const Matrix<T,2>& A)
{return SymmetricMatrix<T,2>(A.x[0][0],(T).5*(A.x[1][0]+A.x[0][1]),A.x[1][1]);}

template<class T> inline SymmetricMatrix<T,2> twice_symmetric_part(const Matrix<T,2>& A)
{return SymmetricMatrix<T,2>(2*A.x[0][0],A.x[1][0]+A.x[0][1],2*A.x[1][1]);}

template<class T> inline SymmetricMatrix<T,2> assume_symmetric(const Matrix<T,2>& A)
{return SymmetricMatrix<T,2>(A.x[0][0],A.x[0][1],A.x[1][1]);}

template<class T> inline DiagonalMatrix<T,2> diagonal_part(const Matrix<T,2>& A)
{return DiagonalMatrix<T,2>(A.x[0][0],A.x[1][1]);}

template<class T> inline T antisymmetric_part_cross_product_vector(const Matrix<T,2>& A)
{return (T).5*(A.x[1][0]-A.x[0][1]);}

template<class T>
inline std::istream& operator>>(std::istream& input_stream,Matrix<T,2>& A)
{for(int i=0;i<2;i++) for(int j=0;j<2;j++) input_stream>>A.x[i][j];return input_stream;}

}
