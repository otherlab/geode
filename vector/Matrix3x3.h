//#####################################################################
// Class Matrix<T,3,3>
//#####################################################################
#pragma once

#include <other/core/vector/Vector3d.h>
#include <other/core/vector/ArithmeticPolicy.h>
#include <other/core/math/min.h>
#include <other/core/math/max.h>
#include <other/core/math/sqr.h>
#include <cfloat>
namespace other {

template<class T>
class Matrix<T,3>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=3};
    enum Workaround2 {n=3};
    static const bool is_const=false;

    T x[3][3];

    Matrix()
    {
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]=T();
    }

    template<class T2> explicit
    Matrix(const Matrix<T2,3>& matrix)
    {
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]=(T)matrix.x[i][j];
    }

    Matrix(const DiagonalMatrix<T,3>& matrix)
    {
        x[0][0]=matrix.x00;x[1][1]=matrix.x11;x[2][2]=matrix.x22;x[1][0]=x[2][0]=x[0][1]=x[2][1]=x[0][2]=x[1][2]=0;
    }

    Matrix(const SymmetricMatrix<T,3>& matrix)
    {
        x[0][0]=matrix.x00;x[1][0]=x[0][1]=matrix.x10;x[2][0]=x[0][2]=matrix.x20;x[1][1]=matrix.x11;x[2][1]=x[1][2]=matrix.x21;x[2][2]=matrix.x22;
    }

    Matrix(const UpperTriangularMatrix<T,3>& matrix)
    {
        x[0][0]=matrix.x00;x[0][1]=matrix.x01;x[1][1]=matrix.x11;x[0][2]=matrix.x02;x[1][2]=matrix.x12;x[2][2]=matrix.x22;x[1][0]=x[2][0]=x[2][1]=0;
    }

    Matrix(const T x00,const T x10,const T x20,const T x01,const T x11,const T x21,const T x02,const T x12,const T x22)
    {
        x[0][0]=x00;x[1][0]=x10;x[2][0]=x20;x[0][1]=x01;x[1][1]=x11;x[2][1]=x21;x[0][2]=x02;x[1][2]=x12;x[2][2]=x22;
    }

    Matrix(const Vector<T,3>& column0,const Vector<T,3>& column1,const Vector<T,3>& column2)
    {
        x[0][0]=column0.x;x[1][0]=column0.y;x[2][0]=column0.z;x[0][1]=column1.x;x[1][1]=column1.y;x[2][1]=column1.z;x[0][2]=column2.x;x[1][2]=column2.y;x[2][2]=column2.z;
    }

    explicit Matrix(const T matrix[3][3])
    {
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]=matrix[i][j];
    }

    Matrix& operator=(const Matrix& matrix)
    {
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]=matrix.x[i][j];return *this;
    }

    void copy(const Matrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(3,3);}

    int rows() const
    {return 3;}

    int columns() const
    {return 3;}

    T& operator()(const int i,const int j)
    {assert(unsigned(i)<3 && unsigned(j)<3);return x[i][j];}

    const T& operator()(const int i,const int j) const
    {assert(unsigned(i)<3 && unsigned(j)<3);return x[i][j];}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<3 && unsigned(j)<3;}

    Vector<T,3>& operator[](const int i)
    {assert(unsigned(i)<3);return *(Vector<T,3>*)x[i];}

    const Vector<T,3>& operator[](const int i) const
    {assert(unsigned(i)<3);return *(const Vector<T,3>*)x[i];}

    const Vector<T,3> column(const int j) const
    {assert(unsigned(j)<3);return Vector<T,3>(x[0][j],x[1][j],x[2][j]);}

    void set_column(const int j,const Vector<T,3>& c)
    {assert(unsigned(j)<3);x[0][j]=c[0];x[1][j]=c[1];x[2][j]=c[2];}

    bool operator==(const Matrix& A) const
    {for(int i=0;i<3;i++) for(int j=0;j<3;j++) if(x[i][j]!=A.x[i][j]) return false;return true;}

    bool operator!=(const Matrix& A) const
    {return !(*this==A);}

    Vector<T,3> column_sum() const
    {return Vector<T,3>(x[0][0]+x[0][1]+x[0][2],x[1][0]+x[1][1]+x[1][2],x[2][0]+x[2][1]+x[2][2]);}

    static Matrix componentwise_min(const Matrix& v1,const Matrix& v2)
    {return Matrix(min(v1.x[0][0],v2.x[0][0]),min(v1.x[1][0],v2.x[1][0]),min(v1.x[2][0],v2.x[2][0]),min(v1.x[0][1],v2.x[0][1]),min(v1.x[1][1],v2.x[1][1]),min(v1.x[2][1],v2.x[2][1]),
        min(v1.x[0][2],v2.x[0][2]),min(v1.x[1][2],v2.x[1][2]),min(v1.x[2][2],v2.x[2][2]));}

    static Matrix componentwise_max(const Matrix& v1,const Matrix& v2)
    {return Matrix(max(v1.x[0][0],v2.x[0][0]),max(v1.x[1][0],v2.x[1][0]),max(v1.x[2][0],v2.x[2][0]),max(v1.x[0][1],v2.x[0][1]),max(v1.x[1][1],v2.x[1][1]),max(v1.x[2][1],v2.x[2][1]),
        max(v1.x[0][2],v2.x[0][2]),max(v1.x[1][2],v2.x[1][2]),max(v1.x[2][2],v2.x[2][2]));}

    Matrix operator-() const
    {return Matrix(-x[0][0],-x[1][0],-x[2][0],-x[0][1],-x[1][1],-x[2][1],-x[0][2],-x[1][2],-x[2][2]);}

    Matrix& operator+=(const Matrix& A)
    {for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]+=A.x[i][j];return *this;}

    Matrix& operator+=(const T& a)
    {x[0][0]+=a;x[1][1]+=a;x[2][2]+=a;return *this;}

    Matrix& operator-=(const Matrix& A)
    {for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]-=A.x[i][j];return *this;}

    Matrix& operator-=(const T& a)
    {x[0][0]-=a;x[1][1]-=a;x[2][2]-=a;return *this;}

    Matrix& operator*=(const Matrix& A)
    {return *this=*this*A;}

    Matrix& operator*=(const T a)
    {for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]*=a;return *this;}

    Matrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;for(int i=0;i<3;i++) for(int j=0;j<3;j++) x[i][j]*=s;return *this;}

    Matrix operator+(const Matrix& A) const
    {return Matrix(x[0][0]+A.x[0][0],x[1][0]+A.x[1][0],x[2][0]+A.x[2][0],x[0][1]+A.x[0][1],x[1][1]+A.x[1][1],x[2][1]+A.x[2][1],x[0][2]+A.x[0][2],x[1][2]+A.x[1][2],x[2][2]+A.x[2][2]);}

    Matrix operator+(const T a) const
    {return Matrix(x[0][0]+a,x[1][0],x[2][0],x[0][1],x[1][1]+a,x[2][1],x[0][2],x[1][2],x[2][2]+a);}

    Matrix operator-(const Matrix& A) const
    {return Matrix(x[0][0]-A.x[0][0],x[1][0]-A.x[1][0],x[2][0]-A.x[2][0],x[0][1]-A.x[0][1],x[1][1]-A.x[1][1],x[2][1]-A.x[2][1],x[0][2]-A.x[0][2],x[1][2]-A.x[1][2],x[2][2]-A.x[2][2]);}

    Matrix operator-(const T a) const
    {return Matrix(x[0][0]-a,x[1][0],x[2][0],x[0][1],x[1][1]-a,x[2][1],x[0][2],x[1][2],x[2][2]-a);}

    Matrix operator*(const Matrix& A) const // 27 mults, 18 adds
    {return Matrix(x[0][0]*A.x[0][0]+x[0][1]*A.x[1][0]+x[0][2]*A.x[2][0],x[1][0]*A.x[0][0]+x[1][1]*A.x[1][0]+x[1][2]*A.x[2][0],x[2][0]*A.x[0][0]+x[2][1]*A.x[1][0]+x[2][2]*A.x[2][0],
        x[0][0]*A.x[0][1]+x[0][1]*A.x[1][1]+x[0][2]*A.x[2][1],x[1][0]*A.x[0][1]+x[1][1]*A.x[1][1]+x[1][2]*A.x[2][1],x[2][0]*A.x[0][1]+x[2][1]*A.x[1][1]+x[2][2]*A.x[2][1],
        x[0][0]*A.x[0][2]+x[0][1]*A.x[1][2]+x[0][2]*A.x[2][2],x[1][0]*A.x[0][2]+x[1][1]*A.x[1][2]+x[1][2]*A.x[2][2],x[2][0]*A.x[0][2]+x[2][1]*A.x[1][2]+x[2][2]*A.x[2][2]);}

    template<int p> Matrix<T,3,p>
    operator*(const Matrix<T,3,p>& A) const
    {Matrix<T,3,p> matrix;for(int i=0;i<3;i++) for(int j=0;j<p;j++) for(int k=0;k<3;k++) matrix(i,j)+=(*this)(i,k)*A(k,j);return matrix;}

    Matrix operator*(const T a) const
    {return Matrix(a*x[0][0],a*x[1][0],a*x[2][0],a*x[0][1],a*x[1][1],a*x[2][1],a*x[0][2],a*x[1][2],a*x[2][2]);}

    Matrix operator/(const T a) const
    {assert(a!=0);T s=1/a;return Matrix(s*x[0][0],s*x[1][0],s*x[2][0],s*x[0][1],s*x[1][1],s*x[2][1],s*x[0][2],s*x[1][2],s*x[2][2]);}

    Vector<T,3> operator*(const Vector<T,3>& v) const // 9 mults, 6 adds
    {return Vector<T,3>(x[0][0]*v.x+x[0][1]*v.y+x[0][2]*v.z,x[1][0]*v.x+x[1][1]*v.y+x[1][2]*v.z,x[2][0]*v.x+x[2][1]*v.y+x[2][2]*v.z);}

    Vector<T,2> homogeneous_times(const Vector<T,2>& v) const // assumes w=1 is the 3rd coordinate of v
    {T w=x[2][0]*v.x+x[2][1]*v.y+x[2][2];assert(w!=0);
    T s=1/w;return Vector<T,2>(s*(x[0][0]*v.x+x[0][1]*v.y+x[0][2]),s*(x[1][0]*v.x+x[1][1]*v.y+x[1][2]));} // rescale so w=1

    Vector<T,2> Transform_2X2(const Vector<T,2>& v) const // multiplies vector by upper 2x2 of matrix only
    {return Vector<T,2>(x[0][0]*v.x+x[0][1]*v.y,x[1][0]*v.x+x[1][1]*v.y);}

    T determinant() const // 9 mults, 5 adds
    {return x[0][0]*(x[1][1]*x[2][2]-x[1][2]*x[2][1])+x[0][1]*(x[1][2]*x[2][0]-x[1][0]*x[2][2])+x[0][2]*(x[1][0]*x[2][1]-x[1][1]*x[2][0]);}

    T parallelepiped_measure() const
    {return determinant();}

    void invert()
    {*this=inverse();}

    Matrix inverse() const
    {T cofactor11=x[1][1]*x[2][2]-x[1][2]*x[2][1],cofactor12=x[1][2]*x[2][0]-x[1][0]*x[2][2],cofactor13=x[1][0]*x[2][1]-x[1][1]*x[2][0];
    T determinant=x[0][0]*cofactor11+x[0][1]*cofactor12+x[0][2]*cofactor13;assert(determinant!=0);T s=1/determinant;
    return s*Matrix(cofactor11,cofactor12,cofactor13,x[0][2]*x[2][1]-x[0][1]*x[2][2],x[0][0]*x[2][2]-x[0][2]*x[2][0],x[0][1]*x[2][0]-x[0][0]*x[2][1],x[0][1]*x[1][2]-x[0][2]*x[1][1],x[0][2]*x[1][0]-x[0][0]*x[1][2],x[0][0]*x[1][1]-x[0][1]*x[1][0]);}

    Matrix inverse_transposed() const
    {return inverse().transposed();}

    Vector<T,3> solve_linear_system(const Vector<T,3>& b) const // 33 mults, 17 adds, 1 div
    {T cofactor11=x[1][1]*x[2][2]-x[1][2]*x[2][1],cofactor12=x[1][2]*x[2][0]-x[1][0]*x[2][2],cofactor13=x[1][0]*x[2][1]-x[1][1]*x[2][0];
    T determinant=x[0][0]*cofactor11+x[0][1]*cofactor12+x[0][2]*cofactor13;assert(determinant!=0);
    return Matrix(cofactor11,cofactor12,cofactor13,x[0][2]*x[2][1]-x[0][1]*x[2][2],x[0][0]*x[2][2]-x[0][2]*x[2][0],x[0][1]*x[2][0]-x[0][0]*x[2][1],x[0][1]*x[1][2]-x[0][2]*x[1][1],x[0][2]*x[1][0]-x[0][0]*x[1][2],x[0][0]*x[1][1]-x[0][1]*x[1][0])*b/determinant;}

    Vector<T,3> robust_solve_linear_system(const Vector<T,3>& b) const // 34 mults, 17 adds, 1 div
    {T cofactor11=x[1][1]*x[2][2]-x[1][2]*x[2][1],cofactor12=x[1][2]*x[2][0]-x[1][0]*x[2][2],cofactor13=x[1][0]*x[2][1]-x[1][1]*x[2][0];
    T determinant=x[0][0]*cofactor11+x[0][1]*cofactor12+x[0][2]*cofactor13;
    Vector<T,3> unscaled_result=Matrix(cofactor11,cofactor12,cofactor13,x[0][2]*x[2][1]-x[0][1]*x[2][2],x[0][0]*x[2][2]-x[0][2]*x[2][0],x[0][1]*x[2][0]-x[0][0]*x[2][1],x[0][1]*x[1][2]-x[0][2]*x[1][1],x[0][2]*x[1][0]-x[0][0]*x[1][2],
        x[0][0]*x[1][1]-x[0][1]*x[1][0])*b;
    T relative_tolerance=(T)FLT_MIN*unscaled_result.maxabs();
    if(abs(determinant)<=relative_tolerance){relative_tolerance=max(relative_tolerance,(T)FLT_MIN);determinant=determinant>=0?relative_tolerance:-relative_tolerance;}
    return unscaled_result/determinant;}

    Matrix rotation_only() const
    {return Matrix(x[0][0],x[1][0],0,x[0][1],x[1][1],0,0,0,1);}

    const Vector<T,2> translation() const
    {return Vector<T,2>(x[0][2],x[1][2]);}

    void set_translation(const Vector<T,2>& t)
    {x[0][2]=t[0];x[1][2]=t[1];}

    Matrix<T,2> extract_rotation() const
    {return Matrix<T,2>(x[0][0],x[1][0],x[0][1],x[1][1]);}

    static Matrix from_linear(const Matrix<T,2>& M) // Create a homogeneous 3x3 matrix corresponding to a 2x2 transform
    {return Matrix(M.x[0][0],M.x[1][0],0,M.x[0][1],M.x[1][1],0,0,0,1);}

    void transpose()
    {swap(x[1][0],x[0][1]);swap(x[2][0],x[0][2]);swap(x[2][1],x[1][2]);}

    Matrix transposed() const
    {return Matrix(x[0][0],x[0][1],x[0][2],x[1][0],x[1][1],x[1][2],x[2][0],x[2][1],x[2][2]);}

    T trace() const
    {return x[0][0]+x[1][1]+x[2][2];}

    Matrix Q_From_QR_Factorization() const // Gram Schmidt
    {int k;Matrix Q=*this;
    T one_over_r11=1/sqrt((sqr(Q.x[0][0])+sqr(Q.x[1][0])+sqr(Q.x[2][0])));for(k=0;k<3;k++) Q.x[k][0]*=one_over_r11;
    T r12=Q.x[0][0]*Q.x[0][1]+Q.x[1][0]*Q.x[1][1]+Q.x[2][0]*Q.x[2][1];Q.x[0][1]-=r12*Q.x[0][0];Q.x[1][1]-=r12*Q.x[1][0];Q.x[2][1]-=r12*Q.x[2][0];
    T r13=Q.x[0][0]*Q.x[0][2]+Q.x[1][0]*Q.x[1][2]+Q.x[2][0]*Q.x[2][2];Q.x[0][2]-=r13*Q.x[0][0];Q.x[1][2]-=r13*Q.x[1][0];Q.x[2][2]-=r13*Q.x[2][0];
    T one_over_r22=1/sqrt((sqr(Q.x[0][1])+sqr(Q.x[1][1])+sqr(Q.x[2][1])));for(k=0;k<3;k++) Q.x[k][1]*=one_over_r22;
    T r23=Q.x[0][1]*Q.x[0][2]+Q.x[1][1]*Q.x[1][2]+Q.x[2][1]*Q.x[2][2];Q.x[0][2]-=r23*Q.x[0][1];Q.x[1][2]-=r23*Q.x[1][1];Q.x[2][2]-=r23*Q.x[2][1];
    T one_over_r33=1/sqrt((sqr(Q.x[0][2])+sqr(Q.x[1][2])+sqr(Q.x[2][2])));for(k=0;k<3;k++) Q.x[k][2]*=one_over_r33;
    return Q;}

    UpperTriangularMatrix<T,3> R_from_QR_factorization() const // Gram Schmidt
    {int k;Matrix Q=*this;UpperTriangularMatrix<T,3> R;
    R.x00=sqrt((sqr(Q.x[0][0])+sqr(Q.x[1][0])+sqr(Q.x[2][0])));T one_over_r11=1/R.x00;for(k=0;k<3;k++) Q.x[k][0]*=one_over_r11;
    R.x01=Q.x[0][0]*Q.x[0][1]+Q.x[1][0]*Q.x[1][1]+Q.x[2][0]*Q.x[2][1];Q.x[0][1]-=R.x01*Q.x[0][0];Q.x[1][1]-=R.x01*Q.x[1][0];Q.x[2][1]-=R.x01*Q.x[2][0];
    R.x02=Q.x[0][0]*Q.x[0][2]+Q.x[1][0]*Q.x[1][2]+Q.x[2][0]*Q.x[2][2];Q.x[0][2]-=R.x02*Q.x[0][0];Q.x[1][2]-=R.x02*Q.x[1][0];Q.x[2][2]-=R.x02*Q.x[2][0];
    R.x11=sqrt((sqr(Q.x[0][1])+sqr(Q.x[1][1])+sqr(Q.x[2][1])));T one_over_r22=1/R.x11;for(k=0;k<3;k++) Q.x[k][1]*=one_over_r22;
    R.x12=Q.x[0][1]*Q.x[0][2]+Q.x[1][1]*Q.x[1][2]+Q.x[2][1]*Q.x[2][2];Q.x[0][2]-=R.x12*Q.x[0][1];Q.x[1][2]-=R.x12*Q.x[1][1];Q.x[2][2]-=R.x12*Q.x[2][1];
    R.x22=sqrt((sqr(Q.x[0][2])+sqr(Q.x[1][2])+sqr(Q.x[2][2])));
    return R;}

    Matrix cofactor_matrix() const // 18 mults, 9 adds
    {return Matrix(x[1][1]*x[2][2]-x[2][1]*x[1][2],x[2][1]*x[0][2]-x[0][1]*x[2][2],x[0][1]*x[1][2]-x[1][1]*x[0][2],
                   x[2][0]*x[1][2]-x[1][0]*x[2][2],x[0][0]*x[2][2]-x[2][0]*x[0][2],x[1][0]*x[0][2]-x[0][0]*x[1][2],
                   x[1][0]*x[2][1]-x[2][0]*x[1][1],x[2][0]*x[0][1]-x[0][0]*x[2][1],x[0][0]*x[1][1]-x[1][0]*x[0][1]);}

    SymmetricMatrix<T,3> outer_product_matrix() const // 18 mults, 12 adds
    {return SymmetricMatrix<T,3>(x[0][0]*x[0][0]+x[0][1]*x[0][1]+x[0][2]*x[0][2],x[1][0]*x[0][0]+x[1][1]*x[0][1]+x[1][2]*x[0][2],x[2][0]*x[0][0]+x[2][1]*x[0][1]+x[2][2]*x[0][2],
                                  x[1][0]*x[1][0]+x[1][1]*x[1][1]+x[1][2]*x[1][2],x[2][0]*x[1][0]+x[2][1]*x[1][1]+x[2][2]*x[1][2],x[2][0]*x[2][0]+x[2][1]*x[2][1]+x[2][2]*x[2][2]);}

    SymmetricMatrix<T,3> normal_equations_matrix() const // 18 mults, 12 adds
    {return SymmetricMatrix<T,3>(x[0][0]*x[0][0]+x[1][0]*x[1][0]+x[2][0]*x[2][0],x[0][1]*x[0][0]+x[1][1]*x[1][0]+x[2][1]*x[2][0],x[0][2]*x[0][0]+x[1][2]*x[1][0]+x[2][2]*x[2][0],
                                  x[0][1]*x[0][1]+x[1][1]*x[1][1]+x[2][1]*x[2][1],x[0][2]*x[0][1]+x[1][2]*x[1][1]+x[2][2]*x[2][1],x[0][2]*x[0][2]+x[1][2]*x[1][2]+x[2][2]*x[2][2]);}

    void normalize_columns()
    {T magnitude=sqrt(sqr(x[0][0])+sqr(x[1][0])+sqr(x[2][0]));assert(magnitude!=0);T s=1/magnitude;x[0][0]*=s;x[1][0]*=s;x[2][0]*=s;
    magnitude=sqrt(sqr(x[0][1])+sqr(x[1][1])+sqr(x[2][1]));assert(magnitude!=0);s=1/magnitude;x[0][1]*=s;x[1][1]*=s;x[2][1]*=s;
    magnitude=sqrt(sqr(x[0][2])+sqr(x[1][2])+sqr(x[2][2]));assert(magnitude!=0);s=1/magnitude;x[0][2]*=s;x[1][2]*=s;x[2][2]*=s;}

    Vector<T,3> column_magnitudes() const
    {return Vector<T,3>(column(0).magnitude(),column(1).magnitude(),column(2).magnitude());}

    Vector<T,3> largest_normalized_column() const
    {T scale1=sqr(x[0][0])+sqr(x[1][0])+sqr(x[2][0]),scale2=sqr(x[0][1])+sqr(x[1][1])+sqr(x[2][1]),scale3=sqr(x[0][2])+sqr(x[1][2])+sqr(x[2][2]);
    if(scale1>scale2){if(scale1>scale3)return Vector<T,3>(x[0][0],x[1][0],x[2][0])/sqrt(scale1);}
    else if(scale2>scale3)return Vector<T,3>(x[0][1],x[1][1],x[2][1])/sqrt(scale2);
    return Vector<T,3>(x[0][2],x[1][2],x[2][2])/sqrt(scale3);}

    static Matrix transpose(const Matrix& A)
    {return Matrix(A.x[0][0],A.x[0][1],A.x[0][2],A.x[1][0],A.x[1][1],A.x[1][2],A.x[2][0],A.x[2][1],A.x[2][2]);}

    static Matrix translation_matrix(const Vector<T,2>& translation) // treating the 3x3 matrix as a homogeneous transformation on 2d vectors
    {return Matrix(1,0,0,0,1,0,translation.x,translation.y,1);}

    static Matrix identity_matrix()
    {return Matrix(1,0,0,0,1,0,0,0,1);}

    static Matrix rotation_matrix_x_axis(const T radians)
    {T c=cos(radians),s=sin(radians);return Matrix(1,0,0,0,c,s,0,-s,c);}

    static Matrix rotation_matrix_y_axis(const T radians)
    {T c=cos(radians),s=sin(radians);return Matrix(c,0,-s,0,1,0,s,0,c);}

    static Matrix rotation_matrix_z_axis(const T radians)
    {T c=cos(radians),s=sin(radians);return Matrix(c,s,0,-s,c,0,0,0,1);}

    static Matrix rotation_matrix(const Vector<T,3>& axis,const T radians)
    {return Rotation<Vector<T,3> >(radians,axis).matrix();}

    static Matrix rotation_matrix(const Vector<T,3>& rotation)
    {return Rotation<Vector<T,3> >::from_rotation_vector(rotation).matrix();}

    static Matrix rotation_matrix(const Vector<T,3>& x_final,const Vector<T,3>& y_final,const Vector<T,3>& z_final)
    {return Matrix(x_final,y_final,z_final);}

    static Matrix rotation_matrix(const Vector<T,3>& initial_vector,const Vector<T,3>& final_vector)
    {return Rotation<Vector<T,3> >::from_rotated_vector(initial_vector,final_vector).matrix();}

    static Matrix scale_matrix(const Vector<T,2>& scale_vector)
    {return Matrix(scale_vector.x,0,0,0,scale_vector.y,0,0,0,1);}

    static Matrix scale_matrix(const T scale)
    {return Matrix(scale,0,0,0,scale,0,0,0,1);}

    static SymmetricMatrix<T,3> right_multiply_with_symmetric_result(const Matrix& A,const DiagonalMatrix<T,3>& B)
    {return SymmetricMatrix<T,3>(B.x00*A.x[0][0],B.x00*A.x[1][0],B.x00*A.x[2][0],B.x11*A.x[1][1],B.x11*A.x[2][1],B.x22*A.x[2][2]);}

    T maxabs() const
    {return other::maxabs(x[0][0],x[1][0],x[2][0],x[0][1],x[1][1],x[2][1],x[0][2],x[1][2],x[2][2]);}

    T sqr_frobenius_norm() const
    {return sqr(x[0][0])+sqr(x[1][0])+sqr(x[2][0])+sqr(x[0][1])+sqr(x[1][1])+sqr(x[2][1])+sqr(x[0][2])+sqr(x[1][2])+sqr(x[2][2]);}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    Matrix operator*(const DiagonalMatrix<T,3>& A) const // 9 mults
    {return Matrix(x[0][0]*A.x00,x[1][0]*A.x00,x[2][0]*A.x00,x[0][1]*A.x11,x[1][1]*A.x11,x[2][1]*A.x11,x[0][2]*A.x22,x[1][2]*A.x22,x[2][2]*A.x22);}

    Matrix operator*(const UpperTriangularMatrix<T,3>& A) const // 18 mults, 9 adds
    {return Matrix(x[0][0]*A.x00,x[1][0]*A.x00,x[2][0]*A.x00,x[0][0]*A.x01+x[0][1]*A.x11,x[1][0]*A.x01+x[1][1]*A.x11,x[2][0]*A.x01+x[2][1]*A.x11,
                          x[0][0]*A.x02+x[0][1]*A.x12+x[0][2]*A.x22,x[1][0]*A.x02+x[1][1]*A.x12+x[1][2]*A.x22,x[2][0]*A.x02+x[2][1]*A.x12+x[2][2]*A.x22);}

    Matrix operator*(const SymmetricMatrix<T,3>& A) const // 27 mults, 18 adds
    {return Matrix(x[0][0]*A.x00+x[0][1]*A.x10+x[0][2]*A.x20,x[1][0]*A.x00+x[1][1]*A.x10+x[1][2]*A.x20,x[2][0]*A.x00+x[2][1]*A.x10+x[2][2]*A.x20,
                          x[0][0]*A.x10+x[0][1]*A.x11+x[0][2]*A.x21,x[1][0]*A.x10+x[1][1]*A.x11+x[1][2]*A.x21,x[2][0]*A.x10+x[2][1]*A.x11+x[2][2]*A.x21,
                          x[0][0]*A.x20+x[0][1]*A.x21+x[0][2]*A.x22,x[1][0]*A.x20+x[1][1]*A.x21+x[1][2]*A.x22,x[2][0]*A.x20+x[2][1]*A.x21+x[2][2]*A.x22);}

    Matrix times_transpose(const UpperTriangularMatrix<T,3>& A) const
    {return Matrix(x[0][0]*A.x00+x[0][1]*A.x01+x[0][2]*A.x02,x[1][0]*A.x00+x[1][1]*A.x01+x[1][2]*A.x02,x[2][0]*A.x00+x[2][1]*A.x01+x[2][2]*A.x02,
                          x[0][1]*A.x11+x[0][2]*A.x12,x[1][1]*A.x11+x[1][2]*A.x12,x[2][1]*A.x11+x[2][2]*A.x12,x[0][2]*A.x22,x[1][2]*A.x22,x[2][2]*A.x22);}

    Matrix transpose_times(const Matrix& A) const
    {return Matrix(x[0][0]*A.x[0][0]+x[1][0]*A.x[1][0]+x[2][0]*A.x[2][0],x[0][1]*A.x[0][0]+x[1][1]*A.x[1][0]+x[2][1]*A.x[2][0],x[0][2]*A.x[0][0]+x[1][2]*A.x[1][0]+x[2][2]*A.x[2][0],
                   x[0][0]*A.x[0][1]+x[1][0]*A.x[1][1]+x[2][0]*A.x[2][1],x[0][1]*A.x[0][1]+x[1][1]*A.x[1][1]+x[2][1]*A.x[2][1],x[0][2]*A.x[0][1]+x[1][2]*A.x[1][1]+x[2][2]*A.x[2][1],
                   x[0][0]*A.x[0][2]+x[1][0]*A.x[1][2]+x[2][0]*A.x[2][2],x[0][1]*A.x[0][2]+x[1][1]*A.x[1][2]+x[2][1]*A.x[2][2],x[0][2]*A.x[0][2]+x[1][2]*A.x[1][2]+x[2][2]*A.x[2][2]);}

    template<class TMatrix>
    typename TransposeProduct<Matrix,TMatrix>::type
    transpose_times(const TMatrix& A) const
    {return transposed()*A;}

    Vector<T,3> transpose_times(const Vector<T,3>& v) const
    {return Vector<T,3>(x[0][0]*v.x+x[1][0]*v.y+x[2][0]*v.z,x[0][1]*v.x+x[1][1]*v.y+x[2][1]*v.z,x[0][2]*v.x+x[1][2]*v.y+x[2][2]*v.z);}

    Matrix times_transpose(const Matrix& A) const
    {return Matrix(x[0][0]*A.x[0][0]+x[0][1]*A.x[0][1]+x[0][2]*A.x[0][2],x[1][0]*A.x[0][0]+x[1][1]*A.x[0][1]+x[1][2]*A.x[0][2],x[2][0]*A.x[0][0]+x[2][1]*A.x[0][1]+x[2][2]*A.x[0][2],
                   x[0][0]*A.x[1][0]+x[0][1]*A.x[1][1]+x[0][2]*A.x[1][2],x[1][0]*A.x[1][0]+x[1][1]*A.x[1][1]+x[1][2]*A.x[1][2],x[2][0]*A.x[1][0]+x[2][1]*A.x[1][1]+x[2][2]*A.x[1][2],
                   x[0][0]*A.x[2][0]+x[0][1]*A.x[2][1]+x[0][2]*A.x[2][2],x[1][0]*A.x[2][0]+x[1][1]*A.x[2][1]+x[1][2]*A.x[2][2],x[2][0]*A.x[2][0]+x[2][1]*A.x[2][1]+x[2][2]*A.x[2][2]);}

    template<class TMatrix>
    typename ProductTranspose<Matrix,TMatrix>::type
    times_transpose(const TMatrix& A) const
    {return *this*A.transposed();}

    Matrix cross_product_matrix_times(const Vector<T,3>& v) const // (v*) * (*this)
    {return Matrix(-v.z*x[1][0]+v.y*x[2][0],v.z*x[0][0]-v.x*x[2][0],-v.y*x[0][0]+v.x*x[1][0],
                   -v.z*x[1][1]+v.y*x[2][1],v.z*x[0][1]-v.x*x[2][1],-v.y*x[0][1]+v.x*x[1][1],
                   -v.z*x[1][2]+v.y*x[2][2],v.z*x[0][2]-v.x*x[2][2],-v.y*x[0][2]+v.x*x[1][2]);}

    Matrix cross_product_matrix_transpose_times(const Vector<T,3>& v) const // (v*)^T * (*this)
    {return cross_product_matrix_times(-v);}

    Matrix times_cross_product_matrix(const Vector<T,3>& v) const // (*this) * (v*)
    {return Matrix( x[0][1]*v.z-x[0][2]*v.y, x[1][1]*v.z-x[1][2]*v.y, x[2][1]*v.z-x[2][2]*v.y,
                   -x[0][0]*v.z+x[0][2]*v.x,-x[1][0]*v.z+x[1][2]*v.x,-x[2][0]*v.z+x[2][2]*v.x,
                    x[0][0]*v.y-x[0][1]*v.x, x[1][0]*v.y-x[1][1]*v.x, x[2][0]*v.y-x[2][1]*v.x);}

    Matrix times_cross_product_matrix_transpose(const Vector<T,3>& v) const // (*this) * (v*)^T
    {return times_cross_product_matrix(-v);}

    SymmetricMatrix<T,3> cross_product_matrix_times_with_symmetric_result(const Vector<T,3>& v) const // (v*) * (*this)
    {return SymmetricMatrix<T,3>(-v.z*x[1][0]+v.y*x[2][0],v.z*x[0][0]-v.x*x[2][0],-v.y*x[0][0]+v.x*x[1][0],v.z*x[0][1]-v.x*x[2][1],-v.y*x[0][1]+v.x*x[1][1],-v.y*x[0][2]+v.x*x[1][2]);}

    SymmetricMatrix<T,3> times_cross_product_matrix_with_symmetric_result(const Vector<T,3>& v) const // (*this) * (v*)
    {return SymmetricMatrix<T,3>(x[0][1]*v.z-x[0][2]*v.y,x[1][1]*v.z-x[1][2]*v.y,x[2][1]*v.z-x[2][2]*v.y,-x[1][0]*v.z+x[1][2]*v.x,-x[2][0]*v.z+x[2][2]*v.x,x[2][0]*v.y-x[2][1]*v.x);}

    SymmetricMatrix<T,3> times_cross_product_matrix_transpose_with_symmetric_result(const Vector<T,3>& v) const // (*this) * (v*)^T
    {return times_cross_product_matrix_with_symmetric_result(-v);}

    static Matrix left_procrustes_rotation(const Matrix& A,const Matrix& B)
    {Matrix U,V;DiagonalMatrix<T,3> D;A.times_transpose(B).fast_singular_value_decomposition(U,D,V);return U.times_transpose(V);}

//#####################################################################
    explicit Matrix(RawArray<const T,2> matrix);
    Matrix higham_iterate(const T tolerance=1e-5,const int max_iterations=20,const bool exit_on_max_iterations=false) const;
    void fast_singular_value_decomposition(Matrix<T,3>& U,DiagonalMatrix<T,3>& singular_values,Matrix<T,3>& V) const OTHER_EXPORT;
    void fast_indefinite_polar_decomposition(Matrix<T,3>& Q,SymmetricMatrix<T,3>& S) const OTHER_EXPORT;
    T simplex_minimum_altitude() const OTHER_EXPORT;
//#####################################################################
};
// global functions
template<class T>
inline Matrix<T,3> operator+(const T a,const Matrix<T,3>& A)
{return A+a;}

template<class T>
inline Matrix<T,3> operator*(const T a,const Matrix<T,3>& A)
{return A*a;}

template<class T>
inline Matrix<T,3> operator-(const T a,const Matrix<T,3>& A)
{return Matrix<T,3>(a-A.x[0][0],-A.x[1][0],-A.x[2][0],-A.x[0][1],a-A.x[1][1],-A.x[2][1],-A.x[0][2],-A.x[1][2],a-A.x[2][2]);}

template<class T>
inline Vector<T,3> operator*(const Vector<T,3>& v,const Matrix<T,3>& A)
{return Vector<T,3>(v.x*A.x[0][0]+v.y*A.x[1][0]+v.z*A.x[2][0],v.x*A.x[0][1]+v.y*A.x[1][1]+v.z*A.x[2][1],v.x*A.x[0][2]+v.y*A.x[1][2]+v.z*A.x[2][2]);}

template<class T>
inline Matrix<T,3> operator*(const DiagonalMatrix<T,3>& A,const Matrix<T,3>& B)
{return Matrix<T,3>(A.x00*B.x[0][0],A.x11*B.x[1][0],A.x22*B.x[2][0],A.x00*B.x[0][1],A.x11*B.x[1][1],A.x22*B.x[2][1],A.x00*B.x[0][2],A.x11*B.x[1][2],A.x22*B.x[2][2]);}

template<class T>
inline Matrix<T,3> operator*(const UpperTriangularMatrix<T,3>& A,const Matrix<T,3>& B)
{return Matrix<T,3>(A.x00*B.x[0][0]+A.x01*B.x[1][0]+A.x02*B.x[2][0],A.x11*B.x[1][0]+A.x12*B.x[2][0],A.x22*B.x[2][0],A.x00*B.x[0][1]+A.x01*B.x[1][1]+A.x02*B.x[2][1],
                      A.x11*B.x[1][1]+A.x12*B.x[2][1],A.x22*B.x[2][1],A.x00*B.x[0][2]+A.x01*B.x[1][2]+A.x02*B.x[2][2],A.x11*B.x[1][2]+A.x12*B.x[2][2],A.x22*B.x[2][2]);}

template<class T>
inline Matrix<T,3> operator*(const SymmetricMatrix<T,3>& A,const Matrix<T,3>& B)
{return Matrix<T,3>(A.x00*B.x[0][0]+A.x10*B.x[1][0]+A.x20*B.x[2][0],A.x10*B.x[0][0]+A.x11*B.x[1][0]+A.x21*B.x[2][0],A.x20*B.x[0][0]+A.x21*B.x[1][0]+A.x22*B.x[2][0],
                      A.x00*B.x[0][1]+A.x10*B.x[1][1]+A.x20*B.x[2][1],A.x10*B.x[0][1]+A.x11*B.x[1][1]+A.x21*B.x[2][1],A.x20*B.x[0][1]+A.x21*B.x[1][1]+A.x22*B.x[2][1],
                      A.x00*B.x[0][2]+A.x10*B.x[1][2]+A.x20*B.x[2][2],A.x10*B.x[0][2]+A.x11*B.x[1][2]+A.x21*B.x[2][2],A.x20*B.x[0][2]+A.x21*B.x[1][2]+A.x22*B.x[2][2]);}

template<class T>
inline Matrix<T,3> operator+(const SymmetricMatrix<T,3>& A,const Matrix<T,3>& B)
{return Matrix<T,3>(A.x00+B.x[0][0],A.x10+B.x[1][0],A.x20+B.x[2][0],A.x10+B.x[0][1],A.x11+B.x[1][1],A.x21+B.x[2][1],A.x20+B.x[0][2],A.x21+B.x[1][2],A.x22+B.x[2][2]);}

template<class T>
inline Matrix<T,3> operator-(const SymmetricMatrix<T,3>& A,const Matrix<T,3>& B)
{return Matrix<T,3>(A.x00-B.x[0][0],A.x10-B.x[1][0],A.x20-B.x[2][0],A.x10-B.x[0][1],A.x11-B.x[1][1],A.x21-B.x[2][1],A.x20-B.x[0][2],A.x21-B.x[1][2],A.x22-B.x[2][2]);}

template<class T>
inline Matrix<T,3> operator+(const UpperTriangularMatrix<T,3>& A,const Matrix<T,3>& B)
{return Matrix<T,3>(A.x00+B.x[0][0],B.x[1][0],B.x[2][0],A.x01+B.x[0][1],A.x11+B.x[1][1],B.x[2][1],A.x02+B.x[0][2],A.x12+B.x[1][2],A.x22+B.x[2][2]);}

template<class T>
inline Matrix<T,3> operator-(const UpperTriangularMatrix<T,3>& A,const Matrix<T,3>& B)
{return Matrix<T,3>(A.x00-B.x[0][0],-B.x[1][0],-B.x[2][0],A.x01-B.x[0][1],A.x11-B.x[1][1],-B.x[2][1],A.x02-B.x[0][2],A.x12-B.x[1][2],A.x22-B.x[2][2]);}

template<class T> inline Matrix<T,3>
outer_product(const Vector<T,3>& u,const Vector<T,3>& v)
{return Matrix<T,3>(u.x*v.x,u.y*v.x,u.z*v.x,u.x*v.y,u.y*v.y,u.z*v.y,u.x*v.z,u.y*v.z,u.z*v.z);}

template<class T> inline T inner_product(const Matrix<T,3>& A,const Matrix<T,3>& B) {
  return A.x[0][0]*B.x[0][0]+A.x[1][0]*B.x[1][0]+A.x[2][0]*B.x[2][0]+A.x[0][1]*B.x[0][1]+A.x[1][1]*B.x[1][1]+A.x[2][1]*B.x[2][1]+A.x[0][2]*B.x[0][2]+A.x[1][2]*B.x[1][2]+A.x[2][2]*B.x[2][2];
}

template<class T> inline SymmetricMatrix<T,3> symmetric_part(const Matrix<T,3>& A) // 3 mults, 3 adds
{return SymmetricMatrix<T,3>(A.x[0][0],(T).5*(A.x[1][0]+A.x[0][1]),(T).5*(A.x[2][0]+A.x[0][2]),A.x[1][1],(T).5*(A.x[2][1]+A.x[1][2]),A.x[2][2]);}

template<class T> inline SymmetricMatrix<T,3> twice_symmetric_part(const Matrix<T,3>& A) // 3 mults, 3 adds
{return SymmetricMatrix<T,3>(2*A.x[0][0],A.x[1][0]+A.x[0][1],A.x[2][0]+A.x[0][2],2*A.x[1][1],A.x[2][1]+A.x[1][2],2*A.x[2][2]);}

template<class T> inline SymmetricMatrix<T,3> assume_symmetric(const Matrix<T,3>& A)
{return SymmetricMatrix<T,3>(A.x[0][0],A.x[0][1],A.x[0][2],A.x[1][1],A.x[1][2],A.x[2][2]);}

template<class T> inline DiagonalMatrix<T,3> diagonal_part(const Matrix<T,3>& A)
{return DiagonalMatrix<T,3>(A.x[0][0],A.x[1][1],A.x[2][2]);}

template<class T> inline Vector<T,3> antisymmetric_part_cross_product_vector(const Matrix<T,3>& A)
{return (T).5*twice_antisymmetric_part_cross_product_vector(A);}

template<class T> inline Vector<T,3> twice_antisymmetric_part_cross_product_vector(const Matrix<T,3>& A)
{return Vector<T,3>(A.x[2][1]-A.x[1][2],A.x[0][2]-A.x[2][0],A.x[1][0]-A.x[0][1]);}

template<class T>
inline std::istream& operator>>(std::istream& input,Matrix<T,3>& A)
{for(int i=0;i<3;i++) for(int j=0;j<3;j++) input>>A.x[i][j];return input;}

template<class T>
inline std::ostream& operator<<(std::ostream& output,const Matrix<T,3>& A)
{for(int i=0;i<3;i++){for(int j=0;j<3;j++) output<<A.x[i][j]<<" ";output<<std::endl;}return output;}

}
