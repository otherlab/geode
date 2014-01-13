//#####################################################################
// Class Matrix<T,4>
//#####################################################################
#pragma once

#include <geode/vector/Vector.h>
#include <geode/vector/Matrix3x3.h>
#include <geode/math/sqr.h>
namespace geode {

template<class T>
class Matrix<T,4>
{
public:
    typedef T Scalar;
    static const bool is_const=false;

    T x[4][4];

    Matrix()
    {
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) x[i][j]=T();
    }

    Matrix(const Matrix& matrix)
    {
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) x[i][j]=matrix.x[i][j];
    }

    Matrix(const T x11,const T x21,const T x31,const T x41,const T x12,const T x22,const T x32,const T x42,const T x13,const T x23,const T x33,const T x43,const T x14,const T x24,const T x34,
        const T x44)
    {
        x[0][0]=x11;x[1][0]=x21;x[2][0]=x31;x[3][0]=x41;x[0][1]=x12;x[1][1]=x22;x[2][1]=x32;x[3][1]=x42;x[0][2]=x13;x[1][2]=x23;x[2][2]=x33;x[3][2]=x43;x[0][3]=x14;x[1][3]=x24;x[2][3]=x34;x[3][3]=x44;
    }

    explicit Matrix(RawArray<const T,2>& A)
    {
        assert(A.m==4 && A.n==4);for(int j=0;j<4;j++) for(int i=0;i<4;i++) (*this)(i,j)=A(i,j);
    }

    template<class T2> explicit
    Matrix(const Matrix<T2,4>& matrix)
    {
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) x[i][j]=(T)matrix.x[i][j];
    }

    explicit Matrix(const T matrix[4][4])
    {
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) x[i][j]=matrix[i][j];
    }

    void copy(const Matrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(4,4);}

    int rows() const
    {return 4;}

    int columns() const
    {return 4;}

    T& operator()(const int i,const int j)
    {assert(unsigned(i)<4 && unsigned(j)<4);return x[i][j];}

    const T& operator()(const int i,const int j) const
    {assert(unsigned(i)<4 && unsigned(j)<4);return x[i][j];}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<4 && unsigned(j)<4;}

    Vector<T,4>& operator[](const int i)
    {assert(unsigned(i)<4);return *(Vector<T,4>*)x[i];}

    const Vector<T,4>& operator[](const int i) const
    {assert(unsigned(i)<4);return *(const Vector<T,4>*)x[i];}

    const Vector<T,4> column(const int j) const
    {assert(unsigned(j)<4);return Vector<T,4>(x[0][j],x[1][j],x[2][j],x[3][j]);}

    void set_column(const int j,const Vector<T,4>& c)
    {assert(unsigned(j)<4);x[0][j]=c[0];x[1][j]=c[1];x[2][j]=c[2];x[3][j]=c[3];}

    T* data()
    {return (T*)x;}

    const T* data() const
    {return (T*)x;}

    bool operator==(const Matrix& A) const
    {for(int i=0;i<4;i++) for(int j=0;j<4;j++) if(x[i][j]!=A.x[i][j]) return false;return true;}

    bool operator!=(const Matrix& A) const
    {return !(*this==A);}

    Matrix operator-() const
    {return Matrix(-x[0][0],-x[1][0],-x[2][0],-x[3][0],-x[0][1],-x[1][1],-x[2][1],-x[3][1],-x[0][2],-x[1][2],-x[2][2],-x[3][2],-x[0][3],-x[1][3],-x[2][3],-x[3][3]);}

    Matrix& operator+=(const Matrix& A)
    {for(int i=0;i<16;i++) x[0][i]+=A.x[0][i];return *this;}

    Matrix& operator-=(const Matrix& A)
    {for(int i=0;i<16;i++) x[0][i]-=A.x[0][i];return *this;}

    Matrix& operator*=(const Matrix& A)
    {return *this=*this*A;}

    Matrix& operator*=(const T a)
    {for(int i=0;i<16;i++) x[0][i]*=a;return *this;}

    Matrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;for(int i=0;i<16;i++) x[0][i]*=s;return *this;}

    Matrix operator+(const Matrix& A) const
    {return Matrix(x[0][0]+A.x[0][0],x[1][0]+A.x[1][0],x[2][0]+A.x[2][0],x[3][0]+A.x[3][0],x[0][1]+A.x[0][1],x[1][1]+A.x[1][1],x[2][1]+A.x[2][1],x[3][1]+A.x[3][1],
                   x[0][2]+A.x[0][2],x[1][2]+A.x[1][2],x[2][2]+A.x[2][2],x[3][2]+A.x[3][2],x[0][3]+A.x[0][3],x[1][3]+A.x[1][3],x[2][3]+A.x[2][3],x[3][3]+A.x[3][3]);}

    Matrix operator-(const Matrix& A) const
    {return Matrix(x[0][0]-A.x[0][0],x[1][0]-A.x[1][0],x[2][0]-A.x[2][0],x[3][0]-A.x[3][0],x[0][1]-A.x[0][1],x[1][1]-A.x[1][1],x[2][1]-A.x[2][1],x[3][1]-A.x[3][1],
                   x[0][2]-A.x[0][2],x[1][2]-A.x[1][2],x[2][2]-A.x[2][2],x[3][2]-A.x[3][2],x[0][3]-A.x[0][3],x[1][3]-A.x[1][3],x[2][3]-A.x[2][3],x[3][3]-A.x[3][3]);}

    template<int p>
    typename disable_if_c<p==4,Matrix<T,4,p> >::type
    operator*(const Matrix<T,4,p>& A) const
    {Matrix<T,4,p> result;
    for(int i=0;i<4;i++) for(int j=0;j<p;j++) for(int k=0;k<4;k++) result(i,j)+=x[i][k]*A(k,j);
    return result;}

    Vector<T,4> operator*(const Vector<T,4>& v) const
    {return Vector<T,4>(x[0][0]*v[0]+x[0][1]*v[1]+x[0][2]*v[2]+x[0][3]*v[3],x[1][0]*v[0]+x[1][1]*v[1]+x[1][2]*v[2]+x[1][3]*v[3],
                        x[2][0]*v[0]+x[2][1]*v[1]+x[2][2]*v[2]+x[2][3]*v[3],x[3][0]*v[0]+x[3][1]*v[1]+x[3][2]*v[2]+x[3][3]*v[3]);}

    Matrix operator*(const T a) const
    {return Matrix(a*x[0][0],a*x[1][0],a*x[2][0],a*x[3][0],a*x[0][1],a*x[1][1],a*x[2][1],a*x[3][1],a*x[0][2],a*x[1][2],a*x[2][2],a*x[3][2],a*x[0][3],a*x[1][3],a*x[2][3],a*x[3][3]);}

    template<class TMatrix>
    typename TransposeProduct<Matrix,TMatrix>::type
    transpose_times(const TMatrix& matrix) const
    {return transposed()*matrix;}

    template<class TMatrix>
    typename ProductTranspose<Matrix,TMatrix>::type
    times_transpose(const TMatrix& matrix) const
    {return (matrix*transposed()).transposed();}

    Matrix operator/(const T a) const
    {assert(a!=0);T s=1/a;
    return Matrix(s*x[0][0],s*x[1][0],s*x[2][0],s*x[3][0],s*x[0][1],s*x[1][1],s*x[2][1],s*x[3][1],s*x[0][2],s*x[1][2],s*x[2][2],s*x[3][2],s*x[0][3],s*x[1][3],s*x[2][3],s*x[3][3]);}

    Vector<T,3> homogeneous_times(const Vector<T,3>& v) const // assumes w=1 is the 4th coordinate of v
    {T w=x[3][0]*v.x+x[3][1]*v.y+x[3][2]*v.z+x[3][3];assert(w!=0);
    T s=1/w;// rescale so w=1
    return Vector<T,3>(s*(x[0][0]*v.x+x[0][1]*v.y+x[0][2]*v.z+x[0][3]),s*(x[1][0]*v.x+x[1][1]*v.y+x[1][2]*v.z+x[1][3]),s*(x[2][0]*v.x+x[2][1]*v.y+x[2][2]*v.z+x[2][3]));}

    Vector<T,3> Transform_3X3(const Vector<T,3>& v) const // multiplies vector by upper 3x3 of matrix only
    {return Vector<T,3>(x[0][0]*v.x+x[0][1]*v.y+x[0][2]*v.z,x[1][0]*v.x+x[1][1]*v.y+x[1][2]*v.z,x[2][0]*v.x+x[2][1]*v.y+x[2][2]*v.z);}

    void invert_rotation_and_translation()
    {*this=inverse_rotation_and_translation();}

    Matrix inverse_rotation_and_translation()
    {return Matrix(x[0][0],x[0][1],x[0][2],0,x[1][0],x[1][1],x[1][2],0,x[2][0],x[2][1],x[2][2],0,-x[0][0]*x[0][3]-x[1][0]*x[1][3]-x[2][0]*x[2][3],-x[0][1]*x[0][3]-x[1][1]*x[1][3]-x[2][1]*x[2][3],-x[0][2]*x[0][3]-x[1][2]*x[1][3]-x[2][2]*x[2][3],1);}

    Matrix rotation_only() const
    {return Matrix(x[0][0],x[1][0],x[2][0],0,x[0][1],x[1][1],x[2][1],0,x[0][2],x[1][2],x[2][2],0,0,0,0,1);}

    const Vector<T,3> translation() const
    {return Vector<T,3>(x[0][3],x[1][3],x[2][3]);}

    void set_translation(const Vector<T,3>& t)
    {x[0][3]=t[0];x[1][3]=t[1];x[2][3]=t[2];}

    Matrix<T,3> linear() const
    {return Matrix<T,3>(x[0][0],x[1][0],x[2][0],x[0][1],x[1][1],x[2][1],x[0][2],x[1][2],x[2][2]);}

    static Matrix from_linear(const Matrix<T,3>& M) // Create a homogeneous 4x4 matrix corresponding to a 3x3 transform
    {return Matrix(M.x[0][0],M.x[1][0],M.x[2][0],0,M.x[0][1],M.x[1][1],M.x[2][1],0,M.x[0][2],M.x[1][2],M.x[2][2],0,0,0,0,1);}

    void transpose()
    {swap(x[1][0],x[0][1]);swap(x[2][0],x[0][2]);swap(x[3][0],x[0][3]);swap(x[2][1],x[1][2]);swap(x[3][1],x[1][3]);swap(x[3][2],x[2][3]);}

    Matrix transposed() const
    {return Matrix(x[0][0],x[0][1],x[0][2],x[0][3],x[1][0],x[1][1],x[1][2],x[1][3],x[2][0],x[2][1],x[2][2],x[2][3],x[3][0],x[3][1],x[3][2],x[3][3]);}

    static Matrix translation_matrix(const Vector<T,3>& translation)
    {return Matrix(1,0,0,0,0,1,0,0,0,0,1,0,translation.x,translation.y,translation.z,1);}

    static Matrix identity_matrix()
    {return Matrix(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);}

    static Matrix rotation_matrix_x(const T angle)
    {T c=cos(angle),s=sin(angle);return Matrix(1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1);}

    static Matrix rotation_matrix_y(const T angle)
    {T c=cos(angle),s=sin(angle);return Matrix(c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1);}

    static Matrix rotation_matrix_z(const T angle)
    {T c=cos(angle),s=sin(angle);return Matrix(c,s,0,0,-s,c,0,0,0,0,1,0,0,0,0,1);}

    static Matrix rotation_matrix(const Vector<T,3>& axis,const T angle)
    {return from_linear(Matrix<T,3>::rotation_matrix(axis,angle));}

    static Matrix rotation_matrix(const Vector<T,3>& x_final,const Vector<T,3>& y_final,const Vector<T,3>& z_final)
    {return Matrix(x_final.x,x_final.y,x_final.z,0,y_final.x,y_final.y,y_final.z,0,z_final.x,z_final.y,z_final.z,0,0,0,0,1);}

    static Matrix rotation_matrix(const Vector<T,3>& initial_vector,const Vector<T,3>& final_vector)
    {return from_linear(Rotation<Vector<T,3> >::from_rotated_vector(initial_vector,final_vector).matrix());}

    static Matrix scale_matrix(const Vector<T,3>& scale_vector)
    {return Matrix(scale_vector.x,0,0,0,0,scale_vector.y,0,0,0,0,scale_vector.z,0,0,0,0,1);}

    static Matrix scale_matrix(const T scale)
    {return Matrix(scale,0,0,0,0,scale,0,0,0,0,scale,0,0,0,0,1);}

    T sqr_frobenius_norm() const
    {T sum=0;
    for(int i=0;i<16;i++) sum+=sqr(x[0][i]);
    return sum;}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

//#####################################################################
    GEODE_CORE_EXPORT Matrix operator*(const Matrix& A) const;
    GEODE_CORE_EXPORT void invert();
    GEODE_CORE_EXPORT Matrix inverse() const;
    GEODE_CORE_EXPORT Matrix cofactor_matrix() const;
//#####################################################################
};
template<class T>
inline Matrix<T,4> operator*(const T a,const Matrix<T,4>& A)
{return Matrix<T,4>(a*A.x[0][0],a*A.x[1][0],a*A.x[2][0],a*A.x[3][0],a*A.x[0][1],a*A.x[1][1],a*A.x[2][1],a*A.x[3][1],a*A.x[0][2],a*A.x[1][2],a*A.x[2][2],a*A.x[3][2],a*A.x[0][3],a*A.x[1][3],a*A.x[2][3],a*A.x[3][3]);}

template<class T>
inline std::istream& operator>>(std::istream& input,Matrix<T,4>& A)
{for(int i=0;i<4;i++) for(int j=0;j<4;j++) input>>A.x[i][j];return input;}

template<class T>
inline std::ostream& operator<<(std::ostream& output,Matrix<T,4>& A) {
  output << '[' << std::endl;
  for(int i=0;i<4;i++) {
    output << '[' << A.x[i][0];
    for(int j=1;j<4;j++)
      output<<", "<<A.x[i][j];
    output << ']';
    if (i != 3)
      output << ", ";
  }
  output << ']';
  return output;
}

}
