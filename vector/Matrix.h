//#####################################################################
// Class Matrix
//#####################################################################
#pragma once

#include <other/core/vector/forward.h>
#include <other/core/vector/Matrix0x0.h>
#include <other/core/vector/Matrix1x1.h>
#include <other/core/vector/Matrix2x2.h>
#include <other/core/vector/Matrix3x2.h>
#include <other/core/vector/Matrix3x3.h>
#include <other/core/vector/Matrix4x4.h>
#include <other/core/vector/Vector.h>
#include <iomanip>
namespace other {

template<class T,int m,int n> OTHER_CORE_EXPORT PyObject* to_python(const Matrix<T,m,n>& matrix) ;
template<class T,int m,int n> struct FromPython<Matrix<T,m,n> >{OTHER_CORE_EXPORT static Matrix<T,m,n> convert(PyObject* object);};

template<class T,int m_,int n_> // n_=m_
class Matrix
{
public:
    enum Workaround1 {m=m_,n=n_,size=m_*n_};
    BOOST_STATIC_ASSERT((!((m>=n && m<=3 && n>=2 && n<=3) || (m==4 && n==4) || (m==0 && n==0)))); // 0x0, 1x1, 2x2, 3x2, 3x3, and 4x4 are specialized
    static const bool is_const=false;
    typedef T Scalar;

    T x[m][n];

    Matrix()
    {
        for(int i=0;i<m;i++) for(int j=0;j<n;j++) x[i][j]=T();
    }

    explicit Matrix(const Vector<T,size>& column1)
    {
        BOOST_STATIC_ASSERT((m==1 || n==1) && size==m+n-1);
        if (m==1)
          for(int i=0;i<size;i++) x[0][i]=column1[i];
        else
          for(int i=0;i<size;i++) x[i][0]=column1[i];
    }

    explicit Matrix(RawArray<const T,2> A)
    {
        assert(m==A.m && n==A.n);
        for(int i=0;i<m;i++) for(int j=0;j<n;j++)
            x[i][j]=A(i,j);
    }

    static Matrix column_major(const T x00,const T x10,const T x01,const T x11,const T x02,const T x12)
    {
        BOOST_STATIC_ASSERT(m==2 && n==3);
        Matrix r;
        r.x[0][0]=x00;r.x[1][0]=x10;
        r.x[0][1]=x01;r.x[1][1]=x11;
        r.x[0][2]=x02;r.x[1][2]=x12;
        return r;
    }

    void copy(const Matrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(m,n);}

    int rows() const
    {return m;}

    int columns() const
    {return n;}

    T& operator()(const int i,const int j)
    {assert(unsigned(i)<m && unsigned(j)<n);return x[i][j];}

    const T& operator()(const int i,const int j) const
    {assert(unsigned(i)<m && unsigned(j)<n);return x[i][j];}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<m && unsigned(j)<n;}

    Vector<T,n>& operator[](const int i)
    {assert(unsigned(i)<m);return *(Vector<T,n>*)x[i];}

    const Vector<T,n>& operator[](const int i) const
    {assert(unsigned(i)<m);return *(const Vector<T,n>*)x[i];}

    Vector<T,m> column(const int j) const
    {assert(unsigned(j)<n);
    Vector<T,m> c;for(int i=0;i<m;i++) c[i]=x[i][j];
    return c;}

    void set_column(const int j,const Vector<T,m>& c)
    {assert(unsigned(j)<n);
    for(int i=0;i<m;i++) x[i][j]=c[i];}

    bool operator==(const Matrix& A) const
    {for(int i=0;i<m;i++) for(int j=0;j<n;j++) if(x[i][j]!=A.x[i][j]) return false;return true;}

    bool operator!=(const Matrix& A) const
    {return !(*this==A);}

    Matrix& operator=(const Matrix& A)
    {for(int i=0;i<m;i++) for(int j=0;j<n;j++) x[i][j]=A.x[i][j];return *this;}

    Matrix& operator*=(const T a)
    {for(int i=0;i<m;i++) for(int j=0;j<n;j++) x[i][j]*=a;return *this;}

    Matrix& operator/=(const T a)
    {return *this*=(1/a);}

    Matrix& operator+=(const Matrix& A)
    {for(int i=0;i<m;i++) for(int j=0;j<n;j++) x[i][j]+=A.x[i][j];return *this;}

    Matrix& operator-=(const Matrix& A)
    {for(int i=0;i<m;i++) for(int j=0;j<n;j++) x[i][j]-=A.x[i][j];return *this;}

    Matrix operator+(const Matrix& A) const
    {assert(n==A.n && m==A.m);Matrix matrix;for(int i=0;i<m;i++) for(int j=0;j<n;j++) matrix.x[i][j]=x[i][j]+A.x[i][j];return matrix;}

    Matrix operator-(const Matrix& A) const
    {assert(n==A.n && m==A.m);Matrix matrix;for(int i=0;i<m;i++) for(int j=0;j<n;j++) matrix.x[i][j]=x[i][j]-A.x[i][j];return matrix;}

    Matrix operator-() const
    {Matrix matrix;for(int i=0;i<m;i++) for(int j=0;j<n;j++) matrix.x[i][j]=-x[i][j];return matrix;}

    Matrix operator*(const T a) const
    {Matrix matrix;for(int i=0;i<m;i++) for(int j=0;j<n;j++) matrix.x[i][j]=x[i][j]*a;return matrix;}

    Matrix operator/(const T a) const
    {return (*this)*(1/a);}

    Vector<T,m> operator*(const Vector<T,n>& y) const
    {Vector<T,m> result;for(int j=0;j<n;j++) for(int i=0;i<m;i++) result[i]+=(*this)(i,j)*y[j];return result;}

    template<int p>
    Matrix<T,m,p> operator*(const Matrix<T,n,p>& A) const
    {Matrix<T,m,p> matrix;for(int j=0;j<p;j++) for(int k=0;k<n;k++) for(int i=0;i<m;i++) matrix(i,j)+=(*this)(i,k)*A(k,j);return matrix;}

    Matrix<T,m,n> operator*(const SymmetricMatrix<T,n>& A) const
    {Matrix<T,m,n> matrix;for(int j=0;j<n;j++) for(int k=0;k<n;k++) for(int i=0;i<m;i++) matrix(i,j)+=(*this)(i,k)*A(k,j);return matrix;}

    Matrix<T,m,n> operator*(const DiagonalMatrix<T,n>& A) const
    {Matrix<T,m,n> matrix;for(int j=0;j<n;j++) for(int i=0;i<m;i++) matrix(i,j)=(*this)(i,j)*A(j);return matrix;}

    Matrix<T,n,m> transposed() const
    {Matrix<T,n,m> matrix;for(int i=0;i<m;i++) for(int j=0;j<n;j++) matrix(j,i)=(*this)(i,j);return matrix;}

    Vector<T,n> transpose_times(const Vector<T,m>& y) const
    {Vector<T,n> result;for(int j=0;j<n;j++) for(int i=0;i<m;i++) result[j]+=(*this)(i,j)*y[i];return result;}

    template<int p>
    Matrix<T,n,p> transpose_times(const Matrix<T,m,p>& A) const
    {Matrix<T,n,p> matrix;for(int j=0;j<p;j++) for(int i=0;i<n;i++) for(int k=0;k<m;k++) matrix(i,j)+=(*this)(k,i)*A(k,j);return matrix;}

    template<class TMatrix>
    typename TransposeProduct<Matrix,TMatrix>::type
    transpose_times(const TMatrix& A) const
    {return transposed()*A;}

    template<int p>
    Matrix<T,m,p> times_transpose(const Matrix<T,p,n>& A) const
    {Matrix<T,m,p> matrix;for(int i=0;i<m;i++) for(int j=0;j<p;j++) for(int k=0;k<n;k++) matrix(i,j)+=(*this)(i,k)*A(j,k);return matrix;}

    template<class TMatrix>
    typename ProductTranspose<Matrix,TMatrix>::type
    times_transpose(const TMatrix& A) const
    {return (A*transposed()).transposed();}

    Matrix permute_columns(const Vector<int,n>& p) const
    {Matrix m;for(int i=0;i<m;i++) for(int j=0;j<n;j++) m.x[i][j]=x[i][p[j]];return m;}

    Matrix unpermute_columns(const Vector<int,n>& p) const
    {Matrix m;for(int i=0;i<m;i++) for(int j=0;j<n;j++) m.x[i][p[j]]=x[i][j];return m;}

    Matrix<T,n> normal_equations_matrix() const
    {Matrix<T,n> result;for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<m;k++) result.x[i][j]+=x[k][i]*x[k][j];return result;}

    Vector<T,n> normal_equations_solve(const Vector<T,m>& b) const
    {Matrix<T,n> A_transpose_A(normal_equations_matrix());Vector<T,n> A_transpose_b(transpose_times(b));return A_transpose_A.cholesky_solve(A_transpose_b);}

    template<class TVector>
    Vector<T,n> solve_linear_system(const TVector& b)
    {return PLU_Solve(b);}

    T parallelepiped_measure() const
    {BOOST_STATIC_ASSERT(n==1);return sqrt(sqr_frobenius_norm());}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    T sqr_frobenius_norm() const
    {T sum=0;for(int i=0;i<m;i++) for(int j=0;j<n;j++) sum+=sqr(x[i][j]);return sum;}

    void fast_singular_value_decomposition(Matrix<T,2>& U,DiagonalMatrix<T,2>& singular_values,Matrix<T,3,2>& V) const
    {transposed().fast_singular_value_decomposition(V,singular_values,U);}
};

template<class T, int d> Matrix<T, d+1> reflection_around(const int axis, const Vector<T, d> center) {
  auto move_axis_to_zero = Matrix<T, d+1>::translation_matrix(-center);
  auto reflection = Matrix<T, d+1>::identity_matrix();
  reflection(axis, axis) = -1;
  auto move_axis_inverse = Matrix<T, d+1>::translation_matrix(center);
  return move_axis_inverse*reflection*move_axis_to_zero;
}

template<class T, int d> Matrix<T, d+1> promote_homogonous(const Matrix<T, d>& m) {
  Matrix<T, d> new_rotation = Matrix<T, d>::from_linear(m.extract_rotation());
  Matrix<T, d+1> result = Matrix<T, d+1>::from_linear(new_rotation);
  Vector<T, d> new_translation(m.translation());
  result.set_translation(new_translation);
  return result;
}

template<class T,int m,int n>
inline Matrix<T,m,n> operator*(const T a,const Matrix<T,m,n>& A)
{return A*a;}

template<class T,int d,int n> Matrix<T,d,n>
operator*(const DiagonalMatrix<T,d>& A,const Matrix<T,d,n>& B)
{Matrix<T,d,n> M;
for(int k=0;k<n;k++) for(int i=0;i<d;i++) M(i,k)=A(i)*B(i,k);
return M;}

template<class T,int d,int n> Matrix<T,d,n>
operator*(const SymmetricMatrix<T,d>& A,const Matrix<T,d,n>& B)
{return Matrix<T,d,d>(A)*B;}

template<class T,int m,int n> inline Matrix<T,m,n>
outer_product(const Vector<T,m>& u,const Vector<T,n>& v)
{Matrix<T,m,n> result;for(int i=0;i<m;i++) for(int j=0;j<n;j++) result[i][j]=u[i]*v[j];return result;}

template<class T,int m,int n> std::ostream& operator<<(std::ostream& output,const Matrix<T,m,n>& A)
{for(int i=0;i<m;i++){for(int j=0;j<n;j++) output<<std::setw(12)<<A(i,j)<<" ";output<<std::endl;}return output;}

}
