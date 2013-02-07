//#####################################################################
// Class SymmetricMatrix2x2
//#####################################################################
#pragma once

#include <other/core/vector/Vector2d.h>
#include <other/core/math/small_sort.h>
namespace other {

template<class T> struct IsScalarBlock<SymmetricMatrix<T,2> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<SymmetricMatrix<T,2> >:public IsScalarVectorSpace<T>{};

template<class T>
class SymmetricMatrix<T,2>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=2};
    enum Workaround2 {n=2};

    T x00,x10,x11;

    SymmetricMatrix()
        :x00(),x10(),x11()
    {}

    template<class T2> explicit
    SymmetricMatrix(const SymmetricMatrix<T2,2>& matrix)
        :x00((T)matrix.x00),x10((T)matrix.x10),x11((T)matrix.x11)
    {}

    SymmetricMatrix(const DiagonalMatrix<T,2>& matrix)
        :x00(matrix.x00),x10(0),x11(matrix.x11)
    {}

    SymmetricMatrix(const T y00,const T y10,const T y11)
        :x00(y00),x10(y10),x11(y11)
    {}

    void copy(const SymmetricMatrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(2,2);}

    int rows() const
    {return 2;}

    int columns() const
    {return 2;}

    Vector<T,2> column(const int axis) const
    {assert(unsigned(axis)<2);return axis==0?Vector<T,2>(x00,x10):Vector<T,2>(x10,x11);}

    T& operator()(int i,int j)
    {return i<j?element_upper(i,j):element_lower(i,j);}

    const T& operator()(int i,int j) const
    {return i<j?element_upper(i,j):element_lower(i,j);}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<2 && unsigned(j)<2;}

    T& element_upper(int i,int j)
    {return element_lower(j,i);}

    const T& element_upper(int i,int j) const
    {return element_lower(j,i);}

    T& element_lower(int i,int j)
    {assert(unsigned(j)<=unsigned(i) && unsigned(i)<2);return ((T*)this)[((3-j)*j>>1)+i];}

    const T& element_lower(int i,int j) const
    {assert(unsigned(j)<=unsigned(i) && unsigned(i)<2);
    return ((const T*)this)[((3-j)*j>>1)+i];}

    bool operator==(const SymmetricMatrix& A) const
    {return x00==A.x00 && x10==A.x10 && x11==A.x11;}

    bool operator!=(const SymmetricMatrix& A) const
    {return !(*this==A);}

    static SymmetricMatrix componentwise_min(const SymmetricMatrix& v1,const SymmetricMatrix& v2)
    {return SymmetricMatrix(min(v1.x00,v2.x00),min(v1.x10,v2.x10),min(v1.x11,v2.x11));}

    static SymmetricMatrix componentwise_max(const SymmetricMatrix& v1,const SymmetricMatrix& v2)
    {return SymmetricMatrix(max(v1.x00,v2.x00),max(v1.x10,v2.x10),max(v1.x11,v2.x11));}

    SymmetricMatrix operator-() const
    {return SymmetricMatrix(-x00,-x10,-x11);}

    SymmetricMatrix& operator+=(const SymmetricMatrix& A)
    {x00+=A.x00;x10+=A.x10;x11+=A.x11;return *this;}

    SymmetricMatrix& operator+=(const T& a)
    {x00+=a;x11+=a;return *this;}

    SymmetricMatrix& operator-=(const SymmetricMatrix& A)
    {x00-=A.x00;x10-=A.x10;x11-=A.x11;return *this;}

    SymmetricMatrix& operator-=(const T& a)
    {x00-=a;x11-=a;return *this;}

    SymmetricMatrix& operator*=(const T a)
    {x00*=a;x10*=a;x11*=a;return *this;}

    SymmetricMatrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;x00*=s;x10*=s;x11*=s;return *this;}

    SymmetricMatrix operator+(const SymmetricMatrix& A) const
    {return SymmetricMatrix(x00+A.x00,x10+A.x10,x11+A.x11);}

    SymmetricMatrix operator+(const T a) const
    {return SymmetricMatrix(x00+a,x10,x11+a);}

    SymmetricMatrix operator-(const SymmetricMatrix& A) const
    {return SymmetricMatrix(x00-A.x00,x10-A.x10,x11-A.x11);}

    SymmetricMatrix operator-(const T a) const
    {return SymmetricMatrix(x00-a,x10,x11-a);}

    SymmetricMatrix operator*(const T a) const
    {return SymmetricMatrix(a*x00,a*x10,a*x11);}

    SymmetricMatrix operator/(const T a) const
    {assert(a!=0);return *this*(1/a);}

    Vector<T,2> operator*(const Vector<T,2>& v) const
    {return Vector<T,2>(x00*v.x+x10*v.y,x10*v.x+x11*v.y);}

    template<class TMatrix>
    typename Product<SymmetricMatrix,TMatrix>::type transpose_times(const TMatrix& M) const
    {return *this*M;}

    template<class TMatrix>
    typename ProductTranspose<SymmetricMatrix,TMatrix>::type
    times_transpose(const TMatrix& A) const
    {return (A**this).transposed();}

    T determinant() const
    {return x00*x11-x10*x10;}

    SymmetricMatrix inverse() const
    {return SymmetricMatrix(x11,-x10,x00)/determinant();}

    SymmetricMatrix transposed() const
    {return *this;}

    void transpose()
    {}

    Matrix<T,2> times_transpose(const Matrix<T,2>& A) const
    {return *this*A.transposed();}

    Matrix<T,2,3> times_transpose(const Matrix<T,3,2>& A) const
    {return Matrix<T,2,3>::column_major(x00*A(0,0)+x10*A(0,1),x10*A(0,0)+x11*A(0,1),x00*A(1,0)+x10*A(1,1),x10*A(1,0)+x11*A(1,1),x00*A(2,0)+x10*A(2,1),x10*A(2,0)+x11*A(2,1));}

    Matrix<T,2> times_transpose(const DiagonalMatrix<T,2>& A) const
    {return *this*A;}

    Matrix<T,2> times_transpose(const SymmetricMatrix<T,2>& A) const
    {return *this*A;}

    Matrix<T,1,2> cross_product_matrix_times(const Vector<T,2>& v) const
    {return cross_product_matrix(v)*(*this);}

    Vector<T,2> solve_linear_system(const Vector<T,2>& b) const
    {return SymmetricMatrix(x11,-x10,x00)*b/determinant();}

    Vector<T,2> robust_solve_linear_system(const Vector<T,2>& b) const
    {T determinant=determinant();
    Vector<T,2> unscaled_result=SymmetricMatrix(x11,-x10,x00)*b;
    T relative_tolerance=(T)FLT_MIN*unscaled_result.maxabs();
    if(abs(determinant)<=relative_tolerance){relative_tolerance=max(relative_tolerance,(T)FLT_MIN);determinant=determinant>=0?relative_tolerance:-relative_tolerance;}
    return unscaled_result/determinant;}

    T trace() const
    {return x00+x11;}

    T sqr_frobenius_norm() const
    {return x00*x00+x11*x11+2*x10*x10;}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    SymmetricMatrix cofactor_matrix()
    {return SymmetricMatrix(x11,-x10,x00);}

    Vector<T,2> largest_column() const
    {return abs(x00)>abs(x11)?Vector<T,2>(x00,x10):Vector<T,2>(x10,x11);}

    Vector<T,2> largest_column_normalized() const // 5 mults, 2 adds, 1 div, 1 sqrt
    {T sqr11=sqr(x00),sqr12=sqr(x10),sqr22=sqr(x11);
    T scale1=sqr11+sqr12,scale2=sqr12+sqr22;
    if(scale1>scale2) return Vector<T,2>(x00,x10)/sqrt(scale1);
    else if(scale2>0) return Vector<T,2>(x10,x11)/sqrt(scale2);
    else return Vector<T,2>(1,0);}

    T maxabs() const
    {return maxabs(x00,x10,x11);}

    static SymmetricMatrix identity_matrix()
    {return SymmetricMatrix(1,0,1);}

    static SymmetricMatrix unit_matrix(const T scale=1)
    {return SymmetricMatrix(scale,scale,scale);}

    bool positive_definite() const
    {return x00>0 && x00*x11>x10*x10;}

    bool positive_semidefinite(const T tolerance=(T)1e-7) const
    {T scale=maxabs();return !scale || (*this+tolerance*scale).positive_definite();}

    Vector<T,2> first_eigenvector_from_ordered_eigenvalues(const DiagonalMatrix<T,2>& eigenvalues) const
    {return (*this-eigenvalues.x00).cofactor_matrix().largest_column_normalized();}

    Vector<T,2> last_eigenvector_from_ordered_eigenvalues(const DiagonalMatrix<T,2>& eigenvalues) const
    {return (*this-eigenvalues.x11).cofactor_matrix().largest_column_normalized();}

    DiagonalMatrix<T,2> fast_eigenvalues() const
    {T da;
    if(x10==0) da=0;
    else{T theta=(T).5*(x11-x00)/x10,t=1/(abs(theta)+sqrt(1+sqr(theta)));if(theta<0) t=-t;da=t*x10;}
    DiagonalMatrix<T,2> eigenvalues(x00-da,x11+da);
    small_sort(eigenvalues.x00,eigenvalues.x11);return eigenvalues;}

    SymmetricMatrix positive_definite_part() const
    {DiagonalMatrix<T,2> D;Matrix<T,2> V;solve_eigenproblem(D,V);D=D.clamp_min(0);return conjugate(V,D);}

    DiagonalMatrix<T,2> diagonal_part() const
    {return DiagonalMatrix<T,2>(x00,x11);}

    void fast_solve_eigenproblem(DiagonalMatrix<T,2>& eigenvalues,Matrix<T,2>& eigenvectors) const
    {solve_eigenproblem(eigenvalues,eigenvectors);}

//#####################################################################
    void solve_eigenproblem(DiagonalMatrix<T,2>& eigenvalues,Matrix<T,2>& eigenvectors) const;
    Matrix<T,2> operator*(const DiagonalMatrix<T,2>& A) const;
    Matrix<T,2> operator*(const UpperTriangularMatrix<T,2>& A) const;
    SymmetricMatrix operator+(const DiagonalMatrix<T,2>& A) const;
    Matrix<T,2> times_transpose(const UpperTriangularMatrix<T,2>& A) const;
//#####################################################################
};
// global functions
template<class T>
inline SymmetricMatrix<T,2> operator*(const T a,const SymmetricMatrix<T,2>& A) // 4 mults
{return A*a;}

template<class T>
inline SymmetricMatrix<T,2> operator+(const T a,const SymmetricMatrix<T,2>& A) // 2 adds
{return A+a;}

template<class T>
inline SymmetricMatrix<T,2> operator-(const T a,const SymmetricMatrix<T,2>& A) // 2 adds
{return -A+a;}

template<class T>
inline SymmetricMatrix<T,2> clamp(const SymmetricMatrix<T,2>& x,const SymmetricMatrix<T,2>& xmin,const SymmetricMatrix<T,2>& xmax)
{return SymmetricMatrix<T,2>(clamp(x.x00,xmin.x00,xmax.x00),clamp(x.x10,xmin.x10,xmax.x10),clamp(x.x11,xmin.x11,xmax.x11));}

template<class T>
inline SymmetricMatrix<T,2> clamp_min(const SymmetricMatrix<T,2>& x,const SymmetricMatrix<T,2>& xmin)
{return SymmetricMatrix<T,2>(clamp_min(x.x00,xmin.x00),clamp_min(x.x10,xmin.x10),clamp_min(x.x11,xmin.x11));}

template<class T>
inline SymmetricMatrix<T,2> clamp_max(const SymmetricMatrix<T,2>& x,const SymmetricMatrix<T,2>& xmax)
{return SymmetricMatrix<T,2>(clamp_max(x.x00,xmax.x00),clamp_max(x.x10,xmax.x10),clamp_max(x.x11,xmax.x11));}

template<class T> inline SymmetricMatrix<T,2> outer_product(const Vector<T,2>& u) // 3 mults
{return SymmetricMatrix<T,2>(u.x*u.x,u.x*u.y,u.y*u.y);}

template<class T> inline SymmetricMatrix<T,2> scaled_outer_product(const T a,const Vector<T,2>& u) // 5 mults
{Vector<T,2> au=a*u;return SymmetricMatrix<T,2>(au.x*u.x,au.x*u.y,au.y*u.y);}

template<class T>
inline std::ostream& operator<< (std::ostream& output,const SymmetricMatrix<T,2>& A)
{output<<A.x00<<"\n"<<A.x10<<" "<<A.x11<<"\n";return output;}

template<class T>
inline SymmetricMatrix<T,2> log(const SymmetricMatrix<T,2>& A)
{DiagonalMatrix<T,2> D;Matrix<T,2> Q;A.solve_eigenproblem(D,Q);return SymmetricMatrix<T,2>::conjugate(Q,log(D));}

template<class T>
inline SymmetricMatrix<T,2> exp(const SymmetricMatrix<T,2>& A)
{DiagonalMatrix<T,2> D;Matrix<T,2> Q;A.solve_eigenproblem(D,Q);return SymmetricMatrix<T,2>::conjugate(Q,exp(D));}

template<class T> void SymmetricMatrix<T,2>::
solve_eigenproblem(DiagonalMatrix<T,2>& eigenvalues,Matrix<T,2>& eigenvectors) const
{
    typedef Vector<T,2> TV;
    T a=(T).5*(x00+x11),b=(T).5*(x00-x11),c=x10;
    T c_squared=sqr(c),m=sqrt(sqr(b)+c_squared),k=x00*x11-c_squared;
    if(a>=0){eigenvalues.x00=a+m;eigenvalues.x11=eigenvalues.x00?k/eigenvalues.x00:0;}
    else{eigenvalues.x11=a-m;eigenvalues.x00=eigenvalues.x11?k/eigenvalues.x11:0;}
    small_sort(eigenvalues.x11,eigenvalues.x00); // if order is wrong, matrix is nearly scalar
    eigenvectors.set_column(0,(b>=0?TV(m+b,c):TV(-c,b-m)).normalized());
    eigenvectors.set_column(1,perpendicular(eigenvectors.column(0)));
}

template<class T> inline SymmetricMatrix<T,2>
conjugate(const Matrix<T,2>& A,const DiagonalMatrix<T,2>& B) // 10 mults, 3 adds
{
    Matrix<T,2> BA=B*A.transposed();
    return SymmetricMatrix<T,2>(A.x[0][0]*BA.x[0][0]+A.x[0][1]*BA.x[1][0],A.x[1][0]*BA.x[0][0]+A.x[1][1]*BA.x[1][0],A.x[1][0]*BA.x[0][1]+A.x[1][1]*BA.x[1][1]);
}

template<class T> inline SymmetricMatrix<T,2>
conjugate(const Matrix<T,2>& A,const SymmetricMatrix<T,2>& B) // 12 mults, 7 adds
{
    Matrix<T,2> BA=(A*B).transposed();
    return SymmetricMatrix<T,2>(A.x[0][0]*BA.x[0][0]+A.x[0][1]*BA.x[1][0],A.x[1][0]*BA.x[0][0]+A.x[1][1]*BA.x[1][0],A.x[1][0]*BA.x[0][1]+A.x[1][1]*BA.x[1][1]);
}

template<class T> inline SymmetricMatrix<T,2>
conjugate(const Matrix<T,2,3>& A,const DiagonalMatrix<T,3>& B)
{
    return transpose_times_with_symmetric_result(A.transpose,B*A.transpose);
}

template<class T> inline SymmetricMatrix<T,2>
conjugate(const Matrix<T,2,3>& A,const SymmetricMatrix<T,3>& B)
{
    return transpose_times_with_symmetric_result(A.transpose,B*A.transpose);
}

template<class T> inline SymmetricMatrix<T,2>
conjugate(const UpperTriangularMatrix<T,2>& A,const SymmetricMatrix<T,2>& B)
{
    Matrix<T,2> BA=B.times_transpose(A);
    return SymmetricMatrix<T,2>(A.x00*BA.x[0]+A.x01*BA.x[1],A.x11*BA.x[1],A.x11*BA.x[3]);
}

template<class T> inline SymmetricMatrix<T,2>
conjugate_with_transpose(const Matrix<T,2>& A,const DiagonalMatrix<T,2>& B) // 10 mults, 3 adds
{
    return transpose_times_with_symmetric_result(B*A,A);
}

template<class T> inline SymmetricMatrix<T,2>
conjugate_with_transpose(const Matrix<T,2>& A,const SymmetricMatrix<T,2>& B) // 12 mults, 7 adds
{
    return transpose_times_with_symmetric_result(B*A,A);
}

template<class T> inline SymmetricMatrix<T,2>
conjugate_with_transpose(const UpperTriangularMatrix<T,2>& A,const SymmetricMatrix<T,2>& B) // 10 mults, 3 adds
{
    return transpose_times_with_symmetric_result(B*A,A);
}

template<class T> inline SymmetricMatrix<T,2>
conjugate_with_transpose(const Matrix<T,3,2>& A,const SymmetricMatrix<T,3>& B) // 21 mults, 12 adds
{
    return transpose_times_with_symmetric_result(B*A,A);
}

template<class T> inline Matrix<T,2> SymmetricMatrix<T,2>::
times_transpose(const UpperTriangularMatrix<T,2>& A) const // 6 mults, 2 adds
{
    return (A**this).transposed();
}

template<class T> inline Matrix<T,2> SymmetricMatrix<T,2>::
operator*(const DiagonalMatrix<T,2>& A) const // 4 mults
{
    return Matrix<T,2>(x00*A.x00,x10*A.x00,x10*A.x11,x11*A.x11);
}

template<class T> inline Matrix<T,2> SymmetricMatrix<T,2>::
operator*(const UpperTriangularMatrix<T,2>& A) const // 6 mults, 2 adds
{
    return Matrix<T,2>(x00*A.x00,x10*A.x00,x00*A.x01+x10*A.x11,x10*A.x01+x11*A.x11);
}

template<class T> inline Matrix<T,2> operator*(const DiagonalMatrix<T,2>& D,const SymmetricMatrix<T,2>& A) // 4 mults
{
    return Matrix<T,2>(D.x00*A.x00,D.x11*A.x10,D.x00*A.x10,D.x11*A.x11);
}

template<class T>
inline Matrix<T,2> operator*(const UpperTriangularMatrix<T,2>& A,const SymmetricMatrix<T,2>& B) // 6 mults, 2 adds
{
    return Matrix<T,2>(A.x00*B.x00+A.x01*B.x10,A.x11*B.x10,A.x00*B.x10+A.x01*B.x11,A.x11*B.x11);
}

template<class T>
inline Matrix<T,2> operator*(const SymmetricMatrix<T,2>& A,const Matrix<T,2>& B) // 8 mults, 4 mults
{
    return Matrix<T,2>(A.x00*B.x[0][0]+A.x10*B.x[1][0],A.x10*B.x[0][0]+A.x11*B.x[1][0],A.x00*B.x[0][1]+A.x10*B.x[1][1],A.x10*B.x[0][1]+A.x11*B.x[1][1]);
}

template<class T>
inline Matrix<T,2> operator*(const SymmetricMatrix<T,2>& A,const SymmetricMatrix<T,2>& B) // 8 mults, 4 adds
{
    return Matrix<T,2>(A.x00*B.x00+A.x10*B.x10,A.x10*B.x00+A.x11*B.x10,A.x00*B.x10+A.x10*B.x11,A.x10*B.x10+A.x11*B.x11);
}

template<class T> inline SymmetricMatrix<T,2> SymmetricMatrix<T,2>::
operator+(const DiagonalMatrix<T,2>& A) const // 2 adds
{
    return SymmetricMatrix<T,2>(x00+A.x00,x10,x11+A.x11);
}

template<class T> inline SymmetricMatrix<T,2>
operator+(const DiagonalMatrix<T,2>& A,const SymmetricMatrix<T,2>& B) // 2 adds
{
    return B+A;
}

template<class T> inline Matrix<T,2>
operator+(const SymmetricMatrix<T,2>& A,const UpperTriangularMatrix<T,2>& B)
{
    return Matrix<T,2>(A.x00+B.x00,A.x10,A.x10+B.x01,A.x11+B.x11);
}

template<class T> inline Matrix<T,2>
operator+(const UpperTriangularMatrix<T,2>& A,const SymmetricMatrix<T,2>& B)
{
    return B+A;
}

template<class T> inline Matrix<T,2>
operator-(const SymmetricMatrix<T,2>& A,const UpperTriangularMatrix<T,2>& B)
{
    return Matrix<T,2>(A.x00-B.x00,A.x10,A.x10-B.x01,A.x11-B.x11);
}

template<class T> inline Matrix<T,2>
operator-(const UpperTriangularMatrix<T,2>& A,const SymmetricMatrix<T,2>& B)
{
    return -B+A;
}

template<class T> inline SymmetricMatrix<T,2>
operator-(const DiagonalMatrix<T,2>& A,const SymmetricMatrix<T,2>& B) // 2 adds
{
    return SymmetricMatrix<T,2>(A.x00-B.x00,-B.x10,A.x11-B.x11);
}

template<class T> inline T
inner_product(const SymmetricMatrix<T,2>& A,const SymmetricMatrix<T,2>& B) {
  return A.x00*B.x00+A.x11*B.x11+2*A.x10*B.x10;
}

template<class T> inline SymmetricMatrix<T,2>
transpose_times_with_symmetric_result(const Matrix<T,2>& A,const Matrix<T,2>& B) { // A^t*B and assume symmetric result, 6 mults, 3 adds
  return SymmetricMatrix<T,2>(A.x[0][0]*B.x[0][0]+A.x[1][0]*B.x[1][0],A.x[0][1]*B.x[0][0]+A.x[1][1]*B.x[1][0],A.x[0][1]*B.x[0][1]+A.x[1][1]*B.x[1][1]);
}

template<class T> inline SymmetricMatrix<T,2>
transpose_times_with_symmetric_result(const Matrix<T,3,2>& A,const Matrix<T,3,2>& B) { // A^t*B and assume symmetric result, 9 mults, 6 adds
  return SymmetricMatrix<T,2>(A.x[0][0]*B.x[0][0]+A.x[1][0]*B.x[1][0]+A.x[2][0]*B.x[2][0],A.x[0][1]*B.x[0][0]+A.x[1][1]*B.x[1][0]+A.x[2][1]*B.x[2][0],A.x[0][1]*B.x[0][1]+A.x[1][1]*B.x[1][1]+A.x[2][1]*B.x[2][1]);
}

template<class T> inline SymmetricMatrix<T,2>
transpose_times_with_symmetric_result(const Matrix<T,2>& A,const UpperTriangularMatrix<T,2>& B) { // A^t*B and assume symmetric result, 4 mults, 1 adds
  return SymmetricMatrix<T,2>(A.x[0][0]*B.x00,A.x[0][1]*B.x00,A.x[0][1]*B.x01+A.x[1][1]*B.x11);
}

}
