//#####################################################################
// Class SymmetricMatrix3x3
//#####################################################################
#pragma once

#include <other/core/vector/Vector3d.h>
#include <other/core/math/robust.h>
#include <other/core/utility/HasCheapCopy.h>
namespace other {

template<class T> struct HasCheapCopy<SymmetricMatrix<T,3> >:public mpl::true_{};
template<class T> struct IsScalarBlock<SymmetricMatrix<T,3> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<SymmetricMatrix<T,3> >:public IsScalarVectorSpace<T>{};

template<class T>
class SymmetricMatrix<T,3>
{
public:
    typedef T Scalar;
    enum Workaround1 {m=3};
    enum Workaround2 {n=3};

    T x00,x10,x20,x11,x21,x22;

    SymmetricMatrix()
        :x00(),x10(),x20(),x11(),x21(),x22()
    {
        BOOST_STATIC_ASSERT(sizeof(SymmetricMatrix)==6*sizeof(T));
    }

    template<class T2> explicit
    SymmetricMatrix(const SymmetricMatrix<T2,3>& matrix)
        :x00((T)matrix.x00),x10((T)matrix.x10),x20((T)matrix.x20),x11((T)matrix.x11),x21((T)matrix.x21),x22((T)matrix.x22)
    {}

    SymmetricMatrix(const DiagonalMatrix<T,3>& matrix)
        :x00(matrix.x00),x10(0),x20(0),x11(matrix.x11),x21(0),x22(matrix.x22)
    {}

    SymmetricMatrix(const T y00,const T y10,const T y20,const T y11,const T y21,const T y22)
        :x00(y00),x10(y10),x20(y20),x11(y11),x21(y21),x22(y22)
    {}

    void copy(const SymmetricMatrix& A)
    {*this=A;}

    Vector<int,2> sizes() const
    {return Vector<int,2>(3,3);}

    int rows() const
    {return 3;}

    int columns() const
    {return 3;}

    T& operator()(int i,int j)
    {return i<j?element_upper(i,j):element_lower(i,j);}

    const T& operator()(int i,int j) const
    {return i<j?element_upper(i,j):element_lower(i,j);}

    bool valid_index(const int i,const int j) const
    {return unsigned(i)<3 && unsigned(j)<3;}

    T& element_upper(int i,int j)
    {return element_lower(j,i);}

    const T& element_upper(int i,int j) const
    {return element_lower(j,i);}

    T& element_lower(int i,int j)
    {assert(unsigned(j)<=unsigned(i) && unsigned(i)<3);return ((T*)this)[((5-j)*j>>1)+i];}

    const T& element_lower(int i,int j) const
    {assert(unsigned(j)<=unsigned(i) && unsigned(i)<3);return ((const T*)this)[((5-j)*j>>1)+i];}

    Vector<T,3> column(const int axis) const
    {assert(unsigned(axis)<3);return axis==0?Vector<T,3>(x00,x10,x20):axis==1?Vector<T,3>(x10,x11,x21):Vector<T,3>(x20,x21,x22);}

    bool operator==(const SymmetricMatrix& A) const
    {return x00==A.x00 && x10==A.x10 && x20==A.x20 && x11==A.x11 && x21==A.x21 && x22==A.x22;}

    bool operator!=(const SymmetricMatrix& A) const
    {return !(*this==A);}

    static SymmetricMatrix componentwise_min(const SymmetricMatrix& v1,const SymmetricMatrix& v2)
    {return SymmetricMatrix(min(v1.x00,v2.x00),min(v1.x10,v2.x10),min(v1.x20,v2.x20),min(v1.x11,v2.x11),min(v1.x21,v2.x21),min(v1.x22,v2.x22));}

    static SymmetricMatrix componentwise_max(const SymmetricMatrix& v1,const SymmetricMatrix& v2)
    {return SymmetricMatrix(max(v1.x00,v2.x00),max(v1.x10,v2.x10),max(v1.x20,v2.x20),max(v1.x11,v2.x11),max(v1.x21,v2.x21),max(v1.x22,v2.x22));}

    SymmetricMatrix operator-() const
    {return SymmetricMatrix(-x00,-x10,-x20,-x11,-x21,-x22);}

    SymmetricMatrix& operator+=(const SymmetricMatrix& A)
    {x00+=A.x00;x10+=A.x10;x20+=A.x20;x11+=A.x11;x21+=A.x21;x22+=A.x22;return *this;}

    SymmetricMatrix& operator+=(const T& a)
    {x00+=a;x11+=a;x22+=a;return *this;}

    SymmetricMatrix& operator-=(const SymmetricMatrix& A)
    {x00-=A.x00;x10-=A.x10;x20-=A.x20;x11-=A.x11;x21-=A.x21;x22-=A.x22;return *this;}

    SymmetricMatrix& operator-=(const T& a)
    {x00-=a;x11-=a;x22-=a;return *this;}

    SymmetricMatrix& operator*=(const T a)
    {x00*=a;x10*=a;x20*=a;x11*=a;x21*=a;x22*=a;return *this;}

    SymmetricMatrix& operator/=(const T a)
    {assert(a!=0);T s=1/a;x00*=s;x10*=s;x20*=s;x11*=s;x21*=s;x22*=s;return *this;}

    SymmetricMatrix operator+(const SymmetricMatrix& A) const
    {return SymmetricMatrix(x00+A.x00,x10+A.x10,x20+A.x20,x11+A.x11,x21+A.x21,x22+A.x22);}

    SymmetricMatrix operator+(const T a) const
    {return SymmetricMatrix(x00+a,x10,x20,x11+a,x21,x22+a);}

    SymmetricMatrix operator-(const SymmetricMatrix& A) const
    {return SymmetricMatrix(x00-A.x00,x10-A.x10,x20-A.x20,x11-A.x11,x21-A.x21,x22-A.x22);}

    SymmetricMatrix operator-(const T a) const
    {return SymmetricMatrix(x00-a,x10,x20,x11-a,x21,x22-a);}

    SymmetricMatrix operator*(const T a) const
    {return SymmetricMatrix(a*x00,a*x10,a*x20,a*x11,a*x21,a*x22);}

    Matrix<T,3> operator*(const SymmetricMatrix& A) const // 27 mults, 18 adds
    {return Matrix<T,3>(x00*A.x00+x10*A.x10+x20*A.x20,x10*A.x00+x11*A.x10+x21*A.x20,x20*A.x00+x21*A.x10+x22*A.x20,
                        x00*A.x10+x10*A.x11+x20*A.x21,x10*A.x10+x11*A.x11+x21*A.x21,x20*A.x10+x21*A.x11+x22*A.x21,
                        x00*A.x20+x10*A.x21+x20*A.x22,x10*A.x20+x11*A.x21+x21*A.x22,x20*A.x20+x21*A.x21+x22*A.x22);}

    template<class TMatrix>
    typename Product<SymmetricMatrix,TMatrix>::type transpose_times(const TMatrix& M) const
    {return *this*M;}

    template<class TMatrix>
    typename ProductTranspose<SymmetricMatrix,TMatrix>::type times_transpose(const TMatrix& A) const
    {return (A**this).transposed();}

    Matrix<T,3> times_transpose(const SymmetricMatrix& M) const // 27 mults, 18 adds
    {return *this*M;}

    Matrix<T,3> cross_product_matrix_times(const Vector<T,3>& v) const // (v*) * (*this)
    {return Matrix<T,3>(-v.z*x10+v.y*x20,v.z*x00-v.x*x20,-v.y*x00+v.x*x10,-v.z*x11+v.y*x21,v.z*x10-v.x*x21,-v.y*x10+v.x*x11,-v.z*x21+v.y*x22,v.z*x20-v.x*x22,-v.y*x20+v.x*x21);}

    Matrix<T,3> cross_product_matrix_transpose_times(const Vector<T,3>& v) const // (v*)^T * (*this)
    {return Matrix<T,3>(v.z*x10-v.y*x20,-v.z*x00+v.x*x20,v.y*x00-v.x*x10,v.z*x11-v.y*x21,-v.z*x10+v.x*x21,v.y*x10-v.x*x11,v.z*x21-v.y*x22,-v.z*x20+v.x*x22,v.y*x20-v.x*x21);}

    Matrix<T,3> times_cross_product_matrix(const Vector<T,3>& v) const // (*this) * (v*)
    {return Matrix<T,3>(x10*v.z-x20*v.y,x11*v.z-x21*v.y,x21*v.z-x22*v.y,-x00*v.z+x20*v.x,-x10*v.z+x21*v.x,-x20*v.z+x22*v.x,x00*v.y-x10*v.x,x10*v.y-x11*v.x,x20*v.y-x21*v.x);}

    SymmetricMatrix operator/(const T a) const
    {assert(a!=0);T s=1/a;return SymmetricMatrix(s*x00,s*x10,s*x20,s*x11,s*x21,s*x22);}

    Vector<T,3> operator*(const Vector<T,3>& v) const
    {return Vector<T,3>(x00*v.x+x10*v.y+x20*v.z,x10*v.x+x11*v.y+x21*v.z,x20*v.x+x21*v.y+x22*v.z);}

    T determinant() const
    {return x00*(x11*x22-x21*x21)+x10*(2*x21*x20-x10*x22)-x20*x11*x20;}

    SymmetricMatrix inverse() const
    {T cofactor00=x11*x22-x21*x21,cofactor01=x21*x20-x10*x22,cofactor02=x10*x21-x11*x20;
    return SymmetricMatrix(cofactor00,cofactor01,cofactor02,x00*x22-x20*x20,x10*x20-x00*x21,x00*x11-x10*x10)/(x00*cofactor00+x10*cofactor01+x20*cofactor02);}

    SymmetricMatrix transposed() const
    {return *this;}

    void transpose()
    {}

    T dilational() const
    {return T(1./3)*trace();}

    SymmetricMatrix deviatoric() const
    {return *this-dilational();}

    Vector<T,3> solve_linear_system(const Vector<T,3>& b) const // 18 mults, 8 adds
    {T cofactor00=x11*x22-x21*x21,cofactor01=x21*x20-x10*x22,cofactor02=x10*x21-x11*x20;
    return SymmetricMatrix(cofactor00,cofactor01,cofactor02,x00*x22-x20*x20,x10*x20-x00*x21,x00*x11-x10*x10)*b/(x00*cofactor00+x10*cofactor01+x20*cofactor02);}

    Vector<T,3> robust_solve_linear_system(const Vector<T,3>& b) const
    {T cofactor00=x11*x22-x21*x21,cofactor01=x21*x20-x10*x22,cofactor02=x10*x21-x11*x20;
    T determinant=x00*cofactor00+x10*cofactor01+x20*cofactor02;
    Vector<T,3> unscaled_result=SymmetricMatrix(cofactor00,cofactor01,cofactor02,x00*x22-x20*x20,x10*x20-x00*x21,x00*x11-x10*x10)*b;
    T relative_tolerance=(T)FLT_MIN*unscaled_result.maxabs();
    if(abs(determinant)<=relative_tolerance){relative_tolerance=max(relative_tolerance,(T)FLT_MIN);determinant=determinant>=0?relative_tolerance:-relative_tolerance;}
    return unscaled_result/determinant;}

    SymmetricMatrix squared() const
    {return SymmetricMatrix(x00*x00+x10*x10+x20*x20,x10*x00+x11*x10+x21*x20,x20*x00+x21*x10+x22*x20,x10*x10+x11*x11+x21*x21,x20*x10+x21*x11+x22*x21,x20*x20+x21*x21+x22*x22);}

    T trace() const
    {return x00+x11+x22;}

    T sqr_frobenius_norm() const
    {return x00*x00+x11*x11+x22*x22+2*(x10*x10+x20*x20+x21*x21);}

    T frobenius_norm() const
    {return sqrt(sqr_frobenius_norm());}

    SymmetricMatrix cofactor_matrix() // 12 mults, 6 adds
    {return SymmetricMatrix(x11*x22-x21*x21,x21*x20-x10*x22,x10*x21-x11*x20,x00*x22-x20*x20,x10*x20-x00*x21,x00*x11-x10*x10);}

    Vector<T,3> largest_column() const
    {T sqr11=sqr(x00),sqr12=sqr(x10),sqr13=sqr(x20),sqr22=sqr(x11),sqr23=sqr(x21),sqr33=sqr(x22);
    T scale1=sqr11+sqr12+sqr13,scale2=sqr12+sqr22+sqr23,scale3=sqr13+sqr23+sqr33;
    return scale1>scale2?(scale1>scale3?Vector<T,3>(x00,x10,x20):Vector<T,3>(x20,x21,x22)):(scale2>scale3?Vector<T,3>(x10,x11,x21):Vector<T,3>(x20,x21,x22));}

    Vector<T,3> largest_column_normalized() const // 9 mults, 6 adds, 1 div, 1 sqrt
    {T sqr11=sqr(x00),sqr12=sqr(x10),sqr13=sqr(x20),sqr22=sqr(x11),sqr23=sqr(x21),sqr33=sqr(x22);
    T scale1=sqr11+sqr12+sqr13,scale2=sqr12+sqr22+sqr23,scale3=sqr13+sqr23+sqr33;
    if(scale1>scale2){if(scale1>scale3) return Vector<T,3>(x00,x10,x20)/sqrt(scale1);}
    else if(scale2>scale3) return Vector<T,3>(x10,x11,x21)/sqrt(scale2);
    if(scale3>0) return Vector<T,3>(x20,x21,x22)/sqrt(scale3);else return Vector<T,3>(1,0,0);}

    T maxabs() const
    {return other::maxabs(x00,x10,x20,x11,x21,x22);}

    static SymmetricMatrix identity_matrix()
    {return SymmetricMatrix(1,0,0,1,0,1);}

    static SymmetricMatrix unit_matrix(const T scale=1)
    {return SymmetricMatrix(scale,scale,scale,scale,scale,scale);}

    bool positive_definite() const
    {return x00>0 && x00*x11>x10*x10 && determinant()>0;}

    bool positive_semidefinite(const T tolerance=(T)1e-7) const
    {T scale=maxabs();return !scale || (*this+tolerance*scale).positive_definite();}

    Vector<T,3> first_eigenvector_from_ordered_eigenvalues(const DiagonalMatrix<T,3>& eigenvalues,const T tolerance=1e-5) const
    {T scale=other::maxabs(eigenvalues.x00,eigenvalues.x22),scale_inverse=robust_inverse(scale),tiny=tolerance*scale;
    if(eigenvalues.x00-eigenvalues.x11>tiny) return ((*this-eigenvalues.x00)*scale_inverse).cofactor_matrix().largest_column_normalized();
    return ((*this-eigenvalues.x22)*scale_inverse).cofactor_matrix().largest_column().unit_orthogonal_vector();}

    Vector<T,3> last_eigenvector_from_ordered_eigenvalues(const DiagonalMatrix<T,3>& eigenvalues,const T tolerance=1e-5) const
    {T scale=other::maxabs(eigenvalues.x00,eigenvalues.x22),scale_inverse=robust_inverse(scale),tiny=tolerance*scale;
    if(eigenvalues.x11-eigenvalues.x22>tiny) return ((*this-eigenvalues.x22)*scale_inverse).cofactor_matrix().largest_column_normalized();
    return ((*this-eigenvalues.x00)*scale_inverse).cofactor_matrix().largest_column().unit_orthogonal_vector();}

    SymmetricMatrix positive_definite_part() const
    {DiagonalMatrix<T,3> D;Matrix<T,3> V;fast_solve_eigenproblem(D,V);D=D.clamp_min(0);return conjugate(V,D);}

    DiagonalMatrix<T,3> diagonal_part() const
    {return DiagonalMatrix<T,3>(x00,x11,x22);}

    SymmetricMatrix<T,3> conjugate_with_cross_product_matrix(const Vector<T,3>& v) const
    {return cross_product_matrix_times(v).times_cross_product_matrix_with_symmetric_result(-v);}

//#####################################################################
    Matrix<T,3> operator*(const DiagonalMatrix<T,3>& A) const;
    Matrix<T,3> operator*(const UpperTriangularMatrix<T,3>& A) const;
    SymmetricMatrix operator+(const DiagonalMatrix<T,3>& A) const;
    DiagonalMatrix<T,3> fast_eigenvalues() const OTHER_CORE_EXPORT;
    void fast_solve_eigenproblem(DiagonalMatrix<T,3>& eigenvalues,Matrix<T,3>& eigenvectors) const OTHER_CORE_EXPORT;
    void solve_eigenproblem(DiagonalMatrix<T,3>& eigenvalues,Matrix<T,3>& eigenvectors) const;
private:
    static void jacobi_transform(const int sweep,const T threshold,T& app,T& apq,T& aqq,T& arp,T& arq,T& v1p,T& v1q,T& v2p,T& v2q,T& v3p,T& v3q);
//#####################################################################
};
// global functions
template<class T>
inline SymmetricMatrix<T,3> operator*(const T a,const SymmetricMatrix<T,3>& A)
{return A*a;}

template<class T>
inline SymmetricMatrix<T,3> operator+(const T a,const SymmetricMatrix<T,3>& A)
{return A+a;}

template<class T>
inline SymmetricMatrix<T,3> operator-(const T a,const SymmetricMatrix<T,3>& A)
{return -A+a;}

template<class T>
inline SymmetricMatrix<T,3> clamp(const SymmetricMatrix<T,3>& x,const SymmetricMatrix<T,3>& xmin,const SymmetricMatrix<T,3>& xmax)
{return SymmetricMatrix<T,3>(clamp(x.x00,xmin.x00,xmax.x00),clamp(x.x10,xmin.x10,xmax.x10),clamp(x.x20,xmin.x20,xmax.x20),clamp(x.x11,xmin.x11,xmax.x11),clamp(x.x21,xmin.x21,xmax.x21),clamp(x.x22,xmin.x22,xmax.x22));}

template<class T>
inline SymmetricMatrix<T,3> clamp_min(const SymmetricMatrix<T,3>& x,const SymmetricMatrix<T,3>& xmin)
{return SymmetricMatrix<T,3>(clamp_min(x.x00,xmin.x00),clamp_min(x.x10,xmin.x10),clamp_min(x.x20,xmin.x20),clamp_min(x.x11,xmin.x11),clamp_min(x.x21,xmin.x21),clamp_min(x.x22,xmin.x22));}

template<class T>
inline SymmetricMatrix<T,3> clamp_max(const SymmetricMatrix<T,3>& x,const SymmetricMatrix<T,3>& xmax)
{return SymmetricMatrix<T,3>(clamp_max(x.x00,xmax.x00),clamp_max(x.x10,xmax.x10),clamp_max(x.x20,xmax.x20),clamp_max(x.x11,xmax.x11),clamp_max(x.x21,xmax.x21),clamp_max(x.x22,xmax.x22));}

template<class T>
inline std::ostream& operator<< (std::ostream& output,const SymmetricMatrix<T,3>& A)
{output<<A.x00<<"\n"<<A.x10<<" "<<A.x11<<"\n"<<A.x20<<" "<<A.x21<<" "<<A.x22<<"\n";return output;}

template<class T>
inline SymmetricMatrix<T,3> log(const SymmetricMatrix<T,3>& A)
{DiagonalMatrix<T,3> D;Matrix<T,3> Q;A.fast_solve_eigenproblem(D,Q);return A.conjugate(Q,log(D));}

template<class T>
inline SymmetricMatrix<T,3> exp(const SymmetricMatrix<T,3>& A)
{DiagonalMatrix<T,3> D;Matrix<T,3> Q;A.fast_solve_eigenproblem(D,Q);return A.conjugate(Q,exp(D));}

template<class T> inline SymmetricMatrix<T,3>
outer_product(const Vector<T,3>& u) // 6 mults
{return SymmetricMatrix<T,3>(u.x*u.x,u.x*u.y,u.x*u.z,u.y*u.y,u.y*u.z,u.z*u.z);}

template<class T> inline SymmetricMatrix<T,3>
scaled_outer_product(const T a,const Vector<T,3>& u) // 9 mults
{Vector<T,3> au=a*u;return SymmetricMatrix<T,3>(au.x*u.x,au.x*u.y,au.x*u.z,au.y*u.y,au.y*u.z,au.z*u.z);}

template<class T> inline SymmetricMatrix<T,3>
conjugate(const Matrix<T,3>& A,const DiagonalMatrix<T,3>& B) // 27 mults, 12 adds
{
    return times_transpose_with_symmetric_result(A*B,A);
}

template<class T> inline SymmetricMatrix<T,3>
conjugate(const Matrix<T,3>& A,const SymmetricMatrix<T,3>& B)
{
    return times_transpose_with_symmetric_result(A*B,A);
}

template<class T> inline SymmetricMatrix<T,3>
conjugate(const SymmetricMatrix<T,3>& A,const SymmetricMatrix<T,3>& B)
{
    return times_transpose_with_symmetric_result(A*B,A);
}

template<class T> inline SymmetricMatrix<T,3>
conjugate(const Matrix<T,3,2>& A,const DiagonalMatrix<T,2>& B)
{
    return times_transpose_with_symmetric_result(A*B,A);
}

template<class T> inline SymmetricMatrix<T,3>
conjugate(const Matrix<T,3,2>& A,const SymmetricMatrix<T,2>& B)
{
    return times_transpose_with_symmetric_result(A*B,A);
}

template<class T> inline SymmetricMatrix<T,3>
conjugate_with_transpose(const Matrix<T,3>& A,const DiagonalMatrix<T,3>& B)
{
    return transpose_times_with_symmetric_result(B*A,A);
}

template<class T> inline SymmetricMatrix<T,3>
conjugate_with_transpose(const Matrix<T,3>& A,const SymmetricMatrix<T,3>& B)
{
    return transpose_times_with_symmetric_result(B*A,A);
}

template<class T> inline SymmetricMatrix<T,3>
conjugate_with_transpose(const UpperTriangularMatrix<T,3>& A,const SymmetricMatrix<T,3>& B)
{
    return transpose_times_with_symmetric_result(B*A,A);
}

template<class T> inline Matrix<T,3> SymmetricMatrix<T,3>::
operator*(const DiagonalMatrix<T,3>& A) const // 9 mults
{
    return Matrix<T,3>(x00*A.x00,x10*A.x00,x20*A.x00,x10*A.x11,x11*A.x11,x21*A.x11,x20*A.x22,x21*A.x22,x22*A.x22);
}

template<class T> inline Matrix<T,3> SymmetricMatrix<T,3>::
operator*(const UpperTriangularMatrix<T,3>& A) const // 18 mults, 9 adds
{
    return Matrix<T,3>(x00*A.x00,x10*A.x00,x20*A.x00,x00*A.x01+x10*A.x11,x10*A.x01+x11*A.x11,x20*A.x01+x21*A.x11,
                       x00*A.x02+x10*A.x12+x20*A.x22,x10*A.x02+x11*A.x12+x21*A.x22,x20*A.x02+x21*A.x12+x22*A.x22);
}

template<class T> inline Matrix<T,3>
operator*(const DiagonalMatrix<T,3>& D,const SymmetricMatrix<T,3>& A) // 9 mults
{
    return Matrix<T,3>(D.x00*A.x00,D.x11*A.x10,D.x22*A.x20,D.x00*A.x10,D.x11*A.x11,D.x22*A.x21,D.x00*A.x20,D.x11*A.x21,D.x22*A.x22);
}

template<class T> inline Matrix<T,3>
operator*(const UpperTriangularMatrix<T,3>& A,const SymmetricMatrix<T,3>& B) // 18 mults, 9 adds
{
    return Matrix<T,3>(A.x00*B.x00+A.x01*B.x10+A.x02*B.x20,A.x11*B.x10+A.x12*B.x20,A.x22*B.x20,A.x00*B.x10+A.x01*B.x11+A.x02*B.x21,
                       A.x11*B.x11+A.x12*B.x21,A.x22*B.x21,A.x00*B.x20+A.x01*B.x21+A.x02*B.x22,A.x11*B.x21+A.x12*B.x22,A.x22*B.x22);
}

template<class T> inline SymmetricMatrix<T,3> SymmetricMatrix<T,3>::
operator+(const DiagonalMatrix<T,3>& A) const // 3 adds
{
    return SymmetricMatrix<T,3>(x00+A.x00,x10,x20,x11+A.x11,x21,x22+A.x22);
}

template<class T> inline SymmetricMatrix<T,3>
operator+(const DiagonalMatrix<T,3>& A,const SymmetricMatrix<T,3>& B) // 3 adds
{
    return B+A;
}

template<class T> inline Matrix<T,3>
operator+(const SymmetricMatrix<T,3>& A,const UpperTriangularMatrix<T,3>& B)
{
    return Matrix<T,3>(A.x00+B.x00,A.x10,A.x20,A.x10+B.x01,A.x11+B.x11,A.x21,A.x20+B.x02,A.x21+B.x12,A.x22+B.x22);
}

template<class T> inline Matrix<T,3>
operator+(const UpperTriangularMatrix<T,3>& A,const SymmetricMatrix<T,3>& B)
{
    return B+A;
}

template<class T> inline Matrix<T,3>
operator-(const SymmetricMatrix<T,3>& A,const UpperTriangularMatrix<T,3>& B)
{
    return Matrix<T,3>(A.x00-B.x00,A.x10,A.x20,A.x10-B.x01,A.x11-B.x11,A.x21,A.x20-B.x02,A.x21-B.x12,A.x22-B.x22);
}

template<class T> inline Matrix<T,3>
operator-(const UpperTriangularMatrix<T,3>& A,const SymmetricMatrix<T,3>& B)
{
    return -B+A;
}

template<class T> inline SymmetricMatrix<T,3>
operator-(const DiagonalMatrix<T,3>& A,const SymmetricMatrix<T,3>& B) // 3 adds
{
    return SymmetricMatrix<T,3>(A.x00-B.x00,-B.x10,-B.x20,A.x11-B.x11,-B.x21,A.x22-B.x22);
}

template<class T> inline SymmetricMatrix<T,3>
multiply_with_symmetric_result(const Matrix<T,3>& A,const Matrix<T,3>& B) { // A*B and assume symmetric result, 18 mults, 12 adds
  return SymmetricMatrix<T,3>(A.x[0][0]*B.x[0][0]+A.x[0][1]*B.x[1][0]+A.x[0][2]*B.x[2][0],A.x[1][0]*B.x[0][0]+A.x[1][1]*B.x[1][0]+A.x[1][2]*B.x[2][0],
                              A.x[2][0]*B.x[0][0]+A.x[2][1]*B.x[1][0]+A.x[2][2]*B.x[2][0],A.x[1][0]*B.x[0][1]+A.x[1][1]*B.x[1][1]+A.x[1][2]*B.x[2][1],
                              A.x[2][0]*B.x[0][1]+A.x[2][1]*B.x[1][1]+A.x[2][2]*B.x[2][1],A.x[2][0]*B.x[0][2]+A.x[2][1]*B.x[1][2]+A.x[2][2]*B.x[2][2]);
}

template<class T> inline SymmetricMatrix<T,3>
times_transpose_with_symmetric_result(const Matrix<T,3>& A,const Matrix<T,3>& B) { // A*B^t and assume symmetric result, 18 mults, 12 adds
  return SymmetricMatrix<T,3>(A.x[0][0]*B.x[0][0]+A.x[0][1]*B.x[0][1]+A.x[0][2]*B.x[0][2],A.x[1][0]*B.x[0][0]+A.x[1][1]*B.x[0][1]+A.x[1][2]*B.x[0][2],
                              A.x[2][0]*B.x[0][0]+A.x[2][1]*B.x[0][1]+A.x[2][2]*B.x[0][2],A.x[1][0]*B.x[1][0]+A.x[1][1]*B.x[1][1]+A.x[1][2]*B.x[1][2],
                              A.x[2][0]*B.x[1][0]+A.x[2][1]*B.x[1][1]+A.x[2][2]*B.x[1][2],A.x[2][0]*B.x[2][0]+A.x[2][1]*B.x[2][1]+A.x[2][2]*B.x[2][2]);
}

template<class T> inline SymmetricMatrix<T,3>
times_transpose_with_symmetric_result(const Matrix<T,3,2>& A,const Matrix<T,3,2>& B) { // A*B^t and assume symmetric result, 12 mults, 6 adds
  return SymmetricMatrix<T,3>(A.x[0][0]*B.x[0][0]+A.x[0][1]*B.x[0][1],A.x[1][0]*B.x[0][0]+A.x[1][1]*B.x[0][1],A.x[2][0]*B.x[0][0]+A.x[2][1]*B.x[0][1],
                              A.x[1][0]*B.x[1][0]+A.x[1][1]*B.x[1][1],A.x[2][0]*B.x[1][0]+A.x[2][1]*B.x[1][1],A.x[2][0]*B.x[2][0]+A.x[2][1]*B.x[2][1]);
}

template<class T> inline SymmetricMatrix<T,3>
transpose_times_with_symmetric_result(const Matrix<T,3>& A,const Matrix<T,3>& B) { // A^t*B and assume symmetric result, 18 mults, 12 adds
  return SymmetricMatrix<T,3>(A.x[0][0]*B.x[0][0]+A.x[1][0]*B.x[1][0]+A.x[2][0]*B.x[2][0],A.x[0][1]*B.x[0][0]+A.x[1][1]*B.x[1][0]+A.x[2][1]*B.x[2][0],A.x[0][2]*B.x[0][0]+A.x[1][2]*B.x[1][0]+A.x[2][2]*B.x[2][0],
                              A.x[0][1]*B.x[0][1]+A.x[1][1]*B.x[1][1]+A.x[2][1]*B.x[2][1],A.x[0][2]*B.x[0][1]+A.x[1][2]*B.x[1][1]+A.x[2][2]*B.x[2][1],A.x[0][2]*B.x[0][2]+A.x[1][2]*B.x[1][2]+A.x[2][2]*B.x[2][2]);
}

template<class T> inline SymmetricMatrix<T,3>
transpose_times_with_symmetric_result(const Matrix<T,3>& A,const UpperTriangularMatrix<T,3>& B) { // A^t*B and assume symmetric result, 10 mults, 4 adds
  return SymmetricMatrix<T,3>(A.x[0][0]*B.x00,A.x[0][1]*B.x00,A.x[0][2]*B.x00,A.x[0][1]*B.x01+A.x[1][1]*B.x11,A.x[0][2]*B.x01+A.x[1][2]*B.x11,A.x[0][2]*B.x02+A.x[1][2]*B.x12+A.x[2][2]*B.x22);
}

template<class T> inline T
inner_product(const SymmetricMatrix<T,3>& A,const SymmetricMatrix<T,3>& B) {
  return A.x00*B.x00+A.x11*B.x11+A.x22*B.x22+2*(A.x10*B.x10+A.x20*B.x20+A.x21*B.x21);
}

}
