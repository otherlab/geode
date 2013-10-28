//#####################################################################
// Class SymmetricMatrix3x3
//#####################################################################
#include <geode/vector/DiagonalMatrix.h>
#include <geode/vector/Matrix.h>
#include <geode/vector/SymmetricMatrix.h>
#include <geode/vector/UpperTriangularMatrix3x3.h>
#include <geode/math/maxabs.h>
#include <geode/utility/debug.h>
#include <limits>
namespace geode {
//#####################################################################
// Function Fast_Eigenvalues
//#####################################################################
// lambda_x > lambda_y > lambda_z
// Reference: Smith, O. "Eigenvalues of a symmetric 3 x 3 matrix". Commun. ACM 4 (4), p. 168, 1961 (thanks, Gene)
template<class T> DiagonalMatrix<T,3> SymmetricMatrix<T,3>::
fast_eigenvalues() const // 24 mults, 20 adds, 1 atan2, 1 sincos, 2 sqrts
{
    if(!boost::is_same<T,double>::value) return DiagonalMatrix<T,3>(SymmetricMatrix<double,3>(*this).fast_eigenvalues());
    // now T is double
    T m=T(1./3)*(x00+x11+x22);
    T a00=x00-m,a11=x11-m,a22=x22-m,a01_sqr=x10*x10,a02_sqr=x20*x20,a12_sqr=x21*x21;
    T p=T(1./6)*(a00*a00+a11*a11+a22*a22+2*(a01_sqr+a02_sqr+a12_sqr));
    T q=(T).5*(a00*(a11*a22-a12_sqr)-a11*a02_sqr-a22*a01_sqr)+x10*x20*x21;
    T sqrt_p=sqrt(p),disc=p*p*p-q*q;
    T phi=T(1./3)*atan2(sqrt(max((T)0,disc)),q),c=cos(phi),s=sin(phi);
    T sqrt_p_cos=sqrt_p*c,root_three_sqrt_p_sin=(T)sqrt(3)*sqrt_p*s;
    DiagonalMatrix<T,3> lambda(m+2*sqrt_p_cos,m-sqrt_p_cos-root_three_sqrt_p_sin,m-sqrt_p_cos+root_three_sqrt_p_sin);
    small_sort(lambda.x22,lambda.x11,lambda.x00);return lambda;
}
//#####################################################################
// Function Fast_Eigenvectors
//#####################################################################
namespace{
template<class T> Matrix<T,3>
fast_eigenvectors(const SymmetricMatrix<T,3>& A,const DiagonalMatrix<T,3>& lambda) // 71 mults, 44 adds, 3 divs, 3 sqrts
{
    if(!boost::is_same<T,double>::value) GEODE_FATAL_ERROR();
    // T is now always double

    // flip if necessary so that first eigenvalue is the most different
    bool flipped=false;
    DiagonalMatrix<T,3> lambda_flip(lambda);
    if(lambda.x00-lambda.x11<lambda.x11-lambda.x22){ // 2a
        swap(lambda_flip.x00,lambda_flip.x22);
        flipped=true;}

    // get first eigenvector
    Vector<T,3> v0=(A-lambda_flip.x00).cofactor_matrix().largest_column_normalized(); // 3a + 12m+6a + 9m+6a+1d+1s = 21m+15a+1d+1s

    // form basis for orthogonal complement to v0, and reduce A to this space
    Vector<T,3> v0_orthogonal=v0.unit_orthogonal_vector(); // 6m+2a+1d+1s (tweak: 5m+1a+1d+1s)
    Matrix<T,3,2> other_v=HStack(v0_orthogonal,cross(v0,v0_orthogonal)); // 6m+3a (tweak: 4m+1a)
    SymmetricMatrix<T,2> A_reduced=conjugate_with_transpose(other_v,A); // 21m+12a (tweak: 18m+9a)

    // find third eigenvector from A_reduced, and fill in second via cross product
    Vector<T,3> v2=other_v*(A_reduced-lambda_flip.x22).cofactor_matrix().largest_column_normalized(); // 6m+3a + 2a + 5m+2a+1d+1s = 11m+7a+1d+1s (tweak: 10m+6a+1d+1s)
    Vector<T,3> v1=cross(v2,v0); // 6m+3a

    // finish
    return flipped?Matrix<T,3>(v2,v1,-v0):Matrix<T,3>(v0,v1,v2);
}}
//#####################################################################
// Function Fast_Solve_Eigenproblem
//#####################################################################
template<class T> void SymmetricMatrix<T,3>::
fast_solve_eigenproblem(DiagonalMatrix<T,3>& eigenvalues,Matrix<T,3>& eigenvectors) const // roughly 95 mults, 64 adds, 3 divs, 5 sqrts, 1 atan2, 1 sincos
{
    if(!boost::is_same<T,double>::value){
        DiagonalMatrix<double,3> eigenvalues_double;Matrix<double,3> eigenvectors_double;
        SymmetricMatrix<double,3>(*this).fast_solve_eigenproblem(eigenvalues_double,eigenvectors_double);
        eigenvalues=DiagonalMatrix<T,3>(eigenvalues_double);eigenvectors=Matrix<T,3>(eigenvectors_double);return;}
    // now T is double
    eigenvalues=fast_eigenvalues();
    eigenvectors=fast_eigenvectors(*this,eigenvalues);
}
//#####################################################################
// Function Solve_Eigenproblem
//####################################################################
template<class T> void SymmetricMatrix<T,3>::
solve_eigenproblem(DiagonalMatrix<T,3>& eigenvalues,Matrix<T,3>& eigenvectors) const
{
    T a00=x00,a01=x10,a02=x20,a11=x11,a12=x21,a22=x22;
    T v00=1,v01=0,v02=0,v10=0,v11=1,v12=0,v20=0,v21=0,v22=1;
    int sweep;for(sweep=1;sweep<=50;sweep++){
        T sum=abs(a01)+abs(a02)+abs(a12);
        if(sum==0) break;
        T threshold=sweep<4?(T)(1./45)*sum:0;
        jacobi_transform(sweep,threshold,a00,a01,a11,a02,a12,v00,v01,v10,v11,v20,v21);
        jacobi_transform(sweep,threshold,a00,a02,a22,a01,a12,v00,v02,v10,v12,v20,v22);
        jacobi_transform(sweep,threshold,a11,a12,a22,a01,a02,v01,v02,v11,v12,v21,v22);}
    assert(sweep<=50);
    eigenvalues=DiagonalMatrix<T,3>(a00,a11,a22);
    eigenvectors=Matrix<T,3>(v00,v10,v20,v01,v11,v21,v02,v12,v22);
}
//#####################################################################
// Function Jacobi_Transform
//#####################################################################
template<class T> inline void SymmetricMatrix<T,3>::
jacobi_transform(const int sweep,const T threshold,T& app,T& apq,T& aqq,T& arp,T& arq,T& v1p,T& v1q,T& v2p,T& v2q,T& v3p,T& v3q)
{
    T epsilon=std::numeric_limits<T>::epsilon();
    T g=100*abs(apq);
    if(sweep > 4 && epsilon*abs(app)>=g && epsilon*abs(aqq)>=g) apq=0;
    else if(abs(apq) > threshold){
        T h=aqq-app,t;
        if(epsilon*abs(h)>=g) t=apq/h;
        else{T theta=(T).5*h/apq;t=1/(abs(theta)+sqrt(1+sqr(theta)));if(theta < 0) t=-t;}
        T cosine=1/sqrt(1+t*t),sine=t*cosine,tau=sine/(1+cosine),da=t*apq;
        app-=da;aqq+=da;apq=0;
        T arp_tmp=arp-sine*(arq+tau*arp);arq=arq+sine*(arp-tau*arq);arp=arp_tmp;
        T v1p_tmp=v1p-sine*(v1q+tau*v1p);v1q=v1q+sine*(v1p-tau*v1q);v1p=v1p_tmp;
        T v2p_tmp=v2p-sine*(v2q+tau*v2p);v2q=v2q+sine*(v2p-tau*v2q);v2p=v2p_tmp;
        T v3p_tmp=v3p-sine*(v3q+tau*v3p);v3q=v3q+sine*(v3p-tau*v3q);v3p=v3p_tmp;}
}
//#####################################################################
template class SymmetricMatrix<real,3>;
}
