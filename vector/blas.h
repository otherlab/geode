//#####################################################################
// Header blas
//#####################################################################
#ifndef __blas_wrap_iterating__
#ifndef __blas_wrap_h__
#define __blas_wrap_h__

#ifdef USE_MKL
#include <mkl_lapack.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_trans.h>
#else
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif
#endif
#include <other/core/array/RawArray.h>
#include <other/core/array/Array2d.h>
#include <other/core/geometry/Box.h>
#include <boost/preprocessor/cat.hpp>
namespace other{

template<class T> int ilaenv(int ispec,const char* name,const char* opts,int m,int n) OTHER_EXPORT;
template<> struct FromPython<CBLAS_TRANSPOSE>{OTHER_EXPORT static CBLAS_TRANSPOSE convert(PyObject* object);};

#define WRAP(declaration) \
    static declaration OTHER_UNUSED; \
    static declaration
#define CBC(name) BOOST_PP_CAT(cblas_,BOOST_PP_CAT(a,name))

#ifdef USE_MKL
#define DECLARE(name,...)
#define BC(name) BOOST_PP_CAT(a,name)
#else
#define DECLARE(name,...) extern "C" { void BOOST_PP_CAT(a,BOOST_PP_CAT(name,_))(__VA_ARGS__); }
#define BC(name) BOOST_PP_CAT(a,BOOST_PP_CAT(name,_))
#endif

#define __blas_wrap_iterating__

#define a s
#define sa "a"
#define T float
#include "blas.h"
#undef a
#undef sa
#undef T

#define a d
#define sa "d"
#define T double
#include "blas.h"
#undef a
#undef sa
#undef T

#undef __blas_wrap_iterating__
#undef CBC
#undef BC
#undef WRAP
#undef DECLARE

}
#endif
#else // __blas_wrap_iterating__

WRAP(void scal(int n,T alpha,T* x)) // x = ax
{
    if(alpha) CBC(scal)(n,alpha,x,1);
    else memset(x,0,n*sizeof(T)); // mkl scal doesn't work for alpha = 0
}

WRAP(void scal(T alpha,RawArray<T> x)) // x = ax
{
    scal(x.size(),alpha,x.data());
}

WRAP(void scal(T alpha,Subarray<T> x)) // x = ax
{
    if(alpha) CBC(scal)(x.m,alpha,x.data(),x.stride);
    else x.fill(0);
}

DECLARE(geqrf,int*,int*,T*,int*,T*,T*,int*,int*)
DECLARE(gelqf,int*,int*,T*,int*,T*,T*,int*,int*)

WRAP(void geqrf(CBLAS_UPLO uplo,RawArray<T,2> A,RawArray<T> tau,Array<T>& work))
{
    int info,lda=max(1,A.n),r=min(A.m,A.n),lwork=work.max_size();
    OTHER_ASSERT(tau.size()==r);if(!r) return; // lapack requires tau.size()=1 here, so do nothing instead
    (uplo==CblasUpper?BC(gelqf):BC(geqrf))(const_cast<int*>(&A.n),const_cast<int*>(&A.m),A.data(),&lda,tau.data(),work.data(),&lwork,&info);
    OTHER_ASSERT(!info);
}

WRAP(int geqrf_work(CBLAS_UPLO uplo,RawArray<T,2> A))
{
    if(!A.m || !A.n) return 0;
    T work;int info,lda=max(1,A.n),lwork=-1;
    (uplo==CblasUpper?BC(gelqf):BC(geqrf))(const_cast<int*>(&A.n),const_cast<int*>(&A.m),0,&lda,0,&work,&lwork,&info);
    return max((int)work,A.m,A.n); // Mkl returns the wrong work size, so grow it if necessary
}

DECLARE(geqp3,int*,int*,T*,int*,int*,T*,T*,int*,int*)

WRAP(void geqp3(RawArray<T,2> A,RawArray<int> p,RawArray<T> tau,Array<T>& work))
{
    OTHER_ASSERT(tau.size()==max(1,min(A.m,A.n)) && p.size()==A.m);
    int info,lda=max(1,A.n),lwork=work.max_size();p.fill(0);
    BC(geqp3)(const_cast<int*>(&A.n),const_cast<int*>(&A.m),A.data(),&lda,p.data(),tau.data(),work.data(),&lwork,&info);
}

WRAP(int geqp3_work(RawArray<T,2> A))
{
    T work;int info,lda=max(1,A.n),lwork=-1;
    BC(geqp3)(const_cast<int*>(&A.n),const_cast<int*>(&A.m),0,&lda,0,0,&work,&lwork,&info);
    return (int)work;
}

DECLARE(ormqr,char*,char*,int*,int*,int*,T*,int*,T*,T*,int*,T*,int*,int*)
DECLARE(ormlq,char*,char*,int*,int*,int*,T*,int*,T*,T*,int*,T*,int*,int*)

WRAP(void ormqr(CBLAS_UPLO uplo,CBLAS_SIDE s,CBLAS_TRANSPOSE t,Subarray<const T,2> A,RawArray<const T> tau,Subarray<T,2> C,Array<T>& work))
{
    const char *side=s==CblasLeft?"R":"L",*trans=t==CblasTrans?"T":"N"; // side flipped due to row-major vs. column major
    int k=min(A.m,A.n),c=s==CblasLeft?C.m:C.n,lda=max(1,A.stride),ldc=max(1,C.stride),lwork=work.max_size(),info;
    OTHER_ASSERT(k<=c && c<=(uplo==CblasUpper?A.m:A.n) && tau.size()==k);
    if(!k) return; // lapack requires tau.size()=1 here, so do nothing instead
    (uplo==CblasUpper?BC(ormlq):BC(ormqr))(const_cast<char*>(side),const_cast<char*>(trans),const_cast<int*>(&C.n),const_cast<int*>(&C.m),&k,
        const_cast<T*>(A.data()),&lda,const_cast<T*>(tau.data()),C.data(),&ldc,work.data(),&lwork,&info);
    OTHER_ASSERT(!info);
}

WRAP(int ormqr_work(CBLAS_UPLO uplo,CBLAS_SIDE s,CBLAS_TRANSPOSE t,Subarray<const T,2> A,Subarray<T,2> C))
{
    const char *side=s==CblasLeft?"R":"L",*trans=t==CblasTrans?"T":"N"; // side flipped due to row-major vs. column major
    int k=min(A.m,A.n),lda=max(1,A.stride),ldc=max(1,C.stride),lwork=-1,info;T work;
    OTHER_ASSERT((uplo==CblasUpper?A.m:A.n)>=(s==CblasLeft?C.m:C.n));
    if(!k) return 0; // lapack requires tau.size()=1 here, so do nothing instead
    (uplo==CblasUpper?BC(ormlq):BC(ormqr))(const_cast<char*>(side),const_cast<char*>(trans),const_cast<int*>(&C.n),const_cast<int*>(&C.m),&k,0,&lda,0,0,&ldc,&work,&lwork,&info);
    OTHER_ASSERT(!info);
    return (int)work;
}

WRAP(void ormqr(CBLAS_UPLO uplo,CBLAS_SIDE s,CBLAS_TRANSPOSE t,Subarray<const T,2> A,RawArray<const T> tau,RawArray<T> C,Array<T>& work)) // vector case
{
    int r=C.size(),c=1;if(s==CblasRight) swap(r,c);
    ormqr(uplo,s,t,A,tau,C.reshape(r,c),work);
}

WRAP(int ormqr_work(CBLAS_UPLO uplo,CBLAS_SIDE s,CBLAS_TRANSPOSE t,Subarray<const T,2> A,RawArray<T> C)) // C = vector case
{
    int r=C.size(),c=1;if(s==CblasRight) swap(r,c);
    return ormqr_work(uplo,s,t,A,C.reshape(r,c));
}

DECLARE(orgqr,int*,int*,int*,T*,int*,T*,T*,int*,int*)
DECLARE(orglq,int*,int*,int*,T*,int*,T*,T*,int*,int*)

WRAP(void orgqr(CBLAS_UPLO uplo,int r,Subarray<T,2> A,RawArray<const T> tau,Array<T>& work))
{
    int info,m=A.m,n=A.n,lda=max(1,A.stride),lwork=work.max_size();
    OTHER_ASSERT(r<=min(m,n) && (uplo==CblasUpper?n<=m:m<=n) && tau.size()==r);
    (uplo==CblasUpper?BC(orglq):BC(orgqr))(&n,&m,&r,A.data(),&lda,const_cast<T*>(tau.data()),work.data(),&lwork,&info);
}

WRAP(int orgqr_work(CBLAS_UPLO uplo,int r,Subarray<T,2> A))
{
    T work;int info,m=A.m,n=A.n,lda=max(1,A.stride),lwork=-1;
    OTHER_ASSERT(r<=min(m,n) && (uplo==CblasUpper?n<=m:m<=n));
    (uplo==CblasUpper?BC(orglq):BC(orgqr))(&n,&m,&r,0,&lda,0,&work,&lwork,&info);
    return (int)work;
}

// y = alpha*A^op x + beta*y
WRAP(void gemv(CBLAS_TRANSPOSE t,T alpha,Subarray<const T,2> A,const T* x,T beta,T* y))
{
    int m=A.m,n=A.n;if(t==CblasTrans) swap(m,n);
    if(!n) scal(m,beta,y); // work around bug in mkl
    else CBC(gemv)(CblasRowMajor,t,A.m,A.n,alpha,A.data(),max(1,A.stride),x,1,beta,y,1);
}

// y = alpha*A^op x + beta*y
WRAP(void gemv(CBLAS_TRANSPOSE t,T alpha,Subarray<const T,2> A,RawArray<const T> x,T beta,RawArray<T> y))
{
    int m=A.m,n=A.n;if(t==CblasTrans) swap(m,n);
    OTHER_ASSERT(x.size()==n && m==y.size());
    gemv(t,alpha,A,x.data(),beta,y.data());
}

WRAP(void gemv(T alpha,Subarray<const T,2> A,const T* x,T beta,T* y))
{gemv(CblasNoTrans,alpha,A,x,beta,y);}

WRAP(void gemv(T alpha,Subarray<const T,2> A,RawArray<const T> x,T beta,RawArray<T> y))
{gemv(CblasNoTrans,alpha,A,x,beta,y);}

WRAP(void symv(CBLAS_UPLO uplo,T alpha,Subarray<const T,2> A,RawArray<const T> x,T beta,RawArray<T> y))
{
    OTHER_ASSERT(A.m==A.n && x.size()==A.m && y.size()==A.m);
    CBC(symv)(CblasRowMajor,uplo,A.m,alpha,A.data(),max(1,A.stride),x.data(),1,beta,y.data(),1);
}

WRAP(void gbmv(CBLAS_TRANSPOSE t,int m,int n,int kl,int ku,T alpha,RawArray<const T> A,const T* x,T beta,T* y)) // y = alpha*A^op+beta*y
{
    int lda=kl+ku+1;OTHER_ASSERT(kl>=0 && ku>=0 && A.size()==n*lda);
    CBC(gbmv)(CblasColMajor,t,m,n,kl,ku,alpha,A.data(),lda,x,1,beta,y,1);
}

DECLARE(gesdd,char*,int*,int*,T*,int*,T*,T*,int*,T*,int*,T*,int*,int*,int*)

WRAP(void gesdd(Array<T,2>& A,Array<T>& s,Array<T,2>& U,Array<T,2>& Vt,Array<T>& work,bool all))
{
    char job=all?'A':'S';int r=min(A.m,A.n),liwork=8*r,lwork=(work.max_size()*sizeof(T)-liwork*sizeof(int))/sizeof(T),info;
    U.resize(A.m,all?A.m:r);Vt.resize(all?A.n:r,A.n);s.resize(r);
    BC(gesdd)(&job,&A.n,&A.m,A.data(),&A.n,s.data(),Vt.data(),&Vt.n,U.data(),&U.n,work.data(),&lwork,reinterpret_cast<int*>(work.data()+lwork),&info);
}

WRAP(int gesdd_work(Array<T,2>& A,Array<T,2>& U,Array<T,2>& Vt,bool all))
{
    T work;char job=all?'A':'S';int r=min(A.m,A.n),liwork=max(1,8*r),lwork=-1,iwork,info;
    BC(gesdd)(&job,&A.n,&A.m,0,&A.n,0,0,&A.n,0,&A.m,&work,&lwork,&iwork,&info);
    return int(work+(liwork*sizeof(int)+sizeof(T)-1)/sizeof(T));
}

WRAP(void gesdd(Array<T,2>& A,Array<T>& s,Array<T>& work))
{
    char job='N';int r=min(A.m,A.n),one=1,info,liwork=8*r,lwork=(work.max_size()*sizeof(T)-liwork*sizeof(int))/sizeof(T);
    s.resize(r);
    BC(gesdd)(&job,&A.n,&A.m,A.data(),&A.n,s.data(),0,&one,0,&one,work.data(),&lwork,reinterpret_cast<int*>(work.data()+lwork),&info);
}

WRAP(int gesdd_work(Array<T,2>& A))
{
    char job='N';T work;int r=min(A.m,A.n),one=1,info,liwork=8*r,iwork,lwork=-1;
    BC(gesdd)(&job,&A.n,&A.m,0,&A.n,0,0,&one,0,&one,&work,&lwork,&iwork,&info);
    return int(work+(liwork*sizeof(int)+sizeof(T)-1)/sizeof(T));
}

DECLARE(syevd,char*,char*,int*,T*,int*,T*,T*,int*,int*,int*,int*)

WRAP(void syevd(CBLAS_UPLO uplo,Subarray<T,2> A,Array<T>& w,Array<T>& work))
{
    int n=A.n;
    OTHER_ASSERT(A.m==n);
    T lwork_;int liwork,query=-1,lda=A.stride,info;
    char *job=const_cast<char*>("N"),*uplo_=const_cast<char*>(uplo==CblasLower?"U":"L");
    BC(syevd)(job,uplo_,&n,A.data(),&lda,0,&lwork_,&query,&liwork,&query,&info);
    int lwork=(int)lwork_;
    OTHER_ASSERT(lwork*sizeof(T)+liwork*sizeof(int)>=work.max_size()*sizeof(T));
    w.resize(n);
    BC(syevd)(job,uplo_,&n,A.data(),&lda,w.data(),work.data(),&lwork,reinterpret_cast<int*>(work.data()+lwork),&liwork,&info);
}

WRAP(int syevd_work(CBLAS_UPLO uplo,Subarray<const T,2> A))
{
    int n=A.n;
    OTHER_ASSERT(A.m==n);
    T work;int iwork,lda=A.stride,info,lwork=-1;
    char *job=const_cast<char*>("N"),*uplo_=const_cast<char*>(uplo==CblasLower?"U":"L");
    BC(syevd)(job,uplo_,&n,const_cast<T*>(A.data()),&lda,0,&work,&lwork,&iwork,&lwork,&info);
    return int(work+(iwork*sizeof(int)+sizeof(T)-1)/sizeof(T));
}

DECLARE(stedc,char*,int*,T*,T*,T*,int*,T*,int*,int*,int*,int*)

WRAP(void stedc(RawArray<T> d,RawArray<T> e,Array<T,2>& z,Array<char>& work))
{
    int n=d.size();
    OTHER_ASSERT(n-1<=e.size() && e.size()<=n);
    z.resize(n,n,false);
    if(!n) return;
    char *compz=const_cast<char*>("I");
    T lwork_T;int liwork,query=-1,ldz=z.n,info;
    BC(stedc)(compz,&n,d.data(),e.data(),z.data(),&ldz,&lwork_T,&query,&liwork,&query,&info);
    OTHER_ASSERT(!info);
    int lwork=(int)lwork_T;
    BOOST_STATIC_ASSERT(sizeof(T)/sizeof(int)*sizeof(int)==sizeof(T));
    work.preallocate(lwork*sizeof(T)+liwork*sizeof(int));
    BC(stedc)(compz,&n,d.data(),e.data(),z.data(),&ldz,reinterpret_cast<T*>(work.data()),&lwork,
        reinterpret_cast<int*>(work.data()+lwork*sizeof(T)),&liwork,&info);
    OTHER_ASSERT(!info);
}

DECLARE(stegr,char*,char*,int*,T*,T*,T*,T*,int*,int*,T*,int*,T*,T*,int*,int*,T*,int*,int*,int*,int*)

// warning: does not work in Mkl 10.2.3
WRAP(void stegr(RawArray<T> d,RawArray<T> e,Box<T> range,Array<T>& w,Array<T,2>& z,Array<int>& isuppz,Array<char>& work))
{
    int n=d.size();
    OTHER_ASSERT(n==e.size());
    w.resize(n,false);z.resize(n,n,false);isuppz.resize(2*n,false);
    if(!n) return;
    char *jobz=const_cast<char*>("V"),*ran=const_cast<char*>(range.contains(Box<T>::full_box())?"A":"V");
    T lwork_T,unused_T;int liwork,m,query=-1,unused,ldz=z.n,info;
    BC(stegr)(jobz,ran,&n,d.data(),e.data(),&range.min,&range.max,&unused,&unused,&unused_T,&m,w.data(),z.data(),&ldz,
        isuppz.data(),&lwork_T,&query,&liwork,&query,&info);
    OTHER_ASSERT(!info);
    int lwork=(int)lwork_T;
    BOOST_STATIC_ASSERT(sizeof(T)/sizeof(int)*sizeof(int)==sizeof(T));
    work.preallocate(lwork*sizeof(T)+liwork*sizeof(int));
    BC(stegr)(jobz,ran,&n,d.data(),e.data(),&range.min,&range.max,&unused,&unused,&unused_T,&m,w.data(),z.data(),&ldz,
        isuppz.data(),reinterpret_cast<T*>(work.data()),&lwork,reinterpret_cast<int*>(work.data()+lwork*sizeof(T)),&liwork,&info);
    OTHER_ASSERT(!info);
    w.resize(m);z.resize(m,n);isuppz.resize(m);
}

WRAP(void gemm(CBLAS_TRANSPOSE ta,CBLAS_TRANSPOSE tb,T alpha,Subarray<const T,2> A,Subarray<const T,2> B,T beta,Subarray<T,2> C)) // C = alpha*A^ta B^tb+beta*C
{
    int k=ta==CblasTrans?A.m:A.n;
    OTHER_ASSERT(C.m==A.m+A.n-k && k==(tb==CblasTrans?B.n:B.m) && B.m+B.n-k==C.n);
    int lda=max(1,A.stride),ldb=max(1,B.stride),ldc=max(1,C.stride);
    CBC(gemm)(CblasRowMajor,ta,tb,C.m,C.n,k,alpha,A.data(),lda,B.data(),ldb,beta,C.data(),ldc);
}

WRAP(void gemm(T alpha,Subarray<const T,2> A,Subarray<const T,2> B,T beta,Subarray<T,2> C)) // C = alpha*AB+beta*C
{
    gemm(CblasNoTrans,CblasNoTrans,alpha,A,B,beta,C);
}

WRAP(void syrk(CBLAS_UPLO uplo,CBLAS_TRANSPOSE t,T alpha,Subarray<const T,2> A,T beta,Subarray<T,2> C)) // C = alpha*A^t A^!t + beta*C
{
    int k=t==CblasTrans?A.m:A.n,lda=max(A.stride,1),ldc=max(C.stride,1);
    OTHER_ASSERT(C.m==C.n && C.m==A.m+A.n-k);
    CBC(syrk)(CblasRowMajor,uplo,t,C.m,k,alpha,A.data(),lda,beta,C.data(),ldc);
}

DECLARE(gels,char*,int*,int*,int*,T*,int*,T*,int*,T*,int*,int*)

WRAP(void gels(char t,Subarray<T,2> A,RawArray<T> b,Array<T>& work))
{
    OTHER_ASSERT((t=='N' || t=='T') && max(A.m,A.n)==b.size() && (t=='N'?A.m:A.n)==b.size());
    char ot=t=='N'?'T':'N';int one=1,ldb=max(1,b.size()),lwork=work.size(),info;
    if(!lwork){work.resize(1);lwork=-1;}
    BC(gels)(&ot,const_cast<int*>(&A.n),const_cast<int*>(&A.m),&one,A.data(),const_cast<int*>(&A.stride),b.data(),&ldb,work.data(),&lwork,&info);
    if(lwork<0) work.resize(max(1,(int)work[0]));
}

DECLARE(gelsy,int*,int*,int*,T*,int*,T*,int*,int*,T*,int*,T*,int*,int*)

WRAP(void gelsy(Subarray<T,2> A,RawArray<T> b,Array<T>& work)) // solves A^T x = b
{
    OTHER_ASSERT(max(A.m,A.n)==b.size() && A.n==b.size());
    int one=1,ldb=max(1,b.size()),lwork=work.size(),info;
    if(!lwork){work.resize(1);lwork=-1;}
    Array<int> jpvt(A.m);T rcond;int rank;
    BC(gelsy)(const_cast<int*>(&A.n),const_cast<int*>(&A.m),&one,A.data(),const_cast<int*>(&A.stride),b.data(),&ldb,jpvt.data(),&rcond,&rank,work.data(),&lwork,&info);
    if(lwork<0) work.resize(max(1,(int)work[0]));
}

DECLARE(gelsd,int*,int*,int*,T*,int*,T*,int*,T*,T*,int*,T*,int*,int*,int*)

WRAP(void gelsd(Subarray<T,2> A,RawArray<T> b,Array<T>& work,Array<int>& iwork)) // solves A^T x = b
{
    int m=A.m,n=A.n;OTHER_ASSERT(max(m,n)==b.size() && n==b.size());
    int one=1,ldb=max(1,b.size()),lwork=work.size(),info;
    if(!lwork){work.resize(1);iwork.resize(1);lwork=-1;}
    Array<T> s(max(1,min(m,n)),false);T rcond;int rank;
    BC(gelsd)(&n,&m,&one,A.data(),const_cast<int*>(&A.stride),b.data(),&ldb,s.data(),&rcond,&rank,work.data(),&lwork,iwork.data(),&info);
    if(lwork<0){work.resize(max(1,(int)work[0]));iwork.resize(max(1,(int)iwork[0]));}
}

WRAP(void axpy(int n,T alpha,const T* x,T* y)) // y = ax + y
{
    CBC(axpy)(n,alpha,x,1,y,1);
}

WRAP(void axpy(T alpha,RawArray<const T> x,RawArray<T> y)) // y = ax + y
{
    int n=x.size();OTHER_ASSERT(n==y.size());
    CBC(axpy)(n,alpha,x.data(),1,y.data(),1);
}

WRAP(T nrm2(RawArray<const T> x))
{
    return CBC(nrm2)(x.size(),x.data(),1);
}

WRAP(T asum(RawArray<const T> x))
{
    return CBC(asum)(x.size(),x.data(),1);
}

WRAP(T dot(RawArray<const T> x,RawArray<const T> y))
{
    int n=x.size();OTHER_ASSERT(n==y.size());
    return CBC(dot)(n,x.data(),1,y.data(),1);
}

WRAP(void blas_swap(Subarray<T> x,Subarray<T> y))
{
    int n=x.size();OTHER_ASSERT(n==y.size());CBC(swap)(n,x.data(),x.stride,y.data(),y.stride);
}

WRAP(void ger(T alpha,Subarray<const T> x,Subarray<const T> y,Subarray<T,2> A))
{
    OTHER_ASSERT(x.size()==A.m && A.n==y.size());
    int lda=max(1,A.stride),xinc=max(1,x.stride),yinc=max(1,y.stride);
    CBC(ger)(CblasRowMajor,A.m,A.n,alpha,x.data(),xinc,y.data(),yinc,A.data(),lda);
}

DECLARE(laset,char*,int*,int*,T*,T*,T*,int*)

WRAP(void laset(char uplo,T offdiagonal,T diagonal,Subarray<T,2> A))
{
    char uplo_s[2]={uplo,0};int lda=max(1,A.stride);
    BC(laset)(uplo_s,const_cast<int*>(&A.n),const_cast<int*>(&A.m),&offdiagonal,&diagonal,A.data(),&lda);
}

WRAP(void trmm(CBLAS_SIDE side,CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA,CBLAS_DIAG diag,T alpha,Subarray<const T,2> A,Subarray<T,2> B))
{
    int k=side==CblasLeft?B.m:B.n;
    OTHER_ASSERT(A.m>=k && A.n>=k && (uplo==CblasLower?A.m:A.n)==k);
    CBC(trmm)(CblasRowMajor,side,uplo,transA,diag,B.m,B.n,alpha,A.data(),max(1,A.stride),B.data(),max(1,B.stride));
}

WRAP(void trsm(CBLAS_SIDE side,CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA,CBLAS_DIAG diag,T alpha,Subarray<const T,2> A,Subarray<T,2> B))
{
    int k=side==CblasLeft?B.m:B.n;
    OTHER_ASSERT(A.m>=k && A.n>=k && (uplo==CblasLower?A.m:A.n)==k);
    CBC(trsm)(CblasRowMajor,side,uplo,transA,diag,B.m,B.n,alpha,A.data(),max(1,A.stride),B.data(),max(1,B.stride));
}

WRAP(void trsm(CBLAS_SIDE side,CBLAS_UPLO uplo,CBLAS_TRANSPOSE transA,CBLAS_DIAG diag,T alpha,Subarray<const T,2> A,RawArray<T> B))
{
    int r=B.size(),c=1;if(side==CblasRight) swap(r,c);
    trsm(side,uplo,transA,diag,alpha,A,B.reshape(r,c));
}

DECLARE(gesv,int*,int*,T*,int*,int*,T*,int*,int*)

WRAP(void gesv(RawArray<T,2> A,RawArray<T> b)) // warning: solve transposed system
{
    OTHER_ASSERT(A.m==A.n && A.m==b.size());
    int one=1,lda=max(1,A.n),info;
    Array<int> ipiv(A.m,false);
    BC(gesv)(const_cast<int*>(&A.m),&one,A.data(),&lda,ipiv.data(),b.data(),const_cast<int*>(&A.n),&info);
}

DECLARE(getrf,int*,int*,T*,int*,int*,int*)

WRAP(void getrf(RawArray<T,2> A,RawArray<int> ipiv))
{
    int r=min(A.m,A.n),info;OTHER_ASSERT(ipiv.size()==r);
    if(!r) return;
    BC(getrf)(&A.n,&A.m,A.data(),&A.n,ipiv.data(),&info);
}

DECLARE(gecon,char*,int*,T*,int*,T*,T*,T*,int*,int*)

WRAP(T gecon(char norm,RawArray<T,2> A,T anorm))
{
    OTHER_ASSERT(A.m==A.n && (norm=='1' || norm=='0' || norm=='I'));
    T rcond;int info;
    Array<T> work(max(1,4*A.m),false);
    Array<int> iwork(max(1,A.m),false);
    BC(gecon)(&norm,&A.m,A.data(),&A.n,&anorm,&rcond,work.data(),iwork.data(),&info);
    return rcond;
}

DECLARE(sytrf,char*,int*,T*,int*,int*,T*,int*,int*)

WRAP(void sytrf(CBLAS_UPLO uplo,Subarray<T,2> A,RawArray<int> ipiv,Array<T>& work))
{
    OTHER_ASSERT(A.m==A.n && A.m==ipiv.size());
    const char *u=uplo==CblasLower?"L":"U";
    int n=A.m,lda=max(1,A.stride),lwork=work.max_size(),info;
    BC(sytrf)(const_cast<char*>(u),&n,A.data(),&lda,ipiv.data(),work.data(),&lwork,&info);
}

WRAP(int sytrf_work(CBLAS_UPLO uplo,Subarray<T,2> A))
{
    // Mkl appears to have a bug in the sytrf work query, so we do it ourselves
    static const int nb=max(ilaenv<T>(1,sa"sytrf","L",1024,-1),ilaenv<T>(1,sa"sytrf","U",1024,-1));
    return max(1,nb*A.m);
}

DECLARE(sytrs,char*,int*,int*,T*,int*,int*,T*,int*,int*)

WRAP(void sytrs(CBLAS_UPLO uplo,Subarray<const T,2> A,RawArray<const int> ipiv,Subarray<T,2> B)) // Computes B = B A^{-1} since Lapack is column-major
{
    OTHER_ASSERT(A.m==A.n && A.m==ipiv.size() && B.n==A.m);
    const char *u=uplo==CblasLower?"L":"U";
    int n=A.n,nrhs=B.m,lda=max(1,A.stride),ldb=max(1,B.stride),info;
    BC(sytrs)(const_cast<char*>(u),&n,&nrhs,const_cast<T*>(A.data()),&lda,const_cast<int*>(ipiv.data()),B.data(),&ldb,&info);
}

WRAP(void sytrs(CBLAS_UPLO uplo,Subarray<const T,2> A,RawArray<const int> ipiv,RawArray<T> B))
{
    sytrs(uplo,A,ipiv,B.reshape(1,B.size()));
}

#ifdef USE_MKL

WRAP(void imatcopy(T alpha,Subarray<T,2> A))
{
    BOOST_PP_CAT(mkl_,BC(imatcopy))('r','t',A.m,A.n,alpha,A.data(),A.n,A.m);
}

WRAP(void omatcopy(CBLAS_TRANSPOSE trans,T alpha,Subarray<const T,2> A,Subarray<T,2> B))
{
    int k=trans==CblasTrans?A.n:A.m,lda=max(1,A.stride),ldb=max(1,B.stride);
    OTHER_ASSERT(k==B.m && A.m+A.n-k==B.n);
    BOOST_PP_CAT(mkl_,BC(omatcopy))('r',trans==CblasTrans?'t':'n',A.m,A.n,alpha,A.data(),lda,B.data(),ldb);
}

#endif

#endif
