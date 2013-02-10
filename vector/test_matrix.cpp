//#####################################################################
// Matrix tests
//#####################################################################
//
// These are extremely expensive to compile, so they are disabled by default.
// Change Enabled to 1 if you need to use them to test a change.
//
//#####################################################################

#define ENABLED 0

#include <other/core/python/wrap.h>

#if ! Enabled

static void run_tests()
{}

#else

#include <other/core/vector/DiagonalMatrix.h>
#include <other/core/vector/Matrix.h>
#include <other/core/vector/SymmetricMatrix.h>
#include <other/core/random/Random.h>
#include <other/core/utility/Log.h>
#include <limits>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
namespace other{

typedef real T;
typedef Vector<T,3> TV;

namespace{

static Random<T> rand;

template<class TMatrix>
bool assert_zero(const TMatrix& a,T tolerance=0)
{if(a.maxabs()>tolerance) Log::cout<<"nonzero: "<<a.maxabs()<<" < "<<tolerance<<std::endl;return a.maxabs()<=tolerance;}

template<class TMatrix1,class TMatrix2>
bool assert_equal(const TMatrix1& a,const TMatrix2& b,T tolerance=0)
{tolerance*=max(a.maxabs(),b.maxabs(),(T)1);if(!assert_zero(a-b,tolerance)) Log::cout<<"different:\na ="<<a<<"b ="<<b<<std::endl;return assert_zero(a-b,tolerance);}

template<class TMatrix>
void identify_matrix(const char* name,const TMatrix& a)
{Log::cout<<name<<": "<<a.rows()<<" x "<<a.columns()<<", "<<typeid(a).name()<<"\n"<<a<<std::endl;}

void test(const bool test,const std::string& message)
{if(!test) throw AssertionError(message);}

void dynamic_tests_one_size(int m,int n)
{
    Matrix<T> A(m,n),B(m,n),C,D(m,n);rand.fill_uniform_matrix(A,-1,1);rand.fill_uniform_matrix(B,-1,1);rand.fill_uniform_matrix(D,-1,1);
    T s=rand.get_uniform_number(-1,1),t=rand.get_uniform_number(-1,1);
    T tolerance=std::numeric_limits<T>::epsilon();

    try{
        test(A.rows()==m && A.columns()==n,"Dimension tests.");
        test(B.rows()==m && B.columns()==n,"Dimension tests.");

        test(assert_zero(A-A),"Subtraction with self is zero.");
        test(assert_equal(- -A,A),"Negation is its own inverse.");
        test(assert_equal(A+A+A,A*3,tolerance*2),"Integer scaling as addition.");
        test(assert_equal(-A,A*-1),"Integer scaling as negation.");
        test(assert_equal(A*s,s*A,tolerance),"Scalar multiplication commutes.");
        test(assert_equal(A*(1/s),A*(1/s),tolerance),"Scalar division.");
        test(assert_equal(s*(t*A),(s*t)*A,tolerance*2),"Scalar multiplication associates.");

        test(assert_equal(A+B,B+A),"Addition commutes.");
        test(assert_equal(A-B,A+-B),"Definition of subtraction.");
        test(assert_equal(-(A-B),B-A),"Subtraction anticommutes.");
        test(assert_equal((B+A)+D,B+(A+D),tolerance*2),"Addition associates.");
        test(assert_equal(s*A+s*B,s*(A+B),tolerance*2),"Distributivity of scalar multiplication and addition.");

        test(assert_equal(A.transposed().transposed(),A),"Transpose is its own inverse.");
        test(assert_equal(A.transposed()+B.transposed(),(A+B).transposed()),"Transpose of sum.");
        C.copy(A);C.transpose();test(assert_equal(C,A.transposed()),"Transpose vs Transposed.");

        C.copy(A);test(assert_equal(A,C),"Assignment.");
        C.copy(A);C+=B;test(assert_equal(A+B,C,tolerance),"Plus equals.");
        C.copy(A);C-=B;test(assert_equal(A-B,C,tolerance),"Minus equals.");
        C.copy(A);C*=s;test(assert_equal(A*s,C,tolerance),"Scalar times equals.");
        C.copy(A);C/=s;test(assert_equal(A/s,C,tolerance),"Scalar divide equals.");

        test(A==A,"Equaliy on equal matrices.");
        test(!(A==B),"Equaliy on different matrices.");
        test(!(A!=A),"Inequaliy on equal matrices.");
        test(A!=B,"Inequaliy on different matrices.");

        test(assert_equal(A.identity_matrix(m)*B,B,tolerance),"Left multiply identity.");
        test(assert_equal(A*A.identity_matrix(n),A,tolerance),"Right multiply identity.");
        if(m==n){
            C.set_identity_matrix();test(assert_equal(A.identity_matrix(m),C,tolerance),"Set_Identity_Matrix.");
            test(assert_equal(A+s,A+s*A.identity_matrix(m),tolerance),"Addition with scalar.");
            C.copy(A);C+=s;test(assert_equal(C,A+s*A.identity_matrix(m),tolerance),"Plus equals with scalar.");}}
    catch(...){
        identify_matrix("A",A);
        throw;}
}

void dynamic_tests_two_sizes(int m,int n,int p)
{
    Matrix<T> A(m,n),B(m,n),C(n,p),D(n,p);rand.fill_uniform_matrix(A,-1,1);rand.fill_uniform_matrix(B,-1,1);rand.fill_uniform_matrix(C,-1,1);
    rand.fill_uniform_matrix(D,-1,1);T s=rand.get_uniform_number(-1,1),tolerance=std::numeric_limits<T>::epsilon();

    try{
        test(assert_equal((A+B)*C,A*C+B*C,tolerance*A.columns()*2),"Right distributivity.");
        test(assert_equal(A*(C+D),A*C+A*D,tolerance*A.columns()*2),"Left distributivity.");
        test(assert_equal((s*A)*C,s*(A*C),tolerance*A.columns()*2),"Scalar and matrix multiplication associate.");

        test(assert_equal(C.transposed()*A.transposed(),(A*C).transposed()),"Transpose of product.");
        test(assert_equal(A*C,A.transposed().transpose_times(C)),"Transpose_Times.");
        test(assert_equal(A*C,A.times_transpose(C.transposed())),"Times_Transpose.");}
    catch(...){
        identify_matrix("A",A);
        identify_matrix("C",C);
        throw;}
}

void dynamic_tests_three_sizes(int m,int n,int p,int q)
{
    Matrix<T> A(m,n),B(n,p),C(p,q);rand.fill_uniform_matrix(A,-1,1);rand.fill_uniform_matrix(B,-1,1);rand.fill_uniform_matrix(C,-1,1);
    T tolerance=std::numeric_limits<T>::epsilon();

    try{
        test(assert_equal((A*B)*C,A*(B*C),tolerance*B.rows()*B.columns()*2),"Multiplication associates.");}
    catch(...){
        identify_matrix("A",A);
        identify_matrix("B",B);
        identify_matrix("C",C);
        throw;}
}

void dynamic_tests(int size)
{
    for(int c=0;c<20;c++)
        for(int i=1;i<=size;i++) for(int j=1;j<=size;j++){
            dynamic_tests_one_size(i,j);
            for(int k=1;k<=size;k++){
                dynamic_tests_two_sizes(i,j,k);
                for(int m=1;m<=size;m++) dynamic_tests_three_sizes(i,j,k,m);}}
}

template<class T,class TMatrix>
void conversion_test(TMatrix& A,Matrix<T>& B)
{
    test(assert_equal(Matrix<T>(TMatrix(B)),B),"Conversion both ways is exact.");
}

template<class T,int d>
void conversion_test(DiagonalMatrix<T,d>& A,Matrix<T>& B)
{}

template<class T,int d>
void conversion_test(SymmetricMatrix<T,d>& A,Matrix<T>& B)
{
    Matrix<T> C(A);
    for(int i=0;i<d;i++) for(int j=0;j<d;j++){
        test(A(i,j)==C(i,j),"Constructor conversion is exact");
        test(A(i,j)==B(i,j),"Assignment conversion is exact");}
}

template<class TMatrix1>
void one_size(TMatrix1 A)
{
    TMatrix1 B,C;
    T tolerance=std::numeric_limits<T>::epsilon();
    test(B.columns()==A.columns() && B.rows()==A.rows(),"Dimension tests (B).");
    test(C.columns()==A.columns() && C.rows()==A.rows(),"Dimension tests (C).");

    for(int i=0;i<20;i++){
        rand.fill_uniform_matrix(A,-1,1);rand.fill_uniform_matrix(B,-1,1);
        Matrix<T> D,E,F;
        D.copy(A);E.copy(B);
        T s=rand.get_uniform_number(-1,1);

        try{
            conversion_test(A,D);

            test(assert_equal(Matrix<T>(-A),-D,tolerance),"Negation matches.");
            test(assert_equal(Matrix<T>(s*A),s*D,tolerance),"Left scaling matches.");
            test(assert_equal(Matrix<T>(A*s),D*s,tolerance),"Right scaling matches.");
            test(assert_equal(Matrix<T>(A/s),D/s,tolerance),"Scalar division matches.");
            test(assert_equal(Matrix<T>(A+B),D+E,tolerance),"Addition matches.");
            test(assert_equal(Matrix<T>(A-B),D-E,tolerance),"Subtraction matches.");
            test(assert_equal(Matrix<T>(A.transposed()),D.transposed(),tolerance),"Transpsoed matches.");

            C.copy(A);test(assert_equal(Matrix<T>(C),D,tolerance),"Assignment matches.");
            C.copy(A);C*=s;test(assert_equal(Matrix<T>(C),D*s,tolerance),"Scalar times equal matches.");
            C.copy(A);C/=s;test(assert_equal(Matrix<T>(C),D/s,tolerance),"Scalar divide equal matches.");
            C.copy(A);C+=B;test(assert_equal(Matrix<T>(C),D+E,tolerance),"Plus equal matches.");
            C.copy(A);C-=B;test(assert_equal(Matrix<T>(C),D-E,tolerance),"Minus equal matches.");

            test(assert_equal(Matrix<T>(A),D),"Inputs not changed.");
            test(assert_equal(Matrix<T>(B),E),"Inputs not changed.");}
        catch(...){
            identify_matrix("A",A);
            identify_matrix("D",D);
            throw;}}
}

template<class TMatrix1,class TMatrix2>
void two_sizes(TMatrix1 A,TMatrix2 B)
{
    T tolerance=std::numeric_limits<T>::epsilon()*2;
    test(A.columns()==B.rows(),"Dimension tests (A).");

    for(int i=0;i<20;i++){
        Matrix<T> D(A.rows(),A.columns()),E(B.rows(),B.columns());rand.fill_uniform_matrix(A,-1,1);rand.fill_uniform_matrix(B,-1,1);D=Matrix<T>(A);E=Matrix<T>(B);

        try{
            test(assert_equal(Matrix<T>(A*B),D*E,tolerance),"Multiplication matches.");
            test(assert_equal(Matrix<T>(A.transposed().transpose_times(B)),D*E,tolerance),"Transpose_Times matches.");
            test(assert_equal(Matrix<T>(A.times_transpose(B.transposed())),D*E,tolerance),"Times_Transpose matches.");

            test(assert_equal(Matrix<T>(A),D),"Inputs not changed.");
            test(assert_equal(Matrix<T>(B),E),"Inputs not changed.");}
        catch(...){
            identify_matrix("A",A);
            identify_matrix("D",D);
            identify_matrix("B",B);
            identify_matrix("E",E);
            throw;}}
}

template<class M1,class M2,class M3,int a,int b,int c> struct Generate
{
    Generate()
    {
        Generate<M2,M3,Matrix<T,a,b>,b,c,a>();
        square(mpl::bool_<a==b && (a==2 || a==3)>());
    }

    void square(mpl::true_)
    {
        Generate<M2,M3,SymmetricMatrix<T,a>,b,c,a>();
        Generate<M2,M3,DiagonalMatrix<T,a>,b,c,a>();
    }
    void square(mpl::false_){}
};

template<class M2,class M3,int a,int b,int c> struct Generate<bool,M2,M3,a,b,c>
{
    Generate()
    {
        if(a==b) one_size(M2());
        two_sizes(M2(),M3());
    }
};

void run_tests()
{
    dynamic_tests(6);

    #define GEN(size) \
        BOOST_PP_REPEAT(BOOST_PP_MUL(size,BOOST_PP_MUL(size,size)),_LOOP,size)
        #define _LOOP(z, x, size) Generate<int,int,bool,x/size/size+1,x/size%size+1,x%size+1>();
    Gen(4)
}

}
}
using namespace other;

#endif

void wrap_test()
{
    OTHER_FUNCTION(run_tests)
}
