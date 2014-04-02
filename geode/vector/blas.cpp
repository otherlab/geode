//#####################################################################
// Header blas
//#####################################################################
#include <geode/vector/blas.h>
#ifdef GEODE_BLAS
#ifdef GEODE_MKL
#define ILAENV ::ilaenv
#else
extern "C" {
#ifndef __APPLE__
extern int ilaenv_(int*,char*,char*,int*,int*,int*,int*);
#endif
}
#define ILAENV ::ilaenv_
#endif
namespace geode {

template<class T> int ilaenv(int ispec,const char* name,const char* opts,int m,int n)
{
    char xname[20];
    xname[0] = is_same<T,float>::value?'s':'d';
    strcpy(xname+1,name);
#ifndef __APPLE__
    int no = -1;
    return ILAENV(&ispec,xname,(char*)opts,&m,&n,&no,&no);
#else
    GEODE_NOT_IMPLEMENTED();
#endif
}

template int ilaenv<float>(int,const char*,const char*,int,int);
template int ilaenv<double>(int,const char*,const char*,int,int);
}
#endif
