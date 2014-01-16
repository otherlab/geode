//#####################################################################
// Header blas
//#####################################################################
#include "blas.h"
#ifdef GEODE_BLAS
#include <geode/python/from_python.h>
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

#ifdef GEODE_PYTHON
CBLAS_TRANSPOSE FromPython<CBLAS_TRANSPOSE>::convert(PyObject* object) {
  const char* s = from_python<const char*>(object);
  switch (s[0]?s[1]?0:s[0]:0){
    case 'n': case 'N': return CblasNoTrans;
    case 't': case 'T': return CblasTrans;
    default: throw ValueError("expected n or t (or N or T)");}
}
#endif

template int ilaenv<float>(int,const char*,const char*,int,int);
template int ilaenv<double>(int,const char*,const char*,int,int);
}
#endif
