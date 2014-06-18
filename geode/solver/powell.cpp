// Powell's method for multidimensional optimization.  The implementation
// is taken from scipy, which requires the following notice:
// ******NOTICE***************
// optimize.py module by Travis E. Oliphant
//
// You may copy and use this module as you see fit with no
// guarantee implied provided you keep this notice in all copies.
// *****END NOTICE************

#include <geode/solver/powell.h>
#include <geode/solver/brent.h>
#include <geode/array/Array2d.h>
#include <geode/python/function.h>
#include <geode/python/wrap.h>
#include <geode/utility/curry.h>
namespace geode {

typedef real T;

static T along_ray(const function<T(RawArray<const T>)>* f, RawArray<const T> x0, RawArray<const T> dx, RawArray<T> tmp, const T t) {
  tmp = x0+t*dx;
  return (*f)(tmp);
}

// See _linesearch_powell in https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
// Unlike the Python version, p and xi are modified in place.
static T linesearch_powell(const function<T(RawArray<const T>)>& f, RawArray<T> p, RawArray<T> xi, const T xtol, RawArray<T> tmp) {
  const int n = p.size();
  const T atol = min(.1,5*xtol/magnitude(xi));
  const auto alpha_fret_iter = brent(curry(along_ray,&f,p,xi,tmp),Vector<T,2>(0,1),atol,100);
  const T alpha = alpha_fret_iter.x,
          fret = alpha_fret_iter.y;
  for (int i=0;i<n;i++) {
    xi[i] *= alpha;
    p[i] += xi[i];
  }
  return fret;
}

// See fmin_powell in https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
Tuple<T,int> powell(const function<T(RawArray<const T>)>& f, RawArray<T> x, T scale, T xtol, T ftol, int maxiter) {
  GEODE_ASSERT(scale>0 && xtol>=0 && ftol>=0);
  const int n = x.size();
  Array<T> direc1(n,uninit), tmp(n,uninit);

  // Initialize direction matrix to scale
  Array<T,2> direc(n,n);
  for (int i=0;i<n;i++)
    direc(i,i) = scale;

  T fval = f(x);
  const auto x1 = x.copy();
  int iter = 0;
  for (;;) {
    const T fx = fval;
    int bigind = 0;
    T delta = 0;
    for (int i=0;i<n;i++) {
      const T fx2 = fval;
      fval = linesearch_powell(f,x,direc[i],xtol,tmp);
      if (fx2-fval > delta) {
        delta = fx2-fval;
        bigind = i;
      }
    }
    iter++;
    if (fx-fval<=ftol || iter>=maxiter)
      break;

    // Construct the extrapolated point
    for (int i=0;i<n;i++) {
      direc1[i] = x[i]-x1[i];
      tmp[i] = x[i]+direc1[i];
      x1[i] = x[i];
    }
    const T fx2 = f(tmp);

    if (fx > fx2) {
      T t = 2*(fx+fx2-2*fval);
      T temp = fx-fval-delta;
      t *= temp*temp;
      temp = fx-fx2;
      t -= delta*temp*temp;
      if (t < 0) {
        fval = linesearch_powell(f,x,direc1,xtol,tmp);
        direc[bigind] = direc[n-1];
        direc[n-1] = direc1;
      }
    }
  }

  // All done.
  return tuple(fval,iter);
}

static T f_py(const function<T(Array<const T>)>& f, RawArray<const T> p) {
  TemporaryOwner owner;
  return f(owner.share(p));
}

static Tuple<T,int> powell_py(const function<T(Array<const T>)>& f, RawArray<T> x, T scale, T xtol, T ftol, int maxiter) {
  return powell(curry(f_py,f),x,scale,xtol,ftol,maxiter);
}

}
using namespace geode;

void wrap_powell() {
  GEODE_FUNCTION_2(powell,powell_py)
}
