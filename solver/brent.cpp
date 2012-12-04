// Brent's method for one-dimensional optimization.  The implementation
// is taken from scipy, which requires the following notice:
// ******NOTICE***************
// optimize.py module by Travis E. Oliphant
//
// You may copy and use this module as you see fit with no
// guarantee implied provided you keep this notice in all copies.
// *****END NOTICE************

#include <other/core/solver/brent.h>
#include <other/core/python/module.h>
#include <other/core/python/function.h>
#include <other/core/math/copysign.h>


// Windows silliness
#undef small
#undef far
#undef near

namespace other {

typedef real T;

// Bracket a minimum given a starting interval, returning (a,b,c),(f(a),f(b),f(c)) with f(b) < f(a),f(c)
static Tuple<Vector<T,3>,Vector<T,3>> bracket(const function<T(T)>& f, T xa, T xb) {
  const int maxiter = 1000;
  const T grow_limit = 110;
  const T gold = (1+sqrt(5))/2;
  const T small = 1e-21;
  T fa = f(xa);
  T fb = f(xb);
  if (fa < fb) { // Switch so fa > fb
    swap(xa,xb);
    swap(fa,fb);
  }
  T xc = xb+gold*(xb-xa);
  T fc = f(xc);
  int iter = 0;
  while (fc < fb) {
    const T tmp1 = (xb-xa)*(fb-fc);
    const T tmp2 = (xb-xc)*(fb-fa);
    const T val = tmp2-tmp1;
    const T denom = 2*(abs(val)<small?small:val);
    T w = xb-((xb-xc)*tmp2-(xb-xa)*tmp1)/denom;
    const T wlim = xb+grow_limit*(xc-xb);
    if (iter > maxiter)
      throw RuntimeError("bracket: Too many iterations");
    iter++;
    T fw = 0; // Should be left uninitialized, but gcc 4.7 complains
    if ((w-xc)*(xb-w) > 0) {
      T fw = f(w);
      if (fw < fc) {
        xa = xb; xb = w; fa = fb; fb = fw;
        break;
      } else if (fw > fb) {
        xc = w; fc = fw;
        break;
      }
      w = xc+gold*(xc-xb);
      fw = f(w);
    } else if ((w-wlim)*(wlim-xc) >= 0) {
      w = wlim;
      fw = f(w);
    } else if ((w-wlim)*(xc-w)>0) {
      fw = f(w);
      if (fw < fc) {
        xb = xc; xc = w; w = xc + gold*(xc-xb);
        fb = fc; fc = fw; fw = f(w);
      }
    } else {
      w = xc + gold*(xc-xb);
      fw = f(w);
    }
    xa = xb; xb = xc; xc = w;
    fa = fb; fb = fc; fc = fw;
  }
  if (xa > xc) {
    swap(xa,xc);
    swap(fa,fc);
  }
  return tuple(vec(xa,xb,xc),vec(fa,fb,fc));
}

static Tuple<T,T,int> brent_helper(const function<T(T)>& f, T a, T x, T b, T fx, const T xtol, const int maxiter) {
  small_sort(a,b);
  OTHER_ASSERT(a<x && x<b);

  // Core algorithm
  T w = x, v = x;
  T fw = fx, fv = fx;
  const T cg = (3-sqrt(5))/2;
  T deltax = 0;
  int iter = 0;
  while (iter < maxiter) {
    const T xmid = .5*(a+b);
    if (abs(x-xmid) <= 2*xtol-.5*(b - a)) // Check for convergence
      break;
    T rat;
    if (abs(deltax) <= xtol) { // Do a golden section step
      deltax = x>=xmid?a-x:b-x;
      rat = cg*deltax;
    } else { // Do a parabolic step
      const T tmp1 = (x-w)*(fx-fv);
      T tmp2 = (x-v)*(fx-fw);
      T p = (x-v)*tmp2-(x-w)*tmp1;
      tmp2 = 2*(tmp2-tmp1);
      if (tmp2 > 0)
        p = -p;
      tmp2 = abs(tmp2);
      const T dx_temp = deltax;
      deltax = rat;
      // Check parabolic fit
      if (p>tmp2*(a-x) && p<tmp2*(b-x) && abs(p)<abs(.5*tmp2*dx_temp)) { // If parabolic step is useful, take it
        rat = p/tmp2;
        const T u = x + rat;
        if ((u-a)<xtol || (b-u)<xtol)
          rat = copysign(xtol,xmid-x);
      } else { // if it's not do a golden section step
        deltax = x>=xmid?a-x:b-x;
        rat = cg*deltax;
      }
    }

    const T u = x + (abs(rat)>xtol?rat:copysign(xtol,rat)); // Update by at least xtol
    const T fu = f(u); // Calculate new output value
    if (fu > fx) { // If it's bigger than current
      if (u < x)
        a = u;
      else
        b = u;
      if (fu<=fw || w==x) {
        v = w; w = u; fv = fw; fw = fu;
      } else if (fu<=fv || v==x || v==w) {
        v = u; fv = fu;
      }
    } else {
      if (u >= x)
        a = x;
      else
        b = x;
      v = w; w = x; x = u;
      fv = fw; fw = fx; fx = fu;
    }

    iter++;
  }

  return tuple(x,fx,iter);
}

Tuple<T,T,int> brent(const function<T(T)>& f, const Vector<T,2> brack, const T xtol, const int maxiter) {
  // Bracket minimum
  T a,b,c;
  const auto br = bracket(f,brack.x,brack.y);
  br.x.get(a,b,c);
  const T fb = br.y.y;
  // Optimize
  return brent_helper(f,a,b,c,fb,xtol,maxiter);
}

Tuple<T,T,int> brent(const function<T(T)>& f, const Vector<T,3> bracket, const T xtol, const int maxiter) {
  T a,b,c;
  bracket.get(a,b,c);
  return brent_helper(f,a,b,c,f(b),xtol,maxiter);
}

Tuple<T,T,int> brent_py(const function<T(T)>& f, RawArray<const T> bracket, const T xtol, const int maxiter) {
  if (bracket.size()==2)
    return brent(f,vec(bracket[0],bracket[1]),xtol,maxiter);
  else if (bracket.size()==3)
    return brent(f,vec(bracket[0],bracket[1],bracket[2]),xtol,maxiter);
  throw TypeError(format("brent: expected (a,b) or (a,b,c) for bracket, got size %d",bracket.size()));
}

}
using namespace other;

void wrap_brent() {
  OTHER_FUNCTION(bracket)
  OTHER_FUNCTION_2(brent,brent_py)
}
