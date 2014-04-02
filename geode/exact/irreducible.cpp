// Irreducibility testing for integer polynomials
#pragma

#include <geode/exact/irreducible.h>
#include <geode/exact/math.h>
#include <geode/exact/polynomial.h>
#include <geode/exact/perturb.h>
#include <geode/array/Array3d.h>
#include <geode/array/ConstantMap.h>
#include <geode/structure/Hashtable.h>
namespace geode {

typedef uint64_t UI;
using std::cout;
using std::endl;

// Checking a bunch of small primes is enough for our purposes
static const UI primes[] = {13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499};

namespace {
// Simple finite field arithmetic
struct Fp {
  UI x, p; // x mod p

  Fp(const UI x, const UI p)
    : x(x), p(p) {
    assert(x<p);
  }

  explicit operator bool() const { return x != 0; }
  bool operator==(const Fp y) const { assert(y.p==p); return x==y.x; }

  Fp operator-() const { return Fp((p-x)%p,p); }
  Fp operator+(const Fp y) const { assert(y.p==p); return Fp((x+y.x)%p,p); }
  Fp operator-(const Fp y) const { assert(y.p==p); return Fp((x-y.x+p)%p,p); }
  Fp operator*(const Fp y) const { assert(y.p==p); return Fp((x*y.x)%p,p); }
  Fp operator/(const Fp y) const { assert(y.p==p); return *this*inverse(y); }
  Fp& operator+=(const Fp y) { return *this = *this+y; }
  Fp& operator-=(const Fp y) { return *this = *this-y; }
  Fp& operator*=(const Fp y) { return *this = *this*y; }
  Fp& operator/=(const Fp y) { return *this = *this/y; }

  friend Fp pow(Fp x, UI a) {
    if (x.x<2 && a)
      return x;
    Fp r(1,x.p);
    while (a) {
      if (a & 1)
        r *= x;
      x *= x;
      a >>= 1;
    }
    return r;
  }

  friend Fp inverse(const Fp x) {
    GEODE_ASSERT(x);
    // x^(p-1) = 1, so 1/x = x^(p-2)
    return pow(x,x.p-2);
  }

  friend ostream& operator<<(ostream& output, const Fp x) {
    return output << x.x;
  }
};

// Simplistic polynomial arithmetic.
// This code is very slow, since it is used only for testing purposes.
struct Poly {
  Array<const Fp> c;
  const UI p;

  explicit Poly(const Fp s)
    : c(s ? asarray(vec(s)).copy()
          : Array<const Fp>())
    , p(s.p) {}

  explicit Poly(const Array<Fp>& c, const UI p)
    : c(c)
    , p(p) {
    if (c.size()) {
      GEODE_ASSERT(c.back());
      assert(c.back().p==p);
    }
  }

  Poly& operator=(const Poly& f) {
    assert(p==f.p);
    c = f.c;
    return *this;
  }

  int deg() const { return c.size()-1; }
  Fp leading() const { GEODE_ASSERT(c.size()); return c.back(); }
  explicit operator bool() const { return c.size() != 0; }
  bool monic() const { return c.size() && c.back().x==1; }

  Poly operator+(const Poly& g) const {
    auto r = constant_map(1+max(deg(),g.deg()),Fp(0,p)).copy();
    r.slice(0,c.size()) += c;
    r.slice(0,g.c.size()) += g.c;
    while (r.size() && !r.back())
      r.pop();
    return Poly(r,p);
  }

  Poly operator-(const Poly& g) const {
    auto r = constant_map(1+max(deg(),g.deg()),Fp(0,p)).copy();
    r.slice(0,c.size()) += c;
    r.slice(0,g.c.size()) -= g.c;
    while (r.size() && !r.back())
      r.pop();
    return Poly(r,p);
  }

  // Multiply by a power of x
  Poly shift(const int n=1) const {
    GEODE_ASSERT(n >= 0);
    return !*this || !n ? *this : Poly(concatenate(constant_map(n,Fp(0,p)),c),p);
  }

  Poly operator*(const Poly& g) const {
    if (!(*this && g))
      return Poly(Fp(0,p));
    auto r = constant_map(max(0,1+deg()+g.deg()),Fp(0,p)).copy();
    for (const int i : range(1+deg()))
      if (c[i])
        for (const int j : range(1+g.deg()))
          r[i+j] += c[i]*g.c[j];
    return Poly(r,p);
  }

  friend Poly operator*(const Fp a, const Poly& f) {
    return a ? Poly((a*f.c).copy(),a.p) : Poly(a);
  }

  friend Vector<Poly,2> divmod(const Poly& f, const Poly& g) {
    GEODE_ASSERT(g);
    const auto s = inverse(g.leading());
    auto q = Poly(Fp(0,g.p));
    auto r = f;
    while (r.deg() >= g.deg()) {
      const auto h = Poly(s*r.leading()).shift(r.deg()-g.deg());
      q = q+h;
      r = r-h*g;
    }
    return vec(q,r);
  }

  Poly operator/(const Poly& g) const { return divmod(*this,g).x; }
  Poly operator%(const Poly& g) const { return divmod(*this,g).y; }

  friend Poly pow_mod(Poly f, UI a, const Poly& m) {
    Poly r(Fp(1,f.p));
    while (a) {
      if (a & 1)
        r = (r*f)%m;
      f = (f*f)%m;
      a >>= 1;
    }
    return r;
  }

  GEODE_UNUSED friend ostream& operator<<(ostream& output, const Poly& f) {
    bool first = true;
    for (int n=f.deg();n>=0;n--)
      if (f.c[n]) {
        if (!first)
          output << '+';
        first = false;
        if (!n || f.c[n].x!=1)
        output << f.c[n];
        if (n > 1)
          output << "x^" << n;
        else if (n)
          output << 'x';
      }
    if (first)
      output << '0';
    return output;
  }
};
}

template<class I> static I gcd(I a, I b) {
  while (b) {
    const auto t = b;
    b = a % b; 
    a = t;
  }
  return a;
}

// For polynomials modulo a prime, we finally have a simple exact algorithm
static bool irreducible(Poly f) {
  // Make monic
  f = inverse(f.leading())*f;

  // See Algorithm 4.69, Testing a polynomial for irreducibility,
  // in Handbook of Applied Cryptography, http://cacr.uwaterloo.ca/hac/about/chap4.pdf.
  const auto x = Poly(Fp(1,f.p)).shift();
  auto u = x;
  for (int i=0;i<f.deg()/2;i++) {
    u = pow_mod(u,f.p,f);
    if (gcd(f,u-x).deg())
      return false;
  }
  return true;
}

static bool lowest_terms(const Poly f, const Poly g) {
  return gcd(f,g).deg()==0;
}

// Reduce a polynomial modulo prime p
static Poly poly_mod(Subarray<const mp_limb_t,2> poly, const UI p) {
  Array<Fp> mod(poly.m,uninit);
  const auto neg = pow(Fp(2,p),8*sizeof(mp_limb_t)*poly.n);
  for (const int i : range(poly.m)) {
    mod[i] = Fp(mpn_mod_1(poly[i].data(),poly.n,p),p);
    if (mpz_negative(poly[i]))
      mod[i] -= neg;
  }
  while (mod.size() && !mod.back())
    mod.pop();
  return Poly(mod,p);
}

// Approximately test for reducibility by testing modulo several primes
static bool inexact_irreducible(RawArray<const mp_limb_t,2> poly) {
  const int degree = poly.m-1;
  for (const auto p : primes) {
    const auto mod = poly_mod(poly,p);
    if (mod.deg()==degree && irreducible(mod))
      return true;
  }
  return false;
}

// Approximately test for lowest terms by testing modulo several primes
static bool inexact_lowest_terms(RawArray<const mp_limb_t,3> ratio) {
  const int degree = ratio.m-1;
  const int r = ratio.n-1;
  GEODE_ASSERT(degree>=1 && r>=1);
  for (const auto p : primes) {
    const auto den = poly_mod(ratio.sub<1>(r),p);
    GEODE_ASSERT(den.deg()<degree);
    if (den.deg()==degree-1)
      for (const int i : range(r)) {
        const auto num = poly_mod(ratio.sub<1>(i),p);
        if (num.deg()==degree && lowest_terms(num,den))
          return true;
      }
  }
  return false;
}

// Don't check the same function multiple times
static Hashtable<void*> safe;

namespace {
template<int m> struct Helper {
  const int precision;
  const Array<const uint8_t,2> lambda;

  typedef Vector<Exact<1>,m> EV;
  const Array<EV> X_;

  Helper(const int degree, const int inputs)
    : precision(degree*Exact<1>::ratio)
    , lambda(monomials(degree,1))
    , X_(inputs,uninit) {
    GEODE_ASSERT(lambda.m==degree+1);
  }

  RawArray<const EV> X(const int key, const int j) {
    for (const int i : range(X_.size()))
      X_[i] = EV(    perturbation<m>(2*key  ,i)
                 + j*perturbation<m>(2*key+1,i));
    return X_;
  }
};}

static bool warn = true;

template<int m> void
inexact_assert_irreducible(void(*const polynomial)(RawArray<mp_limb_t>,RawArray<const Vector<Exact<1>,m>>),
                           const int degree, const int inputs, const char* name) {
  if (warn)
    GEODE_WARNING("Diagnostic irreducibility checking on");

  // Is it low degree, or have we already seen it?
  if (degree<2 || safe.contains((void*)polynomial))
    return;
  cout << format("irreducibility check: %s, degree %d, R^(%d*%d) -> R",name,degree,inputs,m) << endl;

  Helper<m> H(degree,inputs);
  const Array<mp_limb_t,2> values(degree+1,H.precision,uninit);
  for (const int k : range(100)) {
    // Evaluate polynomial at j = 0,...,degree
    for (const int j : range(degree+1))
      polynomial(values[j],H.X(k,j));

    // Find an interpolating polynomial
    in_place_interpolating_polynomial(degree,H.lambda,values);

    // If it's irreducible, we're done
    if (inexact_irreducible(values)) {
      safe.set((void*)polynomial);
      return;
    }
  }

  // If we reach this point, the multivariate polynomial looks reducible as a univariate polynomial
  // modulo lots of primes.  This doesn't prove that it is reducible, but it's useful evidence.
  throw ArithmeticError(format("Polynomial %s : R^(%d*%d) -> R of degree %d is probably reducible",
                               name,inputs,m,degree));
}

template<int m> void
inexact_assert_lowest_terms(void(*const ratio)(RawArray<mp_limb_t,2>,RawArray<const Vector<Exact<1>,m>>),
                            const int degree, const int inputs, const int outputs, const char* name) {
  if (warn)
    GEODE_WARNING("Diagnostic lowest terms checking on");

  // Is it low degree, or have we already seen it?
  if (degree<2 || safe.contains((void*)ratio))
    return;
  cout << format("lowest terms check: %s, degree %d, R^(%d*%d) -> R^%d/R",name,degree,inputs,m,outputs) << endl;

  Helper<m> H(degree,inputs);
  const Array<mp_limb_t,3> values(degree+1,outputs+1,H.precision,uninit);
  for (const int k : range(100)) {
    // Evaluate ratio at j = 0,...,degree
    for (const int j : range(degree+1))
      ratio(values[j],H.X(k,j));

    // Find an interpolating polynomial
    for (const auto r : range(outputs+1))
      in_place_interpolating_polynomial(degree,H.lambda,values.sub<1>(r));

    // Are we in lowest terms?
    if (inexact_lowest_terms(values)) {
      safe.set((void*)ratio);
      return;
    }
  }

  // If we reach this point, the numerator and denominator have common factors in a lot of different directions.
  throw ArithmeticError(format("Ratio %s : R^(%d*%d) -> R^%d/R of degree %d/%d is probably not in lowest terms",
                               name,inputs,m,outputs,degree,degree-1));
}

namespace {
struct GoodPoly { template<class TV> static PredicateType<2,TV> eval(const TV a, const TV b, const TV c) {
  return edet(b-a,c-a);
}};
struct BadPoly0 { template<class TV> static PredicateType<2,TV> eval(const TV a) {
  return a.x*a.x;
}};
struct BadPoly1 { template<class TV> static PredicateType<4,TV> eval(const TV a, const TV b, const TV c) {
  return edet(a,b)*edet(b-a,c-a);
}};
struct GoodRatio { template<class TV> static ConstructType<2,3,TV> eval(const TV a0, const TV a1,
                                                                        const TV b0, const TV b1) {
  const auto da = a1-a0, db = b1-b0;
  const auto den = edet(da,db);
  return tuple(emul(den,a0)+emul(edet(b0-a0,db),da),den);
}};
struct BadRatio0 { template<class TV> static ConstructType<1,2,TV> eval(const TV a) {
  return tuple(vec(a.x*a.x),a.x);
}};
struct BadRatio1 { template<class TV> static ConstructType<2,5,TV> eval(const TV a0, const TV a1,
                                                                        const TV b0, const TV b1) {
  const auto s = edet(a0,a1);
  const auto r = GoodRatio::eval(a0,a1,b0,b1);
  return tuple(emul(s,r.x),s*r.y);
}};
}
void irreducible_test() {
  warn = false;

  // Test irreducibility
  inexact_assert_irreducible(wrap_predicate<GoodPoly,2>(IRange<3>()),2,3,"GoodPoly");
  assert_raises<ArithmeticError>([](){
    inexact_assert_irreducible(wrap_predicate<BadPoly0,1>(IRange<1>()),2,1,"BadPoly0");});
  assert_raises<ArithmeticError>([](){
    inexact_assert_irreducible(wrap_predicate<BadPoly1,2>(IRange<3>()),4,3,"BadPoly1");});

  // Test lowest terms
  inexact_assert_lowest_terms(wrap_predicate<GoodRatio,2>(IRange<4>()),3,4,2,"GoodRatio");
  assert_raises<ArithmeticError>([](){
    inexact_assert_lowest_terms(wrap_predicate<BadRatio0,1>(IRange<1>()),2,1,1,"BadRatio0");},
    "BadRatio0 should not be in lowest terms");
  assert_raises<ArithmeticError>([](){
    inexact_assert_lowest_terms(wrap_predicate<BadRatio1,2>(IRange<4>()),5,4,2,"BadRatio1");},
    "BadRatio1 should not be in lowest terms");

  warn = true;
}

#define INSTANTIATE(m) \
  template void inexact_assert_irreducible(void(*const)(RawArray<mp_limb_t>,RawArray<const Vector<Exact<1>,m>>), \
                                           int,int,const char*); \
  template void inexact_assert_lowest_terms(void(*const)(RawArray<mp_limb_t,2>,RawArray<const Vector<Exact<1>,m>>), \
                                            int,int,int,const char*);
INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)

}
