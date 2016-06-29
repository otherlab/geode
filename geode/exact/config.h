// Multiprecision integer arithmetic for exact geometric predicates
#pragma once

/* Doubles have 52 bits of mantissa, so they can exactly represent any integer in [-2**53,2**53]
 * (52+1 due to the implicit 1 before the mantissa).  Single precision floats can handle
 * [-2**24,2**24].  To give this some context, 25 bit precision in a 10 meter space is an accuracy
 * of 0.3 um.  Ideally, this would be sufficient for our purposes for now.
 *
 * Unfortunately, some algorithms require additional sentinel bits for special purposes, so 2**24
 * is a bit too tight.  Symbolic perturbation of high degree predicates absorbs a few more bits.
 * While its often possible to carefully arrange for single precision to work, double precision is
 * easy and more precise anyways.  Thus, we quantize into the integer range [-2**53,2**53]/1.01.
 * The 1.01 factor gives algorithms a bit of space for sentinel purposes.
 */

#include <geode/utility/config.h>
#include <geode/structure/Tuple.h>
#include <geode/vector/Vector.h>
#include <stdint.h>
namespace geode {
namespace exact {

// Integer values in [-bound,bound] are safely exactly representable.  To allow a bit of
// slack for algorithms to use, all quantized points will live in roughly [-bound,bound]/1.01.
const int log_bound = 53;
const int64_t bound = (int64_t(1)<<log_bound)-1;

}

// Base integer type for exact arithmetic
#define GEODE_EXACT_INT 64
typedef int64_t ExactInt;
typedef double Quantized;

enum class Perturbation { Explicit, Implicit };

namespace exact {

#ifdef NDEBUG
#else
// Check that values are correctly quantized
// ImplicitlyPerturbed doesn't allow negative zero so that values can be hashed byte by byte
static inline bool is_quantized(Quantized x) { return static_cast<ExactInt>(x) == x && ((x != 0.) || !std::signbit(x)); }
static inline bool is_quantized(Vector<Quantized, 1> v) { return is_quantized(v.x); }
static inline bool is_quantized(Vector<Quantized, 2> v) { return is_quantized(v.x) && is_quantized(v.y); }
static inline bool is_quantized(Vector<Quantized, 3> v) { return is_quantized(v.x) && is_quantized(v.y) && is_quantized(v.z); }
#endif

// Like CGAL, GMP assumes that the C++11 standard library exists whenever C++11 does.  This is false for clang.
#define __GMPXX_USE_CXX11 0

template<int d> struct Perturbed {
  static constexpr Perturbation ps = Perturbation::Explicit;
  typedef Vector<Quantized, d> ValueType;
  static const int m = ValueType::m;
  int seed_;
  ValueType value_;
  int seed() const { return seed_; }
  ValueType value() const { return value_; }

  Perturbed() = default;
  template<class... Args> explicit Perturbed(const int seed_, const Args... value_args)
   : seed_(seed_)
   , value_(value_args...) // Pass args along to choose a Vector constructor
  {}
};

template<int d> struct ImplicitlyPerturbed {
  static constexpr Perturbation ps = Perturbation::Implicit;
  typedef Vector<Quantized, d> ValueType;
  static const int m = ValueType::m;
  ValueType value_;
  ValueType seed() const { assert(is_quantized(value_)); return value_; }
  ValueType value() const { return value_; }

  template<class... Args> explicit ImplicitlyPerturbed(const Args... value_args) : value_(value_args...) {}
};

struct ImplicitlyPerturbedCenter {
  static constexpr Perturbation ps = Perturbation::Implicit;
  typedef Vector<Quantized, 2> ValueType;
  static constexpr auto m = ValueType::m;
  Vector<Quantized, 3> data;
  Vector<Quantized, 3> seed() const { assert(is_quantized(data)); return data; }
  Vector<Quantized, 2> value() const { return data.xy(); }

  template<class... Args> explicit ImplicitlyPerturbedCenter(const Args... value_args) : data(value_args...) {}
};

typedef Vector<Quantized,2> Vec2;
typedef Vector<Quantized,3> Vec3;
typedef Perturbed<2> Perturbed2;
typedef Perturbed<3> Perturbed3;
}

template<int d> std::ostream& operator<<(std::ostream& os, const exact::Perturbed<d> p) { return os << tuple(p.seed_, p.value_); }
template<int d> std::ostream& operator<<(std::ostream& os, const exact::ImplicitlyPerturbed<d> p) { return os << p.value_; }
inline std::ostream& operator<<(std::ostream& os, const exact::ImplicitlyPerturbedCenter p) { return os << p.data; }
}
