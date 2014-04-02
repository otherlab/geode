// Class Sobol: Low discrepancy quasirandom numbers
#pragma once

#include <geode/random/forward.h>
#include <geode/array/Array.h>
#include <geode/utility/Object.h>
#include <geode/vector/Vector.h>
#include <limits>
namespace geode {

template<class TV> class Box;
using std::numeric_limits;

template<class TV> class Sobol : public Object {
public:
  GEODE_NEW_FRIEND

  typedef typename TV::Scalar T;
  static_assert(numeric_limits<T>::radix==2,"");
  static const int max_bits = numeric_limits<T>::digits;
  typedef typename mpl::if_c<(max_bits<=30),uint32_t,uint64_t>::type TI; // pick an integer large enough to hold T's mantissa
  static_assert(numeric_limits<TI>::digits>max_bits,"");
  enum Workaround {d=TV::m};

private:
  const TV offset, scales;
  Vector<TI,d> x; // Last result
  TI n;
private:
  GEODE_CORE_EXPORT Sobol(const Box<TV>& box);
public:
  ~Sobol();

  GEODE_CORE_EXPORT TV vector();
};

}
