// Class Sobol: Low discrepancy quasirandom numbers
#pragma once

#include <other/core/random/forward.h>
#include <other/core/array/Array.h>
#include <other/core/python/Object.h>
#include <other/core/vector/Vector.h>
#include <boost/noncopyable.hpp>
#include <boost/mpl/if.hpp>
#include <limits>
namespace other {

template<class TV> class Box;
using std::numeric_limits;

template<class TV>
class Sobol : public Object {
public:
  OTHER_DECLARE_TYPE

  typedef typename TV::Scalar T;
  BOOST_STATIC_ASSERT(numeric_limits<T>::radix==2);
  static const int max_bits = numeric_limits<T>::digits;
  typedef typename mpl::if_c<(max_bits<=30),uint32_t,uint64_t>::type TI; // pick an integer large enough to hold T's mantissa
  BOOST_STATIC_ASSERT(numeric_limits<TI>::digits>max_bits);
  enum Workaround {d=TV::m};

private:
  const TV offset, scales;
  Vector<TI,d> x; // Last result
  TI n;
private:
  OTHER_CORE_EXPORT Sobol(const Box<TV>& box);
public:
  ~Sobol();

  OTHER_CORE_EXPORT TV vector();
};

}
