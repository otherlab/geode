//#####################################################################
// Class Sobol
//#####################################################################
#pragma once

#include <other/core/random/forward.h>
#include <other/core/array/Array.h>
#include <other/core/python/Object.h>
#include <other/core/vector/Vector.h>
#include <boost/noncopyable.hpp>
#include <boost/mpl/if.hpp>
#include <limits>
namespace other{

template<class TV> class Box;

template<class TV>
class Sobol:public Object
{
    typedef typename TV::Scalar T;
    BOOST_STATIC_ASSERT(std::numeric_limits<T>::radix==2);
    static const int max_bits=std::numeric_limits<T>::digits;
    typedef typename mpl::if_c<(max_bits<=30),uint32_t,uint64_t>::type TI; // pick an integer large enough to hold T's mantissa
    BOOST_STATIC_ASSERT(std::numeric_limits<TI>::digits>max_bits);
    enum Workaround {d=TV::m};
public:
    OTHER_DECLARE_TYPE
private:
    const TV offset,scales;
    Array<Vector<TI,d> > v; // direction numbers
    Vector<TI,d> x; // last result
    TI n;
private:
    Sobol(const Box<TV>& box);
public:
    ~Sobol();

//#####################################################################
    TV get_vector();
//#####################################################################
};
}
