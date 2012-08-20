//#####################################################################
// Class Sobol
//#####################################################################
// See Bratley and Fox. 1988. Algorithm 659: Implementing Sobol's quasirandom sequence generator. Acm Trans. Math. Softw. 14, 88-100.
//#####################################################################
#include <other/core/random/Sobol.h>
#include <other/core/geometry/Box.h>
#include <other/core/math/integer_log.h>
#include <other/core/python/Class.h>
namespace other {

template<> OTHER_DEFINE_TYPE(Sobol<Vector<real,1> >);
template<> OTHER_DEFINE_TYPE(Sobol<Vector<real,2> >);
template<> OTHER_DEFINE_TYPE(Sobol<Vector<real,3> >);

namespace{
const int max_dimension=4;
const Vector<int,max_dimension> polynomial_degree(1,2,3,3);
const Vector<int,max_dimension> polynomial_value(0,1,1,2); // represents x+1, x^2+x+1, x^3+x+1, x^3+x^2+1
int m_initial[max_dimension][max_dimension]={{1},{1,1},{1,3,7},{1,3,3}}; // Todo: should be const, but gcc 4.0.2 is broken.
}

template<class TV> Sobol<TV>::
Sobol(const Box<TV>& box)
    :offset(box.min),scales(box.sizes()/(T)((TI)1<<max_bits))
{
    BOOST_STATIC_ASSERT(d<=max_dimension);

    // compute direction numbers
    v.resize(max_bits);
    for(int i=0;i<d;i++){
        const int degree=polynomial_degree[i];
        const int polynomial=polynomial_value[i];
        // fill in initial values for m (taken from Numerical recipes, since according to Bratley and Fox optimal values satisfy complicated conditions
        Array<TI> m(v.size());
        for(int j=0;j<degree;j++) m(j)=m_initial[i][j];
        // fill in rest of m using recurrence
        for(int j=degree;j<v.size();j++){
            m(j)=(m(j-degree)<<degree)^m(j-degree);
            for(int k=0;k<degree-1;k++)
                if(polynomial&(1<<k)) m(j)^=m(j-k-1)<<(k+1);}
        // compute direction vectors (stored as Vi * 2^v.m)
        for(int j=0;j<v.size();j++)
            v(j)[i]=m(j)<<(v.size()-j-1);}

    // start counting
    n=0;
}

template<class TV> Sobol<TV>::
~Sobol()
{}

template<class TV> TV Sobol<TV>::
get_vector()
{
    int rightmost_zero_position=integer_log_exact(min_bit(~n));
    OTHER_ASSERT(rightmost_zero_position<v.size(),"Ran out of bits (this means floating point precision has already been exhausted)");
    const Vector<TI,d>& vc=v(rightmost_zero_position);
    for(int i=0;i<d;i++) x[i]^=vc[i];
    n++;
    return offset+scales*TV(x);
}

template class Sobol<Vector<real,1> >;
template class Sobol<Vector<real,2> >;
template class Sobol<Vector<real,3> >;

}
using namespace other;

template<int d> static void wrap_helper() {
  typedef Sobol<Vector<real,d> > Self;
  Class<Self>(d==1?"Sobol1d":d==2?"Sobol2d":"Sobol3d")
    .OTHER_INIT(Box<Vector<real,d> >)
    .OTHER_METHOD(get_vector)
    ;
}

void wrap_sobol() {
  wrap_helper<1>();
  wrap_helper<2>();
  wrap_helper<3>();
}
