//#####################################################################
// Class Force
//#####################################################################
#include <other/core/force/Force.h>
#include <other/core/python/Class.h>
#include <other/core/vector/SolidMatrix.h>
#include <other/core/vector/SymmetricMatrix.h>
namespace other{

typedef real T;
template<> OTHER_DEFINE_TYPE(Force<Vector<T,2> >)
template<> OTHER_DEFINE_TYPE(Force<Vector<T,3> >)

template<class TV> Force<TV>::
Force()
{}

template<class TV> Force<TV>::
~Force()
{}

template<class TV> Array<TV> Force<TV>::
elastic_gradient_block_diagonal_times(RawArray<TV> dX) const
{
    Array<SymmetricMatrix<T,d> > dFdX(dX.size());
    add_elastic_gradient_block_diagonal(dFdX);
    Array<TV> dF(dX.size(),false);
    for(int i=0;i<dX.size();i++)
        dF[i] = dFdX[i]*dX[i];
    return dF;
}

template class Force<Vector<T,2> >;
template class Force<Vector<T,3> >;
}

using namespace other;

template<int d> static void wrap_helper()
{
    typedef Force<Vector<T,d> > Self;
    Class<Self>(d==2?"Force2d":"Force3d")
        .OTHER_METHOD(update_position)
        .OTHER_METHOD(elastic_energy)
        .OTHER_METHOD(add_elastic_force)
        .OTHER_METHOD(add_elastic_differential)
        .OTHER_METHOD(damping_energy)
        .OTHER_METHOD(add_damping_force)
        .OTHER_METHOD(add_frequency_squared)
        .OTHER_METHOD(strain_rate)
        .OTHER_METHOD(structure)
        .OTHER_METHOD(add_elastic_gradient)
        .OTHER_METHOD(add_damping_gradient)
        .OTHER_METHOD(elastic_gradient_block_diagonal_times)
        ;
}

void wrap_Force()
{
    wrap_helper<2>();
    wrap_helper<3>();
}
