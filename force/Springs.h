//#####################################################################
// Class Springs
//#####################################################################
//
// Linear spring force
// 
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
#include <other/core/force/Force.h>
#include <other/core/vector/Vector.h>
#include <other/core/geometry/Box.h>
namespace other{

template<class TV> struct SpringInfo
{
    typedef typename TV::Scalar T;
    T restlength,stiffness,damping;
    T length;
    T alpha,beta; // offdiagonal differential is alpha + beta outer(direction)
    TV direction;
};

template<class TV>
class Springs:public Force<TV>
{
public:
    OTHER_DECLARE_TYPE
    typedef Force<TV> Base;
    typedef real T;
    enum {m=TV::m};

    const Array<const Vector<int,2> > springs;
    bool resist_compression;
    Box<T> strain_range;
    T off_axis_damping; // between 0 and 1
private:
    const int nodes;
    Array<const T> mass;
    Array<const TV> X;
    const Array<SpringInfo<TV> > info;
protected:
    Springs(Array<const Vector<int,2> > springs,Array<const T> mass,Array<const TV> X,NdArray<const T> stiffness,NdArray<const T> damping_ratio);
public:
    ~Springs();

    Array<T> restlengths() const;

    void update_position(Array<const TV> X,bool definite);
    void add_frequency_squared(RawArray<T> frequency_squared) const;
    T elastic_energy() const;
    void add_elastic_force(RawArray<TV> F) const;
    void add_elastic_differential(RawArray<TV> dF,RawArray<const TV> dX) const;
    void add_elastic_gradient_block_diagonal(RawArray<SymmetricMatrix<T,m> > dFdX) const;
    T damping_energy(RawArray<const TV> V) const;
    void add_damping_force(RawArray<TV> F,RawArray<const TV> V) const;
    T strain_rate(RawArray<const TV> V) const;

    Box<T> limit_strain(RawArray<TV> X) const;

    void structure(SolidMatrixStructure& structure) const;
    void add_elastic_gradient(SolidMatrix<TV>& matrix) const;
    void add_damping_gradient(SolidMatrix<TV>& matrix) const;
};
}
