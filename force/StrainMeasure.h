//#####################################################################
// Class StrainMeasure
//#####################################################################
#pragma once

#include <other/core/array/Array.h>
#include <other/core/vector/Vector.h>
#include <other/core/utility/debug.h>
namespace other{

template<class TV,int d>
class StrainMeasure:public Object
{
    typedef typename TV::Scalar T;
    typedef Matrix<T,TV::m,d> TMatrix;
    template<int d2> struct Unusable{};
public:
    OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
    typedef Object Base;

    const int nodes;
    const Array<const Vector<int,d+1> > elements;
    const Array<UpperTriangularMatrix<T,d> > Dm_inverse;

protected:
    StrainMeasure(Array<const Vector<int,d+1> > elements,RawArray<const TV> X);
public:
    ~StrainMeasure();

    TMatrix F(RawArray<const TV> X,const int simplex) const
    {return Ds(X,simplex)*Dm_inverse(simplex);}

    T J(RawArray<const TV> X,const int simplex) const
    {return Ds(X,simplex).parallelpiped_measure()*Dm_inverse(simplex).determinant();}

    T rest_altitude(const int simplex) const
    {return Dm_inverse(simplex).inverse().simplex_minimum_altitude();}

    T minimum_rest_altitude() const
    {T altitude=FLT_MAX;for(int t=1;t<=Dm_inverse.m;t++) altitude=min(altitude,rest_altitude(t));return altitude;}

    TMatrix Ds(RawArray<const TV> X,const typename mpl::if_c<d==1,int,Unusable<1> >::type simplex) const
    {int i,j;elements[simplex].get(i,j);
    return TMatrix(X(j)-X(i));}

    TMatrix Ds(RawArray<const TV> X,const typename mpl::if_c<d==2,int,Unusable<2> >::type simplex) const
    {int i,j,k;elements[simplex].get(i,j,k);
    return TMatrix(X(j)-X(i),X(k)-X(i));}

    TMatrix Ds(RawArray<const TV> X,const typename mpl::if_c<d==3,int,Unusable<3> >::type simplex) const
    {int i,j,k,l;elements[simplex].get(i,j,k,l);
    return TMatrix(X(j)-X(i),X(k)-X(i),X(l)-X(i));}

    template<class TArray>
    static TMatrix Ds(const TArray& X,const Vector<int,2>& nodes)
    {int i,j;nodes.get(i,j);return TMatrix(X(j)-X(i));}

    template<class TArray>
    static TMatrix Ds(const TArray& X,const Vector<int,3>& nodes)
    {int i,j,k;nodes.get(i,j,k);return TMatrix(X(j)-X(i),X(k)-X(i));}

    template<class TArray>
    static TMatrix Ds(const TArray& X,const Vector<int,4>& nodes)
    {int i,j,k,l;nodes.get(i,j,k,l);return TMatrix(X(j)-X(i),X(k)-X(i),X(l)-X(i));}

    void distribute_force(RawArray<TV> F,const int element,const TMatrix& forces) const
    {distribute_force(F,elements[element],forces);}

    static void distribute_force(RawArray<TV> F,const Vector<int,3>& nodes,const Matrix<T,TV::m,2>& forces)
    {int i,j,k;nodes.get(i,j,k);
    F(i)-=forces.column(0)+forces.column(1);
    F(j)+=forces.column(0);
    F(k)+=forces.column(1);}

    static void distribute_force(RawArray<TV> F,const Vector<int,4>& nodes,const Matrix<T,3>& forces)
    {int i,j,k,l;nodes.get(i,j,k,l);
    F(i)-=forces.column(0)+forces.column(1)+forces.column(2);
    F(j)+=forces.column(0);
    F(k)+=forces.column(1);
    F(l)+=forces.column(2);}

    void initialize_rest_state_to_equilateral(const T side_length);
    void print_altitude_statistics();
};
}
