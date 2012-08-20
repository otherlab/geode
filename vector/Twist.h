//#####################################################################
// Class Twist
//#####################################################################
#pragma once

#include <other/core/vector/Vector.h>
#include <other/core/vector/VectorPolicy.h>
namespace other {

template<class TV> struct IsScalarBlock<Twist<TV> >:public IsScalarBlock<TV>{};
template<class TV> struct IsScalarVectorSpace<Twist<TV> >:public IsScalarVectorSpace<TV>{};
template<class TV> struct is_packed_pod<Twist<TV> >:public mpl::and_<mpl::bool_<(TV::m>1)>,is_packed_pod<typename TV::Scalar> >{};

template<class TV>
class Twist
{
    typedef typename TV::Scalar T;
    typedef typename VectorPolicy<TV>::Spin TSpin;
public:
    typedef T Scalar;

    enum Workaround {dimension=TV::m+TSpin::m,m=dimension};

    TV linear;
    TSpin angular;

    Twist()
        :angular()
    {}

    Twist(const TV& linear,const TSpin& angular)
        :linear(linear),angular(angular)
    {}

    template<class T2> explicit Twist(const Twist<Vector<T2,TV::m> >& twist)
        :linear((TV)twist.linear),angular((TSpin)twist.angular)
    {}

    bool operator==(const Twist& v) const
    {return linear==v.linear && angular==v.angular;}

    bool operator!=(const Twist& v) const
    {return !(*this==v);}

    Twist& operator+=(const Twist& v)
    {linear+=v.linear;angular+=v.angular;return *this;}

    Twist& operator-=(const Twist& v)
    {linear-=v.linear;angular-=v.angular;return *this;}

    Twist& operator*=(const T a)
    {linear*=a;angular*=a;return *this;}

    Twist operator-()
    {return Twist(-linear,-angular);}

    Twist operator+(const Twist& v) const
    {return Twist(linear+v.linear,angular+v.angular);}

    Twist operator-(const Twist& v) const
    {return Twist(linear-v.linear,angular-v.angular);}

    Twist operator*(const T a) const
    {return Twist<TV>(linear*a,angular*a);}

    Vector<T,dimension> get_vector() const
    {return Vector<T,dimension>(linear,angular);}

    void set_vector(const Vector<T,dimension>& vector)
    {vector.get_subvector(0,linear);vector.get_subvector(TV::m,angular);}
};

template<class TV> inline Twist<TV> operator*(const typename TV::Scalar a,const Twist<TV>& v)
{return Twist<TV>(a*v.linear,a*v.angular);}

template<class TV> inline std::istream& operator>>(std::istream& input,Twist<TV>& v)
{input>>v.linear>>v.angular;return input;}

template<class TV> inline std::ostream& operator<<(std::ostream& output,const Twist<TV>& v)
{output<<v.linear<<" "<<v.angular;return output;}

}
