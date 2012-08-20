//#####################################################################
// Class Complex
//#####################################################################
#pragma once

#include <other/core/vector/Vector2d.h>
namespace other{

using ::std::sin;
using ::std::cos;

template<class T> struct IsScalarBlock<Complex<T> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<Complex<T> >:public IsScalarVectorSpace<T>{};
template<class T> struct is_packed_pod<Complex<T> >:public is_packed_pod<T>{};

template<class T>
class Complex
{
public:
    typedef T Scalar;

    T re,im;

    Complex()
        :re(0),im(0)
    {}

    Complex(T re_input,T im_input)
        :re(re_input),im(im_input)
    {}

    template<class T2> explicit Complex(const Complex<T2>& complex_input)
        :re((T)complex_input.re),im((T)complex_input.im)
    {}

    explicit Complex(const Vector<T,2>& input)
        :re(input.x),im(input.y)
    {}

    Vector<T,2> vector() const
    {return Vector<T,2>(re,im);}

    static Complex<T> one()
    {return Complex(1,0);}

    bool operator==(const Complex<T>& c) const
    {return re==c.re && im==c.im;}

    bool operator!=(const Complex<T>& c) const
    {return re!=c.re || im!=c.im;}

    Complex<T>& operator*=(const Complex<T>& c)
    {T old_re=re;re=re*c.re-im*c.im;im=old_re*c.im+im*c.re;return *this;}

    Complex<T>& operator*=(const T a)
    {re*=a;im*=a;return *this;}

    Complex<T> operator*(const Complex<T>& c) const
    {return Complex<T>(re*c.re-im*c.im,re*c.im+im*c.re);}

    Complex<T> operator*(const T a) const
    {return Complex<T>(a*re,a*im);}

    Complex<T>& operator+=(const Complex<T>& c)
    {re+=c.re;im+=c.im;return *this;}

    Complex<T> operator+(const Complex<T>& c) const
    {return Complex<T>(re+c.re,im+c.im);}

    Complex<T> operator+(const T& a) const
    {return Complex<T>(re+a,im);}

    Complex<T> operator-=(const Complex<T>& c)
    {re-=c.re;im-=c.im;return *this;}

    Complex<T> operator-(const Complex<T>& c) const
    {return Complex<T>(re-c.re,im-c.im);}

    Complex<T> operator-(const T a) const
    {return Complex<T>(re-a,im);}

    Complex<T> operator-() const
    {return Complex<T>(-re,-im);}

    T sqr_magnitude() const
    {return sqr(re)+sqr(im);}

    T magnitude() const
    {return std::sqrt(sqr(re)+sqr(im));}

    void conjugate()
    {im*=-1;}

    Complex<T> conjugated() const
    {return Complex<T>(re,-im);}

    Complex<T> inverse() const
    {assert(re!=0 || im!=0);T denominator=(T)1/(re*re+im*im);return Complex<T>(re*denominator,-im*denominator);}

    Complex<T> sqrt() const
    {T mag=magnitude();return Complex<T>(sqrt((T).5*(mag+re)),sign(im)*sqrt((T).5*(mag-re)));}

    T normalize()
    {T mag=magnitude();if(mag) *this*=1/mag;else{re=1;im=0;}return mag;}

    Complex<T> normalized() const
    {Complex<T> c(*this);c.normalize();return c;}

    bool Is_Normalized(const T tolerance=(T)1e-3) const
    {return abs(sqr_magnitude()-(T)1)<=tolerance;}

    Complex<T> rotated_counter_clockwise_90() const
    {return Complex<T>(-im,re);}

    Complex<T> rotated_clockwise_90() const
    {return Complex<T>(im,-re);}

    static Complex<T> polar(const T r,const T theta)   // r*e^(i*theta) = r(cos(theta)+i*sin(theta))
    {return Complex<T>(r*cos(theta),r*sin(theta));}

    static Complex<T> unit_polar(const T theta)
    {return Complex<T>(cos(theta),sin(theta));}

    T arg() const
    {return atan2(re,im);}
};

template<class T>
inline Complex<T> operator*(const T a,const Complex<T>& c)
{return Complex<T>(a*c.re,a*c.im);}

template<class T>
inline T dot(const Complex<T>& c1,const Complex<T>& c2)
{return c1.re*c2.re+c1.im*c2.im;}

template<class T>
inline std::ostream& operator<<(std::ostream& output,const Complex<T>& c)
{output<<c.re<<" "<<c.im;return output;}

}
