//#####################################################################
// Class Frame
//#####################################################################
#pragma once

#include <other/core/vector/Rotation.h>
#include <other/core/vector/Vector.h>
#include <other/core/vector/VectorPolicy.h>
#include <other/core/utility/HasCheapCopy.h>
#include <other/core/math/One.h>
namespace other {

template<class TV> struct HasCheapCopy<Frame<TV> >:public mpl::true_{};
template<class TV> struct IsScalarBlock<Frame<TV> >:public IsScalarBlock<TV>{};
template<class TV> struct is_packed_pod<Frame<TV> >:public is_packed_pod<typename TV::Scalar>{};

template<class TV> PyObject* to_python(const Frame<TV>& q) OTHER_EXPORT;
template<class TV> struct FromPython<Frame<TV> >{OTHER_EXPORT static Frame<TV> convert(PyObject* object);};

template<class TV>
class Frame
{
    typedef typename TV::Scalar T;
    enum Workaround {d=TV::m};
public:
    typedef T Scalar;

    TV t; // defaults to 0
    Rotation<TV> r; // defaults to 1

    Frame()
    {}

    explicit Frame(const TV& t)
        :t(t)
    {}

    explicit Frame(const Rotation<TV>& r)
        :r(r)
    {}

    Frame(const TV& t,const Rotation<TV>& r)
        :t(t),r(r)
    {}

    explicit Frame(const Matrix<T,d+1>& m_input)
        :t(m_input.translation()),r(m_input.linear())
    {}

    template<class T2> explicit Frame(const Frame<Vector<T2,d> >& f)
        :t(f.t),r(f.r)
    {}

    bool operator==(const Frame& f) const
    {return t==f.t && r==f.r;}

    bool operator!=(const Frame& f) const
    {return t!=f.t || r!=f.r;}

    Frame& operator*=(const Frame& f)
    {t+=r*f.t;r*=f.r;return *this;}

    Frame operator*(const Frame& f) const
    {return Frame(t+r*f.t,r*f.r);}

    TV operator*(const TV& v) const
    {return t+r*v;}
    
    void invert()
    {*this=inverse();}

    Frame inverse() const
    {Rotation<TV> r_inverse=r.inverse();return Frame(-(r_inverse*t),r_inverse);}

    TV inverse_times(const TV& v) const
    {return r.inverse_times(v-t);}

    Frame inverse_times(const Frame& f) const
    {return Frame(r.inverse_times(f.t-t),r.inverse()*f.r);}

    static Frame interpolation(const Frame& f1,const Frame& f2,const T s)
    {return Frame((1-s)*f1.t+s*f2.t,Rotation<TV>::spherical_linear_interpolation(f1.r,f2.r,s));}

    Matrix<T,d+1> matrix() const
    {Matrix<T,d+1> matrix=Matrix<T,d+1>::from_linear(r.matrix());matrix.set_translation(t);return matrix;}

    const Frame& frame() const
    {return *this;}
};

template<class TV> Frame<TV> rotation_around(const TV& center, const Rotation<TV>& r) {
    return Frame<TV>(center)*Frame<TV>(r)*Frame<TV>(center).inverse();
}

template<class T> Frame<Vector<T,2> > rotation_around(const Vector<T,2>& center, const T& theta) {
    return Frame<Vector<T,2> >(center)*Frame<Vector<T,2> >(Rotation<Vector<T,2> >::from_angle(theta))*Frame<Vector<T,2> >(center).inverse();
}

// global functions
template<class TV> inline std::istream& operator>>(std::istream& input,Frame<TV>& f)
{input>>f.t>>f.r;return input;}

template<class TV> inline std::ostream& operator<<(std::ostream& output,const Frame<TV>& f)
{output<<f.t<<" "<<f.r;return output;}

template<class TV> static inline string repr(const Frame<TV>& f)
{return format("Frames(%s,%s)",tuple_repr(f.t),repr(f.r));}

}
