//#####################################################################
// Class Rotation
//#####################################################################
#pragma once

#include <other/core/vector/complex.h>
#include <other/core/vector/Quaternion.h>
#include <boost/utility/enable_if.hpp>
#include <other/core/array/forward.h>
#include <other/core/math/clamp.h>
#include <other/core/math/constants.h>
#include <other/core/math/robust.h>
#include <other/core/python/to_python.h>
#include <other/core/utility/debug.h>
#include <other/core/vector/Matrix.h>
namespace other {

using ::std::sin;
using ::std::cos;
using ::std::atan2;

template<class TV> class Rotation;

template<class TV> struct IsScalarBlock<Rotation<TV> >:public mpl::and_<mpl::bool_<(TV::m>1)>,IsScalarBlock<TV> >{};
template<class TV> struct is_packed_pod<Rotation<TV> >:public mpl::and_<mpl::bool_<(TV::m>1)>,is_packed_pod<typename TV::Scalar> >{};

template<class TV> OTHER_CORE_EXPORT PyObject* to_python(const Rotation<TV>& q) ;
template<class TV> struct FromPython<Rotation<TV> >{OTHER_CORE_EXPORT static Rotation<TV> convert(PyObject* object);};
template<class TV> OTHER_CORE_EXPORT bool rotations_check(PyObject* object) ;

//#####################################################################
// 1D
//#####################################################################
template<class T>
class Rotation<Vector<T,1> >
{
    typedef Vector<T,1> TV;
    typedef Vector<T,0> TSpin;
public:
    typedef T Scalar;

    bool operator==(const Rotation<TV>& r) const
    {return true;}

    bool operator!=(const Rotation<TV>& r) const
    {return false;}

    Rotation<TV>& operator*=(const Rotation<TV>& r)
    {return *this;}

    Rotation<TV> operator*(const Rotation<TV>& r) const
    {return *this;}

    T normalize()
    {return 1;}

    Rotation<TV> normalized() const
    {return *this;}

    Rotation<TV> inverse() const
    {return *this;}

    const TV& inverse_times(const TV& x) const
    {return x;}

    const TV& operator*(const TV& x) const
    {return x;}

    const TSpin& times_spin(const TSpin& spin) const
    {return spin;}

    const Twist<TV>& operator*(const Twist<TV>& twist) const
    {return twist;}

    Vector<T,0> euler_angles() const
    {return Vector<T,0>();}

    static Rotation from_euler_angles(const Vector<T,0>&)
    {return Rotation();}

    Vector<T,0> rotation_vector() const
    {return Vector<T,0>();}

    Matrix<T,1> matrix() const
    {return Matrix<T,1>(1);}

    TV x_axis() const // Q*(1)
    {return TV(1);}

    TV axis(const int axis) const
    {assert(axis==1);return TV(1);}

    static Rotation<TV> from_rotation_vector(const Vector<T,0>)
    {return Rotation<TV>();}

    static Rotation<TV> from_rotated_vector(const TV&,const TV&)
    {return Rotation<TV>();}

    static Rotation<TV> spherical_linear_interpolation(const Rotation<TV>,const Rotation<TV>,const T)
    {return Rotation<TV>();}

    bool is_normalized() const
    {return true;}

    T angle() const
    {return 0;}

    Rotation scale_angle() const
    {return *this;}

    static Rotation average_rotation(const Array<Rotation>&)
    {return Rotation();}

    template<class RW>
    void read(std::istream& input)
    {}

    template<class RW>
    void write(std::ostream& output) const
    {}
};

template<class T>
inline std::ostream& operator<<(std::ostream& output,const Rotation<Vector<T,1> >& r)
{return output;}

template<class T>
inline std::istream& operator>>(std::istream& input,Rotation<Vector<T,1> >& r)
{return input;}

template<class T> inline Hash
hash_reduce(const Rotation<Vector<T,1> >& key)
{return Hash();}

//#####################################################################
// 2D
//#####################################################################
template<class T>
class Rotation<Vector<T,2> >
{
    typedef Vector<T,2> TV;
    typedef Vector<T,1> TSpin;

    std::complex<T> c;

    Rotation(const std::complex<T>& c2)
        :c(c2)
    {}

public:
    typedef T Scalar;

    Rotation()
        :c(1,0)
    {}

    template<class T2> explicit Rotation(const Rotation<Vector<T2,2> >& r)
        :c(r.complex())
    {
        BOOST_STATIC_ASSERT((!boost::is_same<T,T2>::value));
        if(!boost::is_same<T,int>::value) normalize();
    }

    explicit Rotation(const Matrix<T,2>& A)
        :c(A.column(0).normalized().complex())
    {}

    const std::complex<T>& complex() const
    {return c;}

    static Rotation<TV> from_complex(const std::complex<T>& c2)
    {return Rotation<TV>(c2).normalized();}

    bool operator==(const Rotation<TV>& r) const
    {return c==r.c;}

    bool operator!=(const Rotation<TV>& r) const
    {return c!=r.c;}

    Rotation<TV>& operator*=(const Rotation<TV>& r)
    {c*=r.c;return *this;}

    Rotation<TV> operator*(const Rotation<TV>& r) const
    {return Rotation<TV>(c*r.c);}

    Rotation<TV> inverse() const
    {return Rotation<TV>(conj(c));}

    TV operator*(const TV& v) const
    {return TV(c.real()*v.x-c.imag()*v.y,c.imag()*v.x+c.real()*v.y);}

    TV inverse_times(const TV& v) const
    {return TV(c.real()*v.x+c.imag()*v.y,c.real()*v.y-c.imag()*v.x);}

    const TSpin& times_spin(const TSpin& spin) const
    {return spin;}

    Twist<TV> operator*(const Twist<TV>& twist) const
    {return Twist<TV>(*this*(twist.linear),twist.angular);}

    T normalize()
    {return other::normalize(c);}

    Rotation<TV> normalized() const
    {Rotation<TV> r(*this);r.normalize();return r;}

    bool is_normalized(const T tolerance=(T)1e-3) const
    {return abs(c.sqr_magnitude()-(T)1)<=tolerance;}

    void get_rotated_frame(TV& x_axis,TV& y_axis) const
    {assert(is_normalized());x_axis=TV(c.real(),c.imag());y_axis=TV(-c.imag(),c.real());}

    T angle() const
    {return atan2(c.imag(),c.real());}

    Vector<T,1> euler_angles() const
    {return rotation_vector();}

    Vector<T,1> rotation_vector() const
    {return Vector<T,1>(angle());}

    Matrix<T,2> matrix() const
    {return Matrix<T,2>(c.real(),c.imag(),-c.imag(),c.real());}

    TV x_axis() const // Q*(1,0)
    {return TV(c.real(),c.imag());}

    TV y_axis() const // Q*(0,1)
    {return TV(-c.imag(),c.real());}

    TV axis(const int axis) const
    {assert(unsigned(axis)<2);if(axis==0) return x_axis();return y_axis();}

    static Rotation<TV> from_angle(const T& a)
    {return Rotation<TV>(cis(a));}

    static Rotation<TV> from_rotation_vector(const Vector<T,1>& v)
    {return from_angle(v.x);}

    static Rotation<TV> from_rotated_vector(const TV& v1,const TV& v2)
    {return Rotation<TV>(std::complex<T>(v1.x,-v1.y)*std::complex<T>(v2.x,v2.y)).normalized();}

    static Rotation<TV> from_euler_angles(const Vector<T,1>& angle)
    {return from_rotation_vector(angle);}

    Rotation<TV> scale_angle(const T a) const
    {return Rotation<TV>::from_rotation_vector(a*rotation_vector());}

    static Rotation<TV> spherical_linear_interpolation(const Rotation<TV>& r1,const Rotation<TV>& r2,const T t)
    {return r1*(r1.inverse()*r2).scale_angle(t);}

    static Rotation<TV> average_rotation(const Array<Rotation<TV> >& rotations)
    {std::complex<T> sum;for(int i=0;i<rotations.m;i++) sum+=rotations(i).c;return Rotation<TV>(sum.normalized());}
};

template<class T>
inline std::ostream& operator<<(std::ostream& output,const Rotation<Vector<T,2> >& r)
{return output<<r.complex();}

template<class T>
inline std::istream& operator>>(std::istream& input,Rotation<Vector<T,2> >& r)
{complex<T> c;input>>c;r=Rotation<Vector<T,3> >::from_complex(c);return input;}

//#####################################################################
// 3D
//#####################################################################
template<class T>
class Rotation<Vector<T,3> >
{
    typedef Vector<T,3> TV;
    typedef Vector<T,3> TSpin;

    Quaternion<T> q;

    Rotation(const Quaternion<T>& q2)
        :q(q2)
    {}

    Rotation(const T s,const T x,const T y,const T z)
        :q(s,x,y,z)
    {}

    class Unusable {};
public:
    typedef T Scalar;

    Rotation()
        :q(1,0,0,0)
    {}

    template<class T2> explicit Rotation(const Rotation<T2>& r)
        :q(r.quaternion())
    {
        BOOST_STATIC_ASSERT((!boost::is_same<T,T2>::value));
        if(!boost::is_same<T,int>::value) normalize();
    }

    Rotation(const T angle,const TV& direction)
        :q(cos((T).5*angle),direction)
    {
        q.v.normalize();q.v*=sin((T).5*angle);
    }

    explicit Rotation(const Matrix<T,3>& A) // matches A with a quaternion
    {
        T trace=1+A(0,0)+A(1,1)+A(2,2);// trace=4*cos^2(theta/2)
        if(trace>1){q.s=(T).5*sqrt(trace);q.v.x=A(2,1)-A(1,2);q.v.y=A(0,2)-A(2,0);q.v.z=A(1,0)-A(0,1);q.v*=(T).25/q.s;}
        else{int i=A(0,0)>A(1,1)?0:1;i=A(i,i)>A(2,2)?i:2; // set i to be the index of the dominating diagonal term
            switch(i){
                case 0:q.v.x=T(.5)*sqrt(1+A(0,0)-A(1,1)-A(2,2));q.v.y=T(.25)*(A(1,0)+A(0,1))/q.v.x;q.v.z=T(.25)*(A(0,2)+A(2,0))/q.v.x;q.s=T(.25)*(A(2,1)-A(1,2))/q.v.x;break;
                case 1:q.v.y=T(.5)*sqrt(1-A(0,0)+A(1,1)-A(2,2));q.v.x=T(.25)*(A(1,0)+A(0,1))/q.v.y;q.v.z=T(.25)*(A(2,1)+A(1,2))/q.v.y;q.s=T(.25)*(A(0,2)-A(2,0))/q.v.y;break;
                case 2:default:q.v.z=T(.5)*sqrt(1-A(0,0)-A(1,1)+A(2,2));q.v.x=T(.25)*(A(0,2)+A(2,0))/q.v.z;q.v.y=T(.25)*(A(2,1)+A(1,2))/q.v.z;q.s=T(.25)*(A(1,0)-A(0,1))/q.v.z;break;}}
        normalize();
    }

    static Rotation<TV> from_components(const typename mpl::if_<boost::is_same<T,int>,Unusable,T>::type s,const T x,const T y,const T z)
    {return Rotation<TV>(s,x,y,z).normalized();}

    static Rotation<TV> from_components(const typename mpl::if_<boost::is_same<T,int>,int,Unusable>::type s,const int x,const int y,const int z)
    {return Rotation<TV>(s,x,y,z);}

    const Quaternion<T>& quaternion() const
    {return q;}

    static Rotation<TV> from_quaternion(const Quaternion<T>& q)
    {return Rotation<TV>(q).normalized();}

    bool operator==(const Rotation<TV>& r) const
    {return q==r.q || q==-r.q;}

    bool operator!=(const Rotation<TV>& r) const
    {return !(*this==r);}

    Rotation<TV>& operator*=(const Rotation<TV>& r)
    {q*=r.q;return *this;}

    Quaternion<T> operator*(const Quaternion<T>& w) const
    {return q*w;}

    Rotation<TV> operator*(const Rotation<TV>& r) const // 16 mult and 13 add/sub
    {return Rotation<TV>(q*r.q);}

    T normalize()
    {return q.normalize();}

    Rotation<TV> normalized() const
    {Rotation<TV> r(*this);r.normalize();return r;}

    bool is_normalized(const T tolerance=(T)1e-4) const
    {return q.is_normalized(tolerance);}

    Rotation<TV> inverse() const
    {return Rotation<TV>(conj(q));}

    TV rotation_vector() const
    {return 2*atan2_y_x_over_y(q.v.magnitude(),abs(q.s))*(q.s<0?-q.v:q.v);}

    static Rotation<TV> from_rotation_vector(const TV& v)
    {T magnitude=v.magnitude();Rotation<TV> r;r.q.s=cos((T).5*magnitude);r.q.v=(T).5*sinc((T).5*magnitude)*v;return r;}

    static Rotation<TV> from_euler_angle(const int axis,const T euler_angle) // 1 mult, 1 sin, 1 cos
    {T half_angle=(T).5*euler_angle,c=cos(half_angle),s=sin(half_angle);
    Rotation<TV> r;r.q.s=c;r.q.v[axis]=s;return r;}

    // rotation about fixed axes x, then y, then z
    static Rotation<TV> from_euler_angles(const T euler_angle_x,const T euler_angle_y,const T euler_angle_z) // 20 mults, 4 adds, 3 cos, 3 sin
    {T half_x_angle=(T).5*euler_angle_x,half_y_angle=(T).5*euler_angle_y,half_z_angle=(T).5*euler_angle_z;
    T cx=cos(half_x_angle),sx=sin(half_x_angle),cy=cos(half_y_angle),sy=sin(half_y_angle),cz=cos(half_z_angle),sz=sin(half_z_angle);
    return Rotation<TV>(cx*cy*cz+sx*sy*sz,sx*cy*cz-cx*sy*sz,cx*sy*cz+sx*cy*sz,cx*cy*sz-sx*sy*cz);}

    static Rotation<TV> from_euler_angles(const TV& euler_angles)
    {return from_euler_angles(euler_angles.x,euler_angles.y,euler_angles.z);}

    TV operator*(const TV& v) const // 20 mult and 13 add/sub
    {assert(is_normalized());T two_s=q.s+q.s;return two_s*cross(q.v,v)+(two_s*q.s-(T)1)*v+(T)2*dot(q.v,v)*q.v;}

    TV inverse_times(const TV& v) const // 20 mult and 13 add/sub
    {assert(is_normalized());T two_s=q.s+q.s;return two_s*cross(v,q.v)+(two_s*q.s-(T)1)*v+(T)2*dot(q.v,v)*q.v;}

    TSpin times_spin(const TSpin& spin) const
    {return *this*spin;}

    Twist<TV> operator*(const Twist<TV>& twist) const
    {return Twist<TV>(*this*twist.linear,*this*twist.angular);}

    TV euler_angles() const
    {Matrix<T,3> R(matrix());T cos_beta_squared=sqr(R(0,0))+sqr(R(1,0));
    if(cos_beta_squared<1e-30)
      return TV(0,R(2,0)<0?T(.5*pi):-T(.5*pi),-atan2(R(0,1),R(1,1)));
    else
      return TV(
        atan2(R(2,1),R(2,2)), // between -pi and pi
        atan2(-R(2,0),sqrt(cos_beta_squared)), // between -pi/2 and pi/2
        atan2(R(1,0),R(0,0)));} // between -pi and pi

    TV axis(const int axis) const
    {assert(axis>=1 && axis<=3);if(axis==1) return x_axis();if(axis==2) return y_axis();return z_axis();}

    TV x_axis() const // Q*(1,0,0)
    {T vy2=sqr(q.v.y),vz2=sqr(q.v.z),vxvy=q.v.x*q.v.y,vxvz=q.v.x*q.v.z,svy=q.s*q.v.y,svz=q.s*q.v.z;
    return TV(1-2*(vy2+vz2),2*(vxvy+svz),2*(vxvz-svy));}

    TV y_axis() const // Q*(0,1,0)
    {T vx2=sqr(q.v.x),vz2=sqr(q.v.z),vxvy=q.v.x*q.v.y,vyvz=q.v.y*q.v.z,svx=q.s*q.v.x,svz=q.s*q.v.z;
    return TV(2*(vxvy-svz),1-2*(vx2+vz2),2*(vyvz+svx));}

    TV z_axis() const // Q*(0,0,1)
    {T vx2=sqr(q.v.x),vy2=sqr(q.v.y),vxvz=q.v.x*q.v.z,vyvz=q.v.y*q.v.z,svx=q.s*q.v.x,svy=q.s*q.v.y;
    return TV(2*(vxvz+svy),2*(vyvz-svx),1-2*(vx2+vy2));}

    void get_rotated_frame(TV& x_axis,TV& y_axis,TV& z_axis) const
    {Matrix<T,3> M=matrix();x_axis=M.column(1);y_axis=M.column(2);z_axis=M.column(3);}

    void get_angle_axis(T& angle,TV& axis) const
    {axis=q.s<0?-q.v:q.v;angle=2*atan2(axis.normalize(),abs(q.s));}

    T angle() const
    {return 2*atan2(q.v.magnitude(),abs(q.s));}

    TV get_axis() const
    {return (q.s<0?-q.v:q.v).normalized();}

    Matrix<T,3> matrix() const // 12 mult and 12 add/sub
    {TV v2 = (T)2*q.v;T xx=q.v.x*v2.x,yy=q.v.y*v2.y,zz=q.v.z*v2.z,xy=q.v.x*v2.y,xz=q.v.x*v2.z,yz=q.v.y*v2.z,sx=q.s*v2.x,sy=q.s*v2.y,sz=q.s*v2.z;
    return Matrix<T,3>(1-yy-zz,xy+sz,xz-sy,xy-sz,1-xx-zz,yz+sx,xz+sy,yz-sx,1-xx-yy);}

    Rotation<TV> scale_angle(const T a) const
    {T angle;TV axis;get_angle_axis(angle,axis);return Rotation<TV>(a*angle,axis);}

    static Rotation<TV> spherical_linear_interpolation(const Rotation<TV>& r1,const Rotation<TV>& r2,const T t)
    {return r1*(r1.inverse()*r2).scale_angle(t);}

    static Rotation<TV> average_rotation(const Array<Rotation<TV> >& rotations)
    {if(rotations.m==0) return Rotation<TV>();Array<Rotation<TV> > r(rotations);
    for(int i=1;i<r.m;i+=2) r.append(spherical_linear_interpolation(r(i),r(i+1),(T).5));return r(r.m);}

    static Rotation<TV> from_rotated_vector(const TV& initial_vector,const TV& final_vector)
    {TV initial_unit=initial_vector.normalized(),final_unit=final_vector.normalized();
    T cos_theta=clamp(dot(initial_unit,final_unit),(T)-1,(T)1);
    TV v=cross(initial_unit,final_unit);
    T v_magnitude=v.magnitude();
    if(v_magnitude==0){ // initial and final vectors are collinear
      v=initial_unit.orthogonal_vector();
      v_magnitude=v.magnitude();}
    T s_squared=(T).5*(1+cos_theta); // uses the half angle formula
    T v_magnitude_desired=sqrt(1-s_squared);v*=v_magnitude_desired/v_magnitude;
    return Rotation<TV>(sqrt(s_squared),v.x,v.y,v.z);}

    template<class RW>
    void read(std::istream& input)
    {q.template read<RW>(input);
    if(!boost::is_same<T,int>::value && !is_normalized()) OTHER_FATAL_ERROR("Read nonnormalized rotation");}

    template<class RW>
    void write(std::ostream& output) const
    {q.template write<RW>(output);}
};
//#####################################################################

template<> inline Vector<int,3> Rotation<Vector<int,3> >::
operator*(const Vector<int,3>& v) const // homogenous of degree 2 in q, since we can't usefully assume normalization for integer case
{return 2*q.s*cross(q.v,v)+(q.s*q.s-q.v.sqr_magnitude())*v+2*dot(q.v,v)*q.v;}

template<> inline Vector<int,3> Rotation<Vector<int,3> >::
inverse_times(const Vector<int,3>& v) const // homogenous of degree 2 in q, since we can't usefully assume normalization for integer case
{return 2*q.s*cross(v,q.v)+(q.s*q.s-q.v.sqr_magnitude())*v+2*dot(q.v,v)*q.v;}

template<class T>
inline std::ostream& operator<<(std::ostream& output,const Rotation<Vector<T,3> >& r)
{return output<<r.quaternion();}

template<class T>
inline std::istream& operator>>(std::istream& input,Rotation<Vector<T,3> >& r)
{Quaternion<T> q;input>>q;r=Rotation<Vector<T,3> >::from_quaternion(q);return input;}

}
