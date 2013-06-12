//#####################################################################
// Class Quaternion
//#####################################################################
#pragma once

#include <other/core/vector/Vector3d.h>
namespace other {

template<class T> struct IsScalarBlock<Quaternion<T> >:public IsScalarBlock<T>{};
template<class T> struct IsScalarVectorSpace<Quaternion<T> >:public IsScalarVectorSpace<T>{};
template<class T> struct is_packed_pod<Quaternion<T> >:public is_packed_pod<T>{};

template<class T>
class Quaternion
{
    typedef Vector<T,3> TV;
public:
    typedef T Scalar;

    T s;
    TV v;

    Quaternion()
        :s() // note that v is also zeroed
    {}

    template<class T2> explicit Quaternion(const Quaternion<T2>& q)
        :s((T)q.s),v(q.v)
    {}

    Quaternion(const T s,const T x,const T y,const T z)
        :s(s),v(x,y,z)
    {}

    Quaternion(const T s,const TV& v)
        :s(s),v(v)
    {}

    explicit Quaternion(const Vector<T,4>& q)
        :s(q[0]),v(q[1],q[2],q[3])
    {}

    Vector<T,4> vector() const
    {return Vector<T,4>(s,v.x,v.y,v.z);}

    static Quaternion one()
    {return Quaternion(1,0,0,0);}

    bool operator==(const Quaternion& q) const
    {return s==q.s && v==q.v;}

    bool operator!=(const Quaternion& q) const
    {return s!=q.s || v!=q.v;}

    Quaternion operator-() const
    {return Quaternion(-s,-v);}

    Quaternion& operator+=(const Quaternion& q)
    {s+=q.s;v+=q.v;return *this;}

    Quaternion& operator-=(const Quaternion& q)
    {s-=q.s;v-=q.v;return *this;}

    Quaternion& operator*=(const Quaternion& q)
    {return *this=*this*q;}

    Quaternion& operator*=(const T a)
    {s*=a;v*=a;return *this;}

    Quaternion& operator/=(const T a)
    {assert(a!=0);T r=1/a;s*=r;v*=r;return *this;}

    Quaternion operator+(const Quaternion& q) const
    {return Quaternion(s+q.s,v+q.v);}

    Quaternion operator-(const Quaternion& q) const
    {return Quaternion(s-q.s,v-q.v);}

    Quaternion operator*(const Quaternion& q) const // 16 mult and 13 add/sub
    {return Quaternion(s*q.s-dot(v,q.v),s*q.v+q.s*v+cross(v,q.v));}

    Quaternion operator*(const T a) const
    {return Quaternion(s*a,v*a);}

    Quaternion operator/(const T a) const
    {assert(a!=0);T r=1/a;return Quaternion(s*r,v*r);}

    T magnitude() const
    {return sqrt(sqr_magnitude());}

    T sqr_magnitude() const
    {return sqr(s)+v.sqr_magnitude();}

    T maxabs() const
    {return maxabs(s,v.maxabs());}

    T normalize()
    {T mag=magnitude();if(mag) *this/=mag;else *this=one();return mag;}

    Quaternion normalized() const
    {Quaternion q(*this);q.normalize();return q;}

    bool is_normalized(const T tolerance=(T)1e-3) const
    {return abs(sqr_magnitude()-(T)1)<=tolerance;}

    Quaternion inverse() const
    {return Quaternion(s,-v)/sqr_magnitude();}
};

template<class T> static inline Quaternion<T> conj(const Quaternion<T>& q) {
  return Quaternion<T>(q.s,-q.v);
}

template<class T>
inline Quaternion<T> operator*(const T a,const Quaternion<T>& q)
{return Quaternion<T>(q.s*a,q.v*a);}

template<class T>
inline T dot(const Quaternion<T>& q1,const Quaternion<T>& q2)
{return q1.s*q2.s+dot(q1.v,q2.v);}

template<class T>
inline std::ostream& operator<<(std::ostream& output,const Quaternion<T>& q)
{output<<q.s<<" "<<q.v;return output;}

template<class T>
inline std::istream& operator>>(std::istream& input,Quaternion<T>& q)
{input>>q.s>>q.v;return input;}

}
