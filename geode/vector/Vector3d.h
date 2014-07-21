//#####################################################################
// Class Vector3d
//#####################################################################
#pragma once

#include <geode/vector/Vector2d.h>
#include <geode/math/clamp.h>
#include <geode/math/inverse.h>
#include <geode/math/argmax.h>
#include <geode/math/argmin.h>
#include <geode/math/isnan.h>
#include <geode/math/max.h>
#include <geode/math/maxabs.h>
#include <geode/math/min.h>
#include <geode/math/sqr.h>
#include <geode/math/wrap.h>
#include <geode/math/cyclic_shift.h>
#include <cmath>
namespace geode {

using ::std::exp;
using ::std::sin;
using ::std::cos;
using ::std::pow;
using ::std::ceil;
using ::std::floor;

template<class TArray,class TIndices> class IndirectArray;

template<class T>
class Vector<T,3>
{
    struct Unusable{};
public:
    typedef typename mpl::if_<IsScalar<T>,T,Unusable>::type Scalar;
    typedef T Element;
    typedef T value_type; // for stl
    typedef T* iterator; // for stl
    typedef const T* const_iterator; // for stl
    template<class V> struct result;
    template<class V> struct result<V(int)>:mpl::if_<is_const<V>,const T&,T&>{};
    enum Workaround1 {dimension=3};
    enum Workaround2 {m=3};
    static const bool is_const=false;

    T x,y,z;

    Vector()
        :x(),y(),z()
    {
        static_assert(sizeof(Vector)==3*sizeof(T),"");
    }

    Vector(const T& x,const T& y,const T& z)
        :x(x),y(y),z(z)
    {}

    Vector(const Vector& vector)
        :x(vector.x),y(vector.y),z(vector.z)
    {}

    template<class T2> explicit Vector(const Vector<T2,3>& vector)
        :x(T(vector.x)),y(T(vector.y)),z(T(vector.z))
    {}

    explicit Vector(const Vector<T,2>& vector)
        :x(vector.x),y(vector.y),z()
    {}

    template<class TVector,class TIndices>
    explicit Vector(const IndirectArray<TVector,TIndices>& v)
        :x(v[0]),y(v[1]),z(v[2])
    {
        static_assert(is_same<T,typename IndirectArray<TVector,TIndices>::Element>::value && IndirectArray<TVector,TIndices>::m==3,"");
    }

    explicit Vector(const Vector<T,2>& vector, const T& z)
      :x(vector.x),y(vector.y),z(z)
    {}

    template<int n>
    Vector(const Vector<T,n>& v1,const Vector<T,3-n>& v2)
    {
        for(int i=0;i<n;i++) (*this)(i)=v1(i);for(int i=n;i<3;i++) (*this)(i)=v2(i-n);
    }

    template<class TVector> typename EnableForVectorLike<T,3,TVector,Vector&>::type
    operator=(const TVector& v)
    {
        x=v[0];y=v[1];z=v[2];return *this;
    }

    Vector& operator=(const Vector& v)
    {
        x=v[0];y=v[1];z=v[2];return *this;
    }

    int size() const
    {return 3;}

    const T& operator[](const int i) const
    {assert(unsigned(i)<3);return *((const T*)(this)+i);}

    T& operator[](const int i)
    {assert(unsigned(i)<3);return *((T*)(this)+i);}

    T* data()
    {return &x;}

    const T* data() const
    {return &x;}

    template<class TIndices>
    IndirectArray<Vector,TIndices&> subset(const TIndices& indices)
    {return IndirectArray<Vector,TIndices&>(*this,indices);}

    template<class TIndices>
    IndirectArray<const Vector,TIndices&> subset(const TIndices& indices) const
    {return IndirectArray<const Vector,TIndices&>(*this,indices);}

    bool operator==(const Vector& v) const
    {return x==v.x && y==v.y && z==v.z;}

    bool operator!=(const Vector& v) const
    {return x!=v.x || y!=v.y || z!=v.z;}

    Vector operator-() const
    {return Vector(-x,-y,-z);}

    Vector& operator+=(const Vector& v)
    {x+=v.x;y+=v.y;z+=v.z;return *this;}

    Vector& operator-=(const Vector& v)
    {x-=v.x;y-=v.y;z-=v.z;return *this;}

    Vector& operator*=(const Vector& v)
    {x*=v.x;y*=v.y;z*=v.z;return *this;}

    Vector& operator+=(const T& a)
    {x+=a;y+=a;z+=a;return *this;}

    Vector& operator-=(const T& a)
    {x-=a;y-=a;z-=a;return *this;}

    Vector& operator*=(const T& a)
    {x*=a;y*=a;z*=a;return *this;}

    Vector& operator*=(const IntInverse<T> a)
    {x*=a;y*=a;z*=a;return *this;}

    Vector& operator/=(const T& a)
    {return *this*=inverse(a);}

    Vector& operator/=(const Vector& v)
    {x/=v.x;y/=v.y;z/=v.z;return *this;}

    Vector operator+(const Vector& v) const
    {return Vector(x+v.x,y+v.y,z+v.z);}

    Vector operator-(const Vector& v) const
    {return Vector(x-v.x,y-v.y,z-v.z);}

    Vector operator*(const Vector& v) const
    {return Vector(x*v.x,y*v.y,z*v.z);}

    Vector operator/(const Vector& v) const
    {return Vector(x/v.x,y/v.y,z/v.z);}

    Vector operator+(const T& a) const
    {return Vector(x+a,y+a,z+a);}

    Vector operator-(const T& a) const
    {return Vector(x-a,y-a,z-a);}

    Vector operator*(const T& a) const
    {return Vector(x*a,y*a,z*a);}

    Vector operator*(const IntInverse<T> a) const
    {return Vector(x*a,y*a,z*a);}

    Vector operator/(const T& a) const
    {return *this*inverse(a);}

    Vector operator&(const T& a) const
    {return Vector(x&a,y&a,z&a);}

    Vector operator>>(const int a) const
    {return Vector(x>>a, y>>a, z>>a);}

    Vector operator<<(const int a) const
    {return Vector(x<<a, y<<a, z<<a);}

    T sqr_magnitude() const
    {return x*x+y*y+z*z;}

    T magnitude() const
    {return sqrt(x*x+y*y+z*z);}

    T Lp_Norm(const T& p) const
    {return pow(pow(abs(x),p)+pow(abs(y),p)+pow(abs(z),p),1/p);}

    T L1_Norm() const
    {return abs(x)+abs(y)+abs(z);}

    T normalize()
    {T mag=magnitude();if(mag) *this*=1/mag;else *this=Vector(1,0,0);return mag;}

    Vector normalized() const // 6 mults, 2 adds, 1 div, 1 sqrt
    {T mag=magnitude();if(mag) return *this*(1/mag);else return Vector(1,0,0);}

    Vector orthogonal_vector() const
    {T abs_x=abs(x),abs_y=abs(y),abs_z=abs(z);
    if(abs_x<abs_y) return abs_x<abs_z?Vector(0,z,-y):Vector(y,-x,0);
    else return abs_y<abs_z?Vector(-z,0,x):Vector(y,-x,0);}

    Vector unit_orthogonal_vector() const // roughly 6 mults, 2 adds, 1 div, 1 sqrt
    {return orthogonal_vector().normalized();}

    T min() const
    {return geode::min(x,y,z);}

    T max() const
    {return geode::max(x,y,z);}

    T maxabs() const
    {return geode::maxabs(x,y,z);}

    int argmin() const
    {return geode::argmin(x,y,z);}

    int argmax() const
    {return geode::argmax(x,y,z);}

    int dominant_axis() const
    {return geode::argmax(abs(x), abs(y), abs(z));}

    bool elements_equal() const
    {return x==y && x==z;}

    static Vector componentwise_min(const Vector& v1,const Vector& v2)
    {return Vector(geode::min(v1.x,v2.x),geode::min(v1.y,v2.y),geode::min(v1.z,v2.z));}

    static Vector componentwise_max(const Vector& v1,const Vector& v2)
    {return Vector(geode::max(v1.x,v2.x),geode::max(v1.y,v2.y),geode::max(v1.z,v2.z));}

    Vector projected_on_unit_direction(const Vector& direction) const
    {return dot(*this,direction)*direction;}

    Vector projected(const Vector& direction) const // un-normalized direction
    {return dot(*this,direction)/direction.sqr_magnitude()*direction;}

    void project_on_unit_direction(const Vector& direction)
    {*this=dot(*this,direction)*direction;}

    void project(const Vector& direction) // un-normalized direction
    {*this=dot(*this,direction)/direction.sqr_magnitude()*direction;}

    Vector projected_orthogonal_to_unit_direction(const Vector& direction) const
    {return *this-dot(*this,direction)*direction;}

    void project_orthogonal_to_unit_direction(const Vector& direction)
    {*this-=dot(*this,direction)*direction;}

    T sum() const
    {return x+y+z;}

    T average() const
    {return T(1./3)*sum();}

    T product() const
    {return x*y*z;}

    const Vector& column_sum() const
    {return *this;}

    int number_true() const
    {STATIC_ASSERT_SAME(T,bool);return x+y+z;}

    static Vector axis_vector(const int axis)
    {Vector vec;vec[axis]=(T)1;return vec;}

    static Vector ones()
    {return Vector(1,1,1);}

    static Vector repeat(const T& constant)
    {return Vector(constant,constant,constant); }

    // shifts vector (wrapped) such that element a is first
    Vector<T,3> roll(const int a) {
      Vector<T,3> v;
      v.x = (*this)[(0+a) % 3];
      v.y = (*this)[(1+a) % 3];
      v.z = (*this)[(2+a) % 3];
      return v;
    }

    Vector<T,2> xy() const {
      return Vector<T,2>(x,y);
    }

    Vector<T,2> xz() const {
      return Vector<T,2>(x,z);
    }

    Vector<T,2> yz() const {
      return Vector<T,2>(y,z);
    }

    Vector<T,2> zx() const {
      return Vector<T,2>(z,x);
    }

    Vector<T,3> yxz() const {
        return Vector<T,3>(y,x,z);
    }

    Vector<T,3> xzy() const {
        return Vector<T,3>(x,z,y);
    }

    Vector<T,2> horizontal_vector() const
    {return Vector<T,2>(x,z);}

    void fill(const T& constant)
    {x=y=z=constant;}

    void get(T& element1,T& element2,T& element3) const
    {element1=x;element2=y;element3=z;}

    void set(const T& element1,const T& element2,const T& element3)
    {x=element1;y=element2;z=element3;}

    template<class TFunction>
    static auto map(const TFunction& f,const Vector& v) -> Vector<decltype(f(v.x)),3>
    {return Vector<decltype(f(v.x)),3>(f(v.x),f(v.y),f(v.z));}

    int find(const T& element) const
    {return x==element?0:y==element?1:z==element?2:-1;}

    bool contains(const T& element) const
    {return x==element || y==element || z==element;}

    template<class TArray>
    bool contains_all(const TArray& elements) const
    {STATIC_ASSERT_SAME(typename TArray::Element,T);
    for(int i=0;i<elements.size();i++) if(!contains(elements[i])) return false;
    return true;}

    template<class TArray>
    bool contains_any(const TArray& elements) const
    {STATIC_ASSERT_SAME(typename TArray::Element,T);
    for(int i=0;i<elements.size();i++) if(contains(elements[i])) return true;
    return false;}

    Vector<T,2> remove_index(const int index) const
    {assert(unsigned(index)<3);return Vector<T,2>(index>0?x:y,index<2?z:y);}

    Vector<T,4> insert(const T& element,const int index) const
    {Vector<T,4> r;r[index]=element;for(int i=0;i<3;i++) r[i+(i>=index)]=(*this)[i];return r;}

    Vector<T,4> append(const T& element) const
    {return Vector<T,4>(x,y,z,element);}

    template<int d2> Vector<T,3+d2> extend(const Vector<T,d2>& elements) const
    {Vector<T,3+d2> r;r[0]=x;r[1]=y;r[2]=z;for(int i=0;i<d2;i++) r[i+3]=elements[i];return r;}

    Vector<T,3> sorted() const
    {Vector<T,3> r(*this);small_sort(r.x,r.y,r.z);return r;}

    Vector reversed() const
    {return Vector(z,y,x);}

    template<int d1,int d2> Vector<T,d2-d1> slice() const
    {static_assert(0<=d1 && d1<=d2 && d2<=3,"");
    Vector<T,d2-d1> r;for(int i=d1;i<d2;i++) r[i-d1]=(*this)[i];return r;}

    template<int n> void split(Vector<T,n>& v1,Vector<T,3-n>& v2) const
    {for(int i=0;i<n;i++) v1(i)=(*this)(i);
    for(int i=n;i<3;i++) v2(i-n)=(*this)(i);}

    template<class TVector>
    void set_subvector(const int istart,const TVector& v)
    {for(int i=0;i<v.size();i++) (*this)[istart+i]=v[i];}

    template<class TVector>
    void add_subvector(const int istart,const TVector& v)
    {for(int i=0;i<v.size();i++) (*this)[istart+i]+=v[i];}

    template<class TVector>
    void get_subvector(const int istart,TVector& v) const
    {for(int i=0;i<v.size();i++) v[i]=(*this)[istart+i];}

    T* begin() // for stl
    {return &x;}

    const T* begin() const // for stl
    {return &x;}

    T* end() // for stl
    {return &z+1;}

    const T* end() const // for stl
    {return &z+1;}

    T& front() { return x; }
    const T& front() const { return x; }
    T& back() { return z; }
    const T& back() const { return z; }

//#####################################################################
};

//#####################################################################
// Miscellaneous free operators and functions
//#####################################################################
template<class T> inline Vector<T,3>
operator+(const typename Hide<T>::type& a,const Vector<T,3>& v)
{return Vector<T,3>(a+v.x,a+v.y,a+v.z);}

template<class T> inline Vector<T,3>
operator-(const typename Hide<T>::type& a,const Vector<T,3>& v)
{return Vector<T,3>(a-v.x,a-v.y,a-v.z);}

template<class T> inline Vector<T,3>
operator*(const typename Hide<T>::type& a,const Vector<T,3>& v)
{return Vector<T,3>(a*v.x,a*v.y,a*v.z);}

template<class T> inline Vector<T,3>
operator/(const typename Hide<T>::type& a,const Vector<T,3>& v)
{return Vector<T,3>(a/v.x,a/v.y,a/v.z);}

template<class T> inline Vector<T,3>
abs(const Vector<T,3>& v)
{return Vector<T,3>(abs(v.x),abs(v.y),abs(v.z));}

template<class T> inline Vector<T,3>
floor(const Vector<T,3>& v)
{return Vector<T,3>(floor(v.x),floor(v.y),floor(v.z));}

template<class T> inline Vector<T,3>
ceil(const Vector<T,3>& v)
{return Vector<T,3>(ceil(v.x),ceil(v.y),ceil(v.z));}

template<class T> inline Vector<T,3>
exp(const Vector<T,3>& v)
{return Vector<T,3>(exp(v.x),exp(v.y),exp(v.z));}

template<class T> inline Vector<T,3>
pow(const Vector<T,3>& v,const T a)
{return Vector<T,3>(pow(v.x,a),pow(v.y,a),pow(v.z,a));}

template<class T> inline Vector<T,3>
pow(const Vector<T,3>& v,const Vector<T,3>& a)
{return Vector<T,3>(pow(v.x,a.x),pow(v.y,a.y),pow(v.z,a.z));}

template<class T> inline Vector<T,3>
sin(const Vector<T,3>& v)
{return Vector<T,3>(sin(v.x),sin(v.y),sin(v.z));}

template<class T> inline Vector<T,3>
cos(const Vector<T,3>& v)
{return Vector<T,3>(cos(v.x),cos(v.y),cos(v.z));}

template<class T> inline Vector<T,3>
sqrt(const Vector<T,3>& v)
{return Vector<T,3>(sqrt(v.x),sqrt(v.y),sqrt(v.z));}

template<class T> inline Vector<T,3>
inverse(const Vector<T,3>& v)
{return Vector<T,3>(1/v.x,1/v.y,1/v.z);}

template<class T>
inline bool isnan(const Vector<T,3>& v)
{return isnan(v.x) || isnan(v.y) || isnan(v.z);}

template<class T> inline void
cyclic_shift(Vector<T,3>& v)
{cyclic_shift(v.x,v.y,v.z);}

template<class T> inline T
dot(const Vector<T,3>& v1,const Vector<T,3>& v2)
{return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;}

template<class T> inline Vector<T,3>
cross(const Vector<T,3>& v1,const Vector<T,3>& v2) // 6 mults, 3 adds
{return Vector<T,3>(v1.y*v2.z-v1.z*v2.y,v1.z*v2.x-v1.x*v2.z,v1.x*v2.y-v1.y*v2.x);}

template<class T> inline T angle_between(const Vector<T,3>& u, const Vector<T,3>& v) { // 0 .. pi
  return atan2(magnitude(cross(u,v)),dot(u,v));
}

template<class T> inline T
det(const Vector<T,3>& u,const Vector<T,3>& v,const Vector<T,3>& w)
{return dot(u,cross(v,w));}

template<class T> inline Matrix<T,3>
cross_product_matrix(const Vector<T,3>& v)
{return Matrix<T,3>(0,v.z,-v.y,-v.z,0,v.x,v.y,-v.x,0);}

template<class T> inline bool
isfinite(const Vector<T,3>& v)
{return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);}

template<class T> inline bool all_greater(const Vector<T,3>& v0, const Vector<T,3>& v1) {
  return v0.x>v1.x && v0.y>v1.y && v0.z>v1.z;
}

template<class T> inline bool all_less(const Vector<T,3>& v0, const Vector<T,3>& v1) {
  return v0.x<v1.x && v0.y<v1.y && v0.z<v1.z;
}

template<class T> inline bool all_greater_equal(const Vector<T,3>& v0, const Vector<T,3>& v1) {
  return v0.x>=v1.x && v0.y>=v1.y && v0.z>=v1.z;
}

template<class T> inline bool all_less_equal(const Vector<T,3>& v0, const Vector<T,3>& v1) {
  return v0.x<=v1.x && v0.y<=v1.y && v0.z<=v1.z;
}

//#####################################################################
// Functions clamp, clamp_min, clamp_max, in_bounds
//#####################################################################
template<class T> inline Vector<T,3>
clamp(const Vector<T,3>& v,const Vector<T,3>& vmin,const Vector<T,3>& vmax)
{return Vector<T,3>(clamp(v.x,vmin.x,vmax.x),clamp(v.y,vmin.y,vmax.y),clamp(v.z,vmin.z,vmax.z));}

template<class T> inline Vector<T,3>
clamp(const Vector<T,3>& v,T min,T max)
{return Vector<T,3>(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max));}

template<class T> inline Vector<T,3>
clamp_min(const Vector<T,3>& v,const Vector<T,3>& vmin)
{return Vector<T,3>(clamp_min(v.x,vmin.x),clamp_min(v.y,vmin.y),clamp_min(v.z,vmin.z));}

template<class T> inline Vector<T,3>
clamp_min(const Vector<T,3>& v,const T& min)
{return Vector<T,3>(clamp_min(v.x,min),clamp_min(v.y,min),clamp_min(v.z,min));}

template<class T> inline Vector<T,3>
clamp_max(const Vector<T,3>& v,const Vector<T,3>& vmax)
{return Vector<T,3>(clamp_max(v.x,vmax.x),clamp_max(v.y,vmax.y),clamp_max(v.z,vmax.z));}

template<class T> inline Vector<T,3>
clamp_max(const Vector<T,3>& v,const T& max)
{return Vector<T,3>(clamp_max(v.x,max),clamp_max(v.y,max),clamp_max(v.z,max));}

template<class T> inline bool
in_bounds(const Vector<T,3>& v,const Vector<T,3>& vmin,const Vector<T,3>& vmax)
{return in_bounds(v.x,vmin.x,vmax.x) && in_bounds(v.y,vmin.y,vmax.y) && in_bounds(v.z,vmin.z,vmax.z);}

template<class T> inline Vector<T,3>
wrap(const Vector<T,3>& v,const Vector<T,3>& vmin,const Vector<T,3>& vmax)
{return Vector<T,3>(wrap(v.x,vmin.x,vmax.x),wrap(v.y,vmin.y,vmax.y),wrap(v.z,vmin.z,vmax.z));}

}
