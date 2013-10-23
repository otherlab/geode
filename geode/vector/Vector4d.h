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
#include <cmath>
namespace geode {

using ::std::exp;
using ::std::sin;
using ::std::cos;
using ::std::pow;

template<class TArray,class TIndices> class IndirectArray;

template<class T>
class Vector<T,4>
{
    struct Unusable{};
public:
    template<class T2> struct Rebind{typedef Vector<T2,4> type;};
    typedef typename mpl::if_<IsScalar<T>,T,Unusable>::type Scalar;
    typedef T Element;
    typedef T value_type; // for stl
    typedef T* iterator; // for stl
    typedef const T* const_iterator; // for stl
    template<class V> struct result;
    template<class V> struct result<V(int)>:mpl::if_<boost::is_const<V>,const T&,T&>{};
    enum Workaround1 {dimension=4};
    enum Workaround2 {m=4};
    static const bool is_const=false;

  T x,y,z,w;

    Vector()
      :x(),y(),z(),w()
    {
        BOOST_STATIC_ASSERT(sizeof(Vector)==4*sizeof(T));
    }

  Vector(const T& x,const T& y,const T& z,const T& w)
    :x(x),y(y),z(z),w(w)
    {}

    Vector(const Vector& vector)
      :x(vector.x),y(vector.y),z(vector.z),w(vector.w)
    {}

    template<class T2> explicit Vector(const Vector<T2,4>& vector)
      :x(T(vector.x)),y(T(vector.y)),z(T(vector.z)),w(T(vector.w))
    {}

    explicit Vector(const Vector<T,2>& vector)
      :x(vector.x),y(vector.y),z(),w()
    {}

    explicit Vector(const Vector<T,3>& vector)
      :x(vector.x),y(vector.y),z(vector.z),w()
    {}

    explicit Vector(const Vector<T,3>& vector, const T& w)
      :x(vector.x),y(vector.y),z(vector.z),w(w)
    {}

    template<class TVector>
    explicit Vector(const TVector& v, typename EnableForVectorLike<T,4,TVector,Unusable>::type=Unusable())
      :x(v[0]),y(v[1]),z(v[2]),w(v[3])
    {}

    template<int n>
    Vector(const Vector<T,n>& v1,const Vector<T,4-n>& v2)
    {
        for(int i=0;i<n;i++) (*this)[i]=v1[i];for(int i=n;i<4;i++) (*this)[i]=v2[i-n];
    }

    template<class TVector> typename EnableForVectorLike<T,4,TVector,Vector&>::type
    operator=(const TVector& v)
    {
      x=v[0];y=v[1];z=v[2];w=v[3];return *this;
    }

    Vector& operator=(const Vector& v)
    {
      x=v[0];y=v[1];z=v[2];w=v[3];return *this;
    }

    int size() const
    {return 4;}

    const T& operator[](const int i) const
    {assert(unsigned(i)<4);return *((const T*)(this)+i);}

    T& operator[](const int i)
    {assert(unsigned(i)<4);return *((T*)(this)+i);}

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
    {return x==v.x && y==v.y && z==v.z && w==v.w;}

    bool operator!=(const Vector& v) const
    {return x!=v.x || y!=v.y || z!=v.z || w!=v.w;}

  Vector operator-() const
  {return Vector(-x,-y,-z,-w);}

  Vector& operator+=(const Vector& v)
  {x+=v.x;y+=v.y;z+=v.z;w+=v.w;return *this;}

  Vector& operator-=(const Vector& v)
  {x-=v.x;y-=v.y;z-=v.z;w-=v.w;return *this;}

    Vector& operator*=(const Vector& v)
  {x*=v.x;y*=v.y;z*=v.z;w*=v.w;return *this;}

    Vector& operator+=(const T& a)
  {x+=a;y+=a;z+=a;w+=a;return *this;}

    Vector& operator-=(const T& a)
  {x-=a;y-=a;z-=a;w-=a;return *this;}

    Vector& operator*=(const T& a)
  {x*=a;y*=a;z*=a;w*=a;return *this;}

    Vector& operator*=(const IntInverse<T> a)
  {x*=a;y*=a;z*=a;w*=a;return *this;}

    Vector& operator/=(const T& a)
    {return *this*=inverse(a);}

    Vector& operator/=(const Vector& v)
  {x/=v.x;y/=v.y;z/=v.z;w/=v.w;return *this;}

    Vector operator+(const Vector& v) const
  {return Vector(x+v.x,y+v.y,z+v.z,w+v.w);}

    Vector operator-(const Vector& v) const
  {return Vector(x-v.x,y-v.y,z-v.z,w-v.w);}

    Vector operator*(const Vector& v) const
  {return Vector(x*v.x,y*v.y,z*v.z,w*v.w);}

    Vector operator/(const Vector& v) const
  {return Vector(x/v.x,y/v.y,z/v.z,w/v.w);}

    Vector operator+(const T& a) const
  {return Vector(x+a,y+a,z+a,w+a);}

    Vector operator-(const T& a) const
  {return Vector(x-a,y-a,z-a,w-a);}

    Vector operator*(const T& a) const
  {return Vector(x*a,y*a,z*a,w*a);}

    Vector operator*(const IntInverse<T> a) const
  {return Vector(x*a,y*a,z*a,w*a);}

    Vector operator/(const T& a) const
    {return *this*inverse(a);}

    Vector operator&(const T& a) const
    {return Vector(x&a,y&a,z&a,w&a);}

    Vector operator^(const Vector& v) const
    {return Vector(x^v.x,y^v.y,z^v.z,w^v.w);}

    T sqr_magnitude() const
    {return x*x+y*y+z*z+w*w;}

    T magnitude() const
    {return sqrt(x*x+y*y+z*z+w*w);}

    T Lp_Norm(const T& p) const
  {return pow(pow(abs(x),p)+pow(abs(y),p)+pow(abs(z),p)+pow(abs(w),p),1/p);}

    T L1_Norm() const
  {return abs(x)+abs(y)+abs(z)+abs(w);}

    T normalize()
    {T mag=magnitude();if(mag) *this*=1/mag;else *this=Vector(1,0,0);return mag;}

    Vector normalized() const // 6 mults, 2 adds, 1 div, 1 sqrt
  {T mag=magnitude();if(mag) return *this*(1/mag);else return Vector(1,0,0,0);}

    T min() const
  {return geode::min(x,y,z,w);}

    T max() const
  {return geode::max(x,y,z,w);}

    T maxabs() const
  {return geode::maxabs(x,y,z,w);}

    int argmin() const
  {return geode::argmin(x,y,z,w);}

    int argmax() const
  {return geode::argmax(x,y,z,w);}

    bool elements_equal() const
  {return x==y && x==z && x==w;}

    static Vector componentwise_min(const Vector& v1,const Vector& v2)
    {return Vector(geode::min(v1.x,v2.x),geode::min(v1.y,v2.y),geode::min(v1.z,v2.z),geode::min(v1.w,v2.w));}

    static Vector componentwise_max(const Vector& v1,const Vector& v2)
    {return Vector(geode::max(v1.x,v2.x),geode::max(v1.y,v2.y),geode::max(v1.z,v2.z),geode::max(v1.w,v2.zw));}

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
    {return x+y+z+w;}

    T average() const
    {return T(1./4)*sum();}

    T product() const
    {return x*y*z*w;}

    const Vector& column_sum() const
    {return *this;}

    int number_true() const
    {STATIC_ASSERT_SAME(T,bool);return x+y+z+w;}
  
    static Vector axis_vector(const int axis)
    {Vector vec;vec[axis]=(T)1;return vec;}

    static Vector ones()
    {return Vector(1,1,1,1);}

    Vector<T,2> horizontal_vector() const
    {return Vector<T,2>(x,z);}

    void fill(const T& constant)
    {x=y=z=w=constant;}

    void get(T& element1,T& element2,T& element3,T& element4) const
    {element1=x;element2=y;element3=z;element4=w;}

    void set(const T& element1,const T& element2,const T& element3, const T& element4)
    {x=element1;y=element2;z=element3;w=element4;}

    template<class TFunction>
    static Vector map(const TFunction& f,const Vector& v)
    {return Vector(f(v.x),f(v.y),f(v.z),f(v.w));}

    int find(const T& element) const
    {return x==element?0:y==element?1:z==element?2:w==element?3:-1;}

    bool contains(const T& element) const
    {return x==element || y==element || z==element || w==element;}

    template<class TArray>
    bool contains_all(const TArray& elements) const
    {STATIC_ASSERT_SAME(typename TArray::Element,T);
    for(int i=0;i<elements.size();i++) if(!contains(elements(i))) return false;
    return true;}

    template<class TArray>
    bool contains_any(const TArray& elements) const
    {STATIC_ASSERT_SAME(typename TArray::Element,T);
    for(int i=0;i<elements.size();i++) if(contains(elements(i))) return true;
    return false;}

    Vector<T,3> remove_index(const int index) const
    {assert(unsigned(index)<4);return Vector<T,3>(index>0?x:y,index>1?y:z,index>2?z:w);}

    Vector<T,2> xy() const
    {return Vector<T,2>(x,y);}

    Vector<T,3> xyz() const
    {return Vector<T,3>(x,y,z);}

    Vector<T,4> insert(const T& element,const int index) const
    {Vector<T,4> r;r[index]=element;for(int i=0;i<4;i++) r[i+(i>=index)]=(*this)[i];return r;}

    Vector<T,4> append(const T& element) const
    {return Vector<T,5>(x,y,z,w,element);}

    template<int d2> Vector<T,4+d2> extend(const Vector<T,d2>& elements) const
    {Vector<T,4+d2> r;r[0]=x;r[1]=y;r[2]=z;r[3]=w;for(int i=0;i<d2;i++) r[i+4]=elements[i];return r;}

    Vector<T,4> sorted() const
    {Vector<T,4> r(*this);small_sort(r.x,r.y,r.z,r.w);return r;}

    Vector reversed() const
    {return Vector(w,z,y,x);}

    template<int d1,int d2> Vector<T,d2-d1> slice() const
    {BOOST_STATIC_ASSERT((mpl::and_<mpl::less_equal<mpl::int_<0>,mpl::int_<d1> >,mpl::less_equal<mpl::int_<d2>,mpl::int_<4> > >::value));
    Vector<T,d2-d1> r;for(int i=d1;i<d2;i++) r[i-d1]=(*this)[i];return r;}

    template<int n> void split(Vector<T,n>& v1,Vector<T,4-n>& v2) const
    {for(int i=0;i<n;i++) v1(i)=(*this)(i);
    for(int i=n;i<4;i++) v2(i-n)=(*this)(i);}

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
    {return &w+1;}

    const T* end() const // for stl
    {return &w+1;}

    T& front() { return x; }
    const T& front() const { return x; }
    T& back() { return w; }
    const T& back() const { return w; }

//#####################################################################
};

//#####################################################################
// Miscellaneous free operators and functions
//#####################################################################
template<class T> inline Vector<T,4>
operator+(const typename Hide<T>::type& a,const Vector<T,4>& v)
  {return Vector<T,4>(a+v.x,a+v.y,a+v.z,a+v.w);}

template<class T> inline Vector<T,4>
operator-(const typename Hide<T>::type& a,const Vector<T,4>& v)
  {return Vector<T,4>(a-v.x,a-v.y,a-v.z,a-v.w);}

template<class T> inline Vector<T,4>
operator*(const typename Hide<T>::type& a,const Vector<T,4>& v)
  {return Vector<T,4>(a*v.x,a*v.y,a*v.z,a*v.w);}

template<class T> inline Vector<T,4>
operator/(const typename Hide<T>::type& a,const Vector<T,4>& v)
  {return Vector<T,4>(a/v.x,a/v.y,a/v.z,a/v.w);}

template<class T> inline Vector<T,4>
abs(const Vector<T,4>& v)
  {return Vector<T,4>(abs(v.x),abs(v.y),abs(v.z),abs(v.w));}

template<class T> inline Vector<T,4>
floor(const Vector<T,4>& v)
  {return Vector<T,4>(floor(v.x),floor(v.y),floor(v.z),floor(v.w));}

template<class T> inline Vector<T,4>
ceil(const Vector<T,4>& v)
  {return Vector<T,4>(ceil(v.x),ceil(v.y),ceil(v.z),ceil(v.w));}

template<class T> inline Vector<T,4>
exp(const Vector<T,4>& v)
  {return Vector<T,4>(exp(v.x),exp(v.y),exp(v.z),exp(v.w));}

template<class T> inline Vector<T,4>
pow(const Vector<T,4>& v,const T a)
  {return Vector<T,4>(pow(v.x,a),pow(v.y,a),pow(v.z,a),pow(v.w,a));}

template<class T> inline Vector<T,4>
pow(const Vector<T,4>& v,const Vector<T,4>& a)
  {return Vector<T,4>(pow(v.x,a.x),pow(v.y,a.y),pow(v.z,a.z),pow(v.w,a.w));}

template<class T> inline Vector<T,4>
sin(const Vector<T,4>& v)
  {return Vector<T,4>(sin(v.x),sin(v.y),sin(v.z),sin(v.w));}

template<class T> inline Vector<T,4>
cos(const Vector<T,4>& v)
  {return Vector<T,4>(cos(v.x),cos(v.y),cos(v.z),cos(v.w));}

template<class T> inline Vector<T,4>
sqrt(const Vector<T,4>& v)
  {return Vector<T,4>(sqrt(v.x),sqrt(v.y),sqrt(v.z),sqrt(v.w));}

template<class T> inline Vector<T,4>
inverse(const Vector<T,4>& v)
  {return Vector<T,4>(1/v.x,1/v.y,1/v.z,1/v.w);}

template<class T>
inline bool isnan(const Vector<T,4>& v)
  {return isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w);}

template<class T> inline void
cyclic_shift(Vector<T,4>& v)
  {cyclic_shift(v.x,v.y,v.z,v.w);}

template<class T> inline T
dot(const Vector<T,4>& v1,const Vector<T,4>& v2)
{return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z+v1.w*v2.w;}

template<class T> inline bool
isfinite(const Vector<T,4>& v)
{return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(v.w);}

template<class T> inline bool all_greater(const Vector<T,4>& v0, const Vector<T,4>& v1) {
  return v0.x>v1.x && v0.y>v1.y && v0.z>v1.z && v0.w>v1.w;
}

template<class T> inline bool all_less(const Vector<T,4>& v0, const Vector<T,4>& v1) {
  return v0.x<v1.x && v0.y<v1.y && v0.z<v1.z && v0.w<v1.w;
}

template<class T> inline bool all_greater_equal(const Vector<T,4>& v0, const Vector<T,4>& v1) {
  return v0.x>=v1.x && v0.y>=v1.y && v0.z>=v1.z && v0.w>=v1.w;
}

template<class T> inline bool all_less_equal(const Vector<T,4>& v0, const Vector<T,4>& v1) {
  return v0.x<=v1.x && v0.y<=v1.y && v0.z<=v1.z && v0.w<=v1.w;
}

//#####################################################################
// Functions clamp, clamp_min, clamp_max, in_bounds
//#####################################################################
template<class T> inline Vector<T,4>
clamp(const Vector<T,4>& v,const Vector<T,4>& vmin,const Vector<T,4>& vmax)
  {return Vector<T,4>(clamp(v.x,vmin.x,vmax.x),clamp(v.y,vmin.y,vmax.y),clamp(v.z,vmin.z,vmax.z),clamp(v.z,vmin.z,vmax.z));}

template<class T> inline Vector<T,4>
clamp(const Vector<T,4>& v,T min,T max)
  {return Vector<T,4>(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max));}

template<class T> inline Vector<T,4>
clamp_min(const Vector<T,4>& v,const Vector<T,4>& vmin)
  {return Vector<T,4>(clamp_min(v.x,vmin.x),clamp_min(v.y,vmin.y),clamp_min(v.z,vmin.z),clamp_min(v.w,vmin.w));}

template<class T> inline Vector<T,4>
clamp_min(const Vector<T,4>& v,const T& min)
  {return Vector<T,4>(clamp_min(v.x,min),clamp_min(v.y,min),clamp_min(v.z,min),clamp_min(v.w,min));}

template<class T> inline Vector<T,4>
clamp_max(const Vector<T,4>& v,const Vector<T,4>& vmax)
  {return Vector<T,4>(clamp_max(v.x,vmax.x),clamp_max(v.y,vmax.y),clamp_max(v.z,vmax.z),clamp_max(v.w,vmax.w));}

template<class T> inline Vector<T,4>
clamp_max(const Vector<T,4>& v,const T& max)
  {return Vector<T,4>(clamp_max(v.x,max),clamp_max(v.y,max),clamp_max(v.z,max),clamp_max(v.w,max));}

template<class T> inline bool
in_bounds(const Vector<T,4>& v,const Vector<T,4>& vmin,const Vector<T,4>& vmax)
{return in_bounds(v.x,vmin.x,vmax.x) && in_bounds(v.y,vmin.y,vmax.y) && in_bounds(v.z,vmin.z,vmax.z) && in_bounds(v.w,vmin.w,vmax.w);}

template<class T> inline Vector<T,4>
wrap(const Vector<T,4>& v,const Vector<T,4>& vmin,const Vector<T,4>& vmax)
  {return Vector<T,4>(wrap(v.x,vmin.x,vmax.x),wrap(v.y,vmin.y,vmax.y),wrap(v.z,vmin.z,vmax.z),wrap(v.w,vmin.w,vmax.w));}

}
