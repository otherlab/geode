//#####################################################################
// Class Vector2d
//#####################################################################
#pragma once

#include <other/core/vector/Vector1d.h>
#include <other/core/vector/complex.h>
#include <other/core/math/inverse.h>
#include <other/core/math/clamp.h>
#include <other/core/math/small_sort.h>
#include <other/core/math/argmax.h>
#include <other/core/math/argmin.h>
#include <other/core/math/isfinite.h>
#include <other/core/math/isnan.h>
#include <other/core/math/max.h>
#include <other/core/math/maxabs.h>
#include <other/core/math/min.h>
#include <other/core/math/minabs.h>
#include <other/core/math/sqr.h>
#include <other/core/math/wrap.h>
#include <cmath>
namespace other {

using ::std::floor;
using ::std::ceil;
using ::std::sin;
using ::std::cos;
using ::std::sqrt;

template<class T>
class Vector<T,2>
{
    struct Unusable{};
public:
    template<class T2> struct Rebind{typedef Vector<T2,2> type;};
    typedef typename mpl::if_<IsScalar<T>,T,Unusable>::type Scalar;
    typedef T Element;
    typedef T value_type; // for stl
    typedef T* iterator; // for stl
    typedef const T* const_iterator; // for stl
    enum Workaround1 {dimension=2};
    enum Workaround2 {m=2};
    static const bool is_const=false;

    T x,y;

    Vector()
        :x(),y()
    {
        BOOST_STATIC_ASSERT(sizeof(Vector)==2*sizeof(T));
    }

    Vector(const T& x,const T& y)
        :x(x),y(y)
    {}

    Vector(const Vector& vector)
        :x(vector.x),y(vector.y)
    {}

    template<class T2> explicit Vector(const Vector<T2,2>& vector)
        :x((T)vector.x),y((T)vector.y)
    {}

    template<class T2> explicit Vector(const Vector<T2,1>& vector)
        :x((T)vector.x),y(T())
    {}

    template<class TVector>
    explicit Vector(const TVector& v)
        :x(v[0]),y(v[1])
    {
        BOOST_STATIC_ASSERT((boost::is_same<T,typename TVector::Element>::value && TVector::m==2));
    }

    explicit Vector(const complex<T>& c)
      : x(c.real()), y(c.imag()) {}

    template<class TVector> typename EnableForVectorLike<T,2,TVector,Vector&>::type
    operator=(const TVector& v)
    {
        x=v[0];y=v[1];return *this;
    }

    Vector& operator=(const Vector& v)
    {
        x=v[0];y=v[1];return *this;
    }

    int size() const
    {return 2;}

    const T& operator[](const int i) const
    {assert(unsigned(i)<2);return *((const T*)(this)+i);}

    T& operator[](const int i)
    {assert(unsigned(i)<2);return *((T*)(this)+i);}

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
    {return x==v.x && y==v.y;}

    bool operator!=(const Vector& v) const
    {return x!=v.x || y!=v.y;}

    Vector operator-() const
    {return Vector(-x,-y);}

    Vector& operator+=(const Vector& v)
    {x+=v.x;y+=v.y;return *this;}

    Vector& operator-=(const Vector& v)
    {x-=v.x;y-=v.y;return *this;}

    Vector& operator*=(const Vector& v)
    {x*=v.x;y*=v.y;return *this;}

    Vector& operator+=(const T& a)
    {x+=a;y+=a;return *this;}

    Vector& operator-=(const T& a)
    {x-=a;y-=a;return *this;}

    Vector& operator*=(const T& a)
    {x*=a;y*=a;return *this;}

    Vector& operator*=(const IntInverse<T> a)
    {x*=a;y*=a;return *this;}

    Vector& operator/=(const T& a)
    {return *this*=inverse(a);}

    Vector& operator/=(const Vector& v)
    {x/=v.x;y/=v.y;return *this;}

    Vector operator+(const Vector& v) const
    {return Vector(x+v.x,y+v.y);}

    Vector operator-(const Vector& v) const
    {return Vector(x-v.x,y-v.y);}

    Vector operator*(const Vector& v) const
    {return Vector(x*v.x,y*v.y);}

    Vector operator/(const Vector& v) const
    {return Vector(x/v.x,y/v.y);}

    Vector operator+(const T& a) const
    {return Vector(x+a,y+a);}

    Vector operator-(const T& a) const
    {return Vector(x-a,y-a);}

    Vector operator*(const T& a) const
    {return Vector(x*a,y*a);}

    Vector operator*(const IntInverse<T> a) const
    {return Vector(x*a,y*a);}

    Vector operator/(const T& a) const
    {return *this*inverse(a);}

    Vector operator&(const T& a) const
    {return Vector(x&a,y&a);}

    Vector& operator|=(const Vector& v)
    {x|=v.x;y|=v.y;return *this;}
    
    Vector operator>>(const int a) const
    {return Vector(x>>a, y>>a);}

    Vector operator<<(const int a) const
    {return Vector(x<<a, y<<a);}

    T sqr_magnitude() const
    {return sqr(x)+sqr(y);}

    T magnitude() const
    {return sqrt(sqr(x)+sqr(y));}

    T Lp_Norm(const T& p) const
    {return pow(pow(abs(x),p)+pow(abs(y),p),1/p);}

    T L1_Norm() const
    {return abs(x)+abs(y);}

    T normalize()
    {T mag=magnitude();if(mag) *this*=1/mag;else *this=Vector(1,0);return mag;}

    Vector normalized() const
    {T mag=magnitude();if(mag) return *this*(1/mag);else return Vector(1,0);}

    Vector orthogonal_vector() const
    {return Vector(-y,x);}

    Vector unit_orthogonal_vector() const
    {return orthogonal_vector().normalized();}

    T min() const
    {return other::min(x,y);}

    T max() const
    {return other::max(x,y);}

    T maxabs() const
    {return other::maxabs(x,y);}

    int argmin() const
    {return other::argmin(x,y);}

    int argmax() const
    {return other::argmax(x,y);}

    bool elements_equal() const
    {return x==y;}

    static Vector componentwise_min(const Vector& v1,const Vector& v2)
    {return Vector(other::min(v1.x,v2.x),other::min(v1.y,v2.y));}

    static Vector componentwise_max(const Vector& v1,const Vector& v2)
    {return Vector(other::max(v1.x,v2.x),other::max(v1.y,v2.y));}

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

    T sum() const
    {return x+y;}

    T average() const
    {return (T).5*sum();}

    T product() const
    {return x*y;}
  
    Vector swap() const {
      return Vector(y,x);
    }

    const Vector& column_sum() const
    {return *this;}

    int number_true() const
    {STATIC_ASSERT_SAME(T,bool);return x+y;}

    static Vector axis_vector(const int axis)
    {Vector vec;vec[axis]=(T)1;return vec;}

    static Vector ones()
    {return Vector(1,1);}

    Vector<T,1> horizontal_vector() const
    {return Vector<T,1>(x);}

    Vector<T,1> vertical_vector() const
    {return Vector<T,1>(y);}

    void fill(const T& constant)
    {x=y=constant;}

    void get(T& element1,T& element2) const
    {element1=x;element2=y;}

    void set(const T& element1,const T& element2)
    {x=element1;y=element2;}

    template<class TFunction>
    static Vector map(const TFunction& f,const Vector& v)
    {return Vector(f(v.x),f(v.y));}

    int find(const T& element) const
    {return x==element?0:y==element?1:-1;}

    bool contains(const T& element) const
    {return x==element || y==element;}

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

    Vector<T,1> remove_index(const int index) const
    {assert(unsigned(index)<2);return Vector<T,1>((*this)[1-index]);}

    Vector<T,3> insert(const T& element,const int index) const
    {Vector<T,3> r;r[index]=element;for(int i=0;i<2;i++) r[i+(i>=index)]=(*this)[i];return r;}

    Vector<T,3> append(const T& element) const
    {return Vector<T,3>(x,y,element);}

    template<int d2> Vector<T,2+d2> extend(const Vector<T,d2>& elements) const
    {Vector<T,2+d2> r;r[0]=x;r[1]=y;for(int i=0;i<d2;i++) r[i+2]=elements[i];return r;}

    Vector<T,2> sorted() const
    {Vector<T,2> r(*this);small_sort(r.x,r.y);return r;}

    Vector reversed() const
    {return Vector(y,x);}

    template<int d1,int d2> Vector<int,d2-d1> slice() const
    {BOOST_STATIC_ASSERT((mpl::and_<mpl::less_equal<mpl::int_<0>,mpl::int_<d1> >,mpl::less_equal<mpl::int_<d2>,mpl::int_<2> > >::value));
    Vector<T,d2-d1> r;for(int i=d1;i<d2;i++) r[i-d1]=(*this)[i];return r;}

    template<int n> void split(Vector<T,n>& v1,Vector<T,2-n>& v2) const
    {for(int i=0;i<n;i++) v1(i)=(*this)(i);
    for(int i=n;i<2;i++) v2(i-n)=(*this)(i);}

    template<class TVector>
    void set_subvector(const int istart,const TVector& v)
    {for(int i=0;i<v.size();i++) (*this)(istart+i)=v[i];}

    template<class TVector>
    void add_subvector(const int istart,const TVector& v)
    {for(int i=0;i<v.size();i++) (*this)(istart+i)+=v[i];}

    template<class TVector>
    void get_subvector(const int istart,TVector& v) const
    {for(int i=0;i<v.size();i++) v[i]=(*this)(istart+i);}

    std::complex<T> complex() const {
      return std::complex<T>(x,y);
    }

    T* begin() // for stl
    {return &x;}

    const T* begin() const // for stl
    {return &x;}

    T* end() // for stl
    {return &y+1;}

    const T* end() const // for stl
    {return &y+1;}

    T& front() { return x; }
    const T& front() const { return x; }
    T& back() { return y; }
    const T& back() const { return y; }
//#####################################################################
};

//#####################################################################
// Miscellaneous free operators and functions
//#####################################################################
template<class T> inline Vector<T,2>
operator+(const typename Hide<T>::type& a,const Vector<T,2>& v)
{return Vector<T,2>(a+v.x,a+v.y);}

template<class T> inline Vector<T,2>
operator-(const typename Hide<T>::type& a,const Vector<T,2>& v)
{return Vector<T,2>(a-v.x,a-v.y);}

template<class T> inline Vector<T,2>
operator*(const typename Hide<T>::type& a,const Vector<T,2>& v)
{return Vector<T,2>(a*v.x,a*v.y);}

template<class T> inline Vector<T,2>
operator/(const typename Hide<T>::type& a,const Vector<T,2>& v)
{return Vector<T,2>(a/v.x,a/v.y);}

template<class T> inline Vector<T,2>
abs(const Vector<T,2>& v)
{return Vector<T,2>(abs(v.x),abs(v.y));}

template<class T> inline Vector<T,2>
floor(const Vector<T,2>& v)
{return Vector<T,2>(floor(v.x),floor(v.y));}

template<class T> inline Vector<T,2>
ceil(const Vector<T,2>& v)
{return Vector<T,2>(ceil(v.x),ceil(v.y));}

template<class T> inline Vector<T,2>
exp(const Vector<T,2>& v)
{return Vector<T,2>(exp(v.x),exp(v.y));}

template<class T> inline Vector<T,2>
sin(const Vector<T,2>& v)
{return Vector<T,2>(sin(v.x),sin(v.y));}

template<class T> inline Vector<T,2>
cos(const Vector<T,2>& v)
{return Vector<T,2>(cos(v.x),cos(v.y));}

template<class T> inline Vector<T,2>
sqrt(const Vector<T,2>& v)
{return Vector<T,2>(sqrt(v.x),sqrt(v.y));}

template<class T> inline Vector<T,2>
inverse(const Vector<T,2>& v)
{return Vector<T,2>(1/v.x,1/v.y);}

template<class T>
inline bool isnan(const Vector<T,2>& v)
{return isnan(v.x) || isnan(v.y);}

template<class T> inline void
cyclic_shift(Vector<T,2>& v)
{swap(v.x,v.y);}

template<class T> inline T
dot(const Vector<T,2>& v1,const Vector<T,2>& v2)
{return v1.x*v2.x+v1.y*v2.y;}

template<class T> inline T
cross(const Vector<T,2>& v1,const Vector<T,2>& v2)
{return v1.x*v2.y-v1.y*v2.x;}

template<class T> inline Vector<T,2>
cross(const Vector<T,2>& v1,const T& v2) // v2 is out of plane
{return Vector<T,2>(v1.y*v2,-v1.x*v2);}

template<class T> inline Vector<T,2>
cross(const T& v1,const Vector<T,2>& v2) // v1 is out of plane
{return Vector<T,2>(-v1*v2.y,v1*v2.x);}

template<class T> inline T
angle(const Vector<T,2>& v)
{return atan2(v.y,v.x);}
  
template<class T> inline Vector<T,2>
polar(const T& a) 
{ return Vector<T,2>(cos(a), sin(a)); }

template<class T> inline T angle_between(const Vector<T,2>& u, const Vector<T,2>& v) {
  return atan2(cross(u,v),dot(u,v));
}

template<class T> inline bool
isfinite(const Vector<T,2>& v)
{return isfinite(v.x) && isfinite(v.y);}

template<class T> inline bool all_greater(const Vector<T,2>& v0, const Vector<T,2>& v1) {
  return v0.x>v1.x && v0.y>v1.y;
}

template<class T> inline bool all_less(const Vector<T,2>& v0, const Vector<T,2>& v1) {
  return v0.x<v1.x && v0.y<v1.y;
}

template<class T> inline bool all_greater_equal(const Vector<T,2>& v0, const Vector<T,2>& v1) {
  return v0.x>=v1.x && v0.y>=v1.y;
}

template<class T> inline bool all_less_equal(const Vector<T,2>& v0, const Vector<T,2>& v1) {
  return v0.x<=v1.x && v0.y<=v1.y;
}

template<class T> inline Matrix<T,1,2> cross_product_matrix(const Vector<T,2>& v) {
  Matrix<T,1,2> result;result(0,0)=-v.y;result(0,1)=v.x;return result;
}

template<class T> inline Vector<T,2> rotate_right_90(const Vector<T,2>& v) {
  return Vector<T,2>(v.y,-v.x);
}

template<class T> inline Vector<T,2> rotate_left_90(const Vector<T,2>& v) {
  return Vector<T,2>(-v.y,v.x);
}

template<class T> inline Vector<T,2> rotate_left_90_times(const Vector<T,2>& v, const int n) {
  const Vector<T,2> r = n&2?-v:v;
  return n&1?rotate_left_90(r):r;
}

template<class T> inline Vector<T,2> perpendicular(const Vector<T,2>& v) {
  return Vector<T,2>(-v.y,v.x);
}

//#####################################################################
// Functions clamp, clamp_min, clamp_max, in_bounds
//#####################################################################
template<class T> inline Vector<T,2>
clamp(const Vector<T,2>& v,const Vector<T,2>& vmin,const Vector<T,2>& vmax)
{return Vector<T,2>(clamp(v.x,vmin.x,vmax.x),clamp(v.y,vmin.y,vmax.y));}

template<class T> inline Vector<T,2>
clamp(const Vector<T,2>& v,T min,T max)
{return Vector<T,2>(clamp(v.x,min,max),clamp(v.y,min,max));}

template<class T> inline Vector<T,2>
clamp_min(const Vector<T,2>& v,const Vector<T,2>& vmin)
{return Vector<T,2>(clamp_min(v.x,vmin.x),clamp_min(v.y,vmin.y));}

template<class T> inline Vector<T,2>
clamp_min(const Vector<T,2>& v,const T& min)
{return Vector<T,2>(clamp_min(v.x,min),clamp_min(v.y,min));}

template<class T> inline Vector<T,2>
clamp_max(const Vector<T,2>& v,const Vector<T,2>& vmax)
{return Vector<T,2>(clamp_max(v.x,vmax.x),clamp_max(v.y,vmax.y));}

template<class T> inline Vector<T,2>
clamp_max(const Vector<T,2>& v,const T& max)
{return Vector<T,2>(clamp_max(v.x,max),clamp_max(v.y,max));}

template<class T> inline bool
in_bounds(const Vector<T,2>& v,const Vector<T,2>& vmin,const Vector<T,2>& vmax)
{return in_bounds(v.x,vmin.x,vmax.x) && in_bounds(v.y,vmin.y,vmax.y);}

template<class T> inline Vector<T,2>
wrap(const Vector<T,2>& v,const Vector<T,2>& vmin,const Vector<T,2>& vmax)
{return Vector<T,2>(wrap(v.x,vmin.x,vmax.x),wrap(v.y,vmin.y,vmax.y));}

}
