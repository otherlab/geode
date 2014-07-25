//#####################################################################
// Class Vector1d
//#####################################################################
#pragma once

#include <geode/vector/ScalarPolicy.h>
#include <geode/vector/Vector0d.h>
#include <geode/math/inverse.h>
#include <geode/math/clamp.h>
#include <geode/math/argmax.h>
#include <geode/math/argmin.h>
#include <geode/math/isnan.h>
#include <geode/math/max.h>
#include <geode/math/min.h>
#include <geode/math/sign.h>
#include <geode/math/sqr.h>
#include <geode/utility/debug.h>
#include <geode/utility/STATIC_ASSERT_SAME.h>
#include <cmath>
namespace geode {

using ::std::floor;
using ::std::ceil;
using ::std::sin;
using ::std::cos;

template<class T>
class Vector<T,1>
{
    struct Unusable{};
public:
    typedef typename mpl::if_<IsScalar<T>,T,Unusable>::type Scalar;
    typedef T Element;
    typedef T value_type; // for stl
    typedef T* iterator; // for stl
    typedef const T* const_iterator; // for stl
    static const int dimension = 1;
    static const int m = 1;
    static const bool is_const = false;

    T x;

    Vector()
        :x()
    {
        static_assert(sizeof(Vector)==sizeof(T),"");
    }

    explicit Vector(const T& x)
        :x(x)
    {}

    template<class T2> explicit Vector(const Vector<T2,1>& vector)
        :x(T(vector.x))
    {}

    Vector(const Vector& vector)
        :x(vector.x)
    {}

    template<class TVector>
    explicit Vector(const TVector& v,typename EnableForVectorLike<T,1,TVector,Unusable>::type unusable=Unusable())
        :x(v[0])
    {}

    template<int n>
    Vector(const Vector<T,n>& v1,const Vector<T,1-n>& v2)
    {
        for(int i=0;i<n;i++) (*this)(i)=v1(i);for(int i=n;i<1;i++) (*this)(i)=v2(i-n);
    }

    template<class TVector> typename EnableForVectorLike<T,1,TVector,Vector&>::type
    operator=(const TVector& v)
    {
        x=v[0];return *this;
    }

    Vector& operator=(const Vector& v)
    {
        x=v[0];return *this;
    }

    int size() const
    {return 1;}

    const T& operator[](const int i) const
    {assert(i==0);return x;}

    T& operator[](const int i)
    {assert(i==0);return x;}

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
    {return x==v.x;}

    bool operator!=(const Vector& v) const
    {return x!=v.x;}

    Vector operator-() const
    {return Vector(-x);}

    Vector& operator+=(const Vector& v)
    {x+=v.x;return *this;}

    Vector& operator-=(const Vector& v)
    {x-=v.x;return *this;}

    Vector& operator*=(const Vector& v)
    {x*=v.x;return *this;}

    Vector& operator+=(const T& a)
    {x+=a;return *this;}

    Vector& operator-=(const T& a)
    {x-=a;return *this;}

    Vector& operator*=(const T& a)
    {x*=a;return *this;}

    Vector& operator*=(const IntInverse<T> a)
    {x*=a;return *this;}

    Vector& operator/=(const T& a)
    {return *this*=inverse(a);}

    Vector& operator/=(const Vector& v)
    {x/=v.x;return *this;}

    Vector operator+(const Vector& v) const
    {return Vector(x+v.x);}

    Vector operator-(const Vector& v) const
    {return Vector(x-v.x);}

    Vector operator*(const Vector& v) const
    {return Vector(x*v.x);}

    Vector operator/(const Vector& v) const
    {return Vector(x/v.x);}

    Vector operator+(const T& a) const
    {return Vector(x+a);}

    Vector operator-(const T& a) const
    {return Vector(x-a);}

    Vector operator*(const T& a) const
    {return Vector(x*a);}

    Vector operator*(const IntInverse<T> a) const
    {return Vector(x*a);}

    Vector operator/(const T& a) const
    {return *this*inverse(a);}

    Vector operator&(const T& a) const
    {return Vector(x&a);}

    T sqr_magnitude() const
    {return sqr(x);}

    T magnitude() const
    {return abs(x);}

    T L1_Norm() const
    {return abs(x);}

    T normalize()
    {T magnitude=abs(x);x=(T)(x>=0?1:-1);return magnitude;}

    Vector normalized() const
    {return Vector((T)(x>=0?1:-1));}

    T min() const
    {return x;}

    T max() const
    {return x;}

    T maxabs() const
    {return abs(x);}

    int argmin() const
    {return 0;}

    int argmax() const
    {return 0;}

    int dominant_axis() const
    {return 0;}

    bool elements_equal() const
    {return true;}

    static Vector componentwise_min(const Vector& v1,const Vector& v2)
    {return Vector(geode::min(v1.x,v2.x));}

    static Vector componentwise_max(const Vector& v1,const Vector& v2)
    {return Vector(geode::max(v1.x,v2.x));}

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
    {return x;}

    T average() const
    {return x;}

    T product() const
    {return x;}

    const Vector& column_sum() const
    {return *this;}

    int number_true() const
    {STATIC_ASSERT_SAME(T,bool);return x;}

    static Vector axis_vector(const int axis)
    {assert(axis==0);return Vector((T)1);}

    static Vector ones()
    {return Vector((T)1);}

    static Vector repeat(const T& constant)
    {return Vector(constant); }

    void fill(const T& constant)
    {x=constant;}

    void get(T& element1) const
    {element1=x;}

    void set(const T& element1)
    {x=element1;}

    template<class TFunction>
    static Vector map(const TFunction& f,const Vector& v)
    {return Vector(f(v.x));}

    int find(const T& element) const
    {return x==element?0:-1;}

    bool contains(const T& element) const
    {return x==element;}

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

    Vector<T,0> remove_index(const int index) const
    {assert(0==index);return Vector<T,0>();}

    Vector<T,2> insert(const T& element,const int index) const
    {Vector<T,2> r;r[index]=element;r[1-index]=x;return r;}

    Vector<T,2> append(const T& element) const
    {return Vector<T,2>(x,element);}

    template<int d2> Vector<T,1+d2> extend(const Vector<T,d2>& elements) const
    {Vector<T,1+d2> r;r[0]=x;for(int i=0;i<d2;i++) r[i+1]=elements[i];return r;}

    Vector sorted() const
    {return *this;}

    Vector reversed() const
    {return *this;}

    template<int d1,int d2> Vector<T,d2-d1> slice() const
    {static_assert(0<=d1 && d1<=d2 && d2<=1,"");
    Vector<T,d2-d1> r;for(int i=d1;i<d2;i++) r[i-d1]=(*this)[i];return r;}

    template<int n> void split(Vector<T,n>& v1,Vector<T,1-n>& v2) const
    {for(int i=0;i<n;i++) v1(i)=(*this)(i);
    for(int i=n;i<1;i++) v2(i-n)=(*this)(i);}

    T* begin() // for stl
    {return &x;}

    const T* begin() const // for stl
    {return &x;}

    T* end() // for stl
    {return &x+1;}

    const T* end() const // for stl
    {return &x+1;}

    T& front() { return x; }
    const T& front() const { return x; }
    T& back() { return x; }
    const T& back() const { return x; }
//#####################################################################
};

//#####################################################################
// Miscellaneous free operators and functions
//#####################################################################
template<class T> inline Vector<T,1>
operator+(const typename Hide<T>::type& a,const Vector<T,1>& v)
{return Vector<T,1>(a+v.x);}

template<class T> inline Vector<T,1>
operator-(const typename Hide<T>::type& a,const Vector<T,1>& v)
{return Vector<T,1>(a-v.x);}

template<class T> inline Vector<T,1>
operator*(const typename Hide<T>::type& a,const Vector<T,1>& v)
{return Vector<T,1>(a*v.x);}

template<class T> inline Vector<T,1>
operator/(const typename Hide<T>::type& a,const Vector<T,1>& v)
{return Vector<T,1>(a/v.x);}

template<class T> inline Vector<T,1>
abs(const Vector<T,1>& v)
{return Vector<T,1>(abs(v.x));}

template<class T> inline Vector<T,1>
floor(const Vector<T,1>& v)
{return Vector<T,1>(floor(v.x));}

template<class T> inline Vector<T,1>
ceil(const Vector<T,1>& v)
{return Vector<T,1>(ceil(v.x));}

template<class T> inline Vector<T,1>
exp(const Vector<T,1>& v)
{return Vector<T,1>(exp(v.x));}

template<class T> inline Vector<T,1>
sin(const Vector<T,1>& v)
{return Vector<T,1>(sin(v.x));}

template<class T> inline Vector<T,1>
cos(const Vector<T,1>& v)
{return Vector<T,1>(cos(v.x));}

template<class T> inline Vector<T,1>
sqrt(const Vector<T,1>& v)
{return Vector<T,1>(sqrt(v.x));}

template<class T> inline Vector<T,1>
inverse(const Vector<T,1>& v)
{return Vector<T,1>(1/v.x);}

template<class T>
inline bool isnan(const Vector<T,1>& v)
{return isnan(v.x);}

template<class T> inline void
cyclic_shift(Vector<T,1>& v)
{}

template<class T> inline T
dot(const Vector<T,1>& v1,const Vector<T,1>& v2)
{return v1.x*v2.x;}

template<class T> inline Vector<T,1>
cross(const Vector<T,0>,const Vector<T,1>&)
{return Vector<T,1>();}

template<class T> inline Vector<T,1>
cross(const Vector<T,1>&,const Vector<T,0>)
{return Vector<T,1>();}

template<class T> inline Vector<T,0>
cross(const Vector<T,1>&,const Vector<T,1>&)
{return Vector<T,0>();}

template<class T> inline bool
isfinite(const Vector<T,1>& v)
{return isfinite(v.x);}

template<class T> inline bool all_greater(const Vector<T,1>& v0, const Vector<T,1>& v1) {
  return v0.x>v1.x;
}

template<class T> inline bool all_less(const Vector<T,1>& v0, const Vector<T,1>& v1) {
  return v0.x<v1.x;
}

template<class T> inline bool all_greater_equal(const Vector<T,1>& v0, const Vector<T,1>& v1) {
  return v0.x>=v1.x;
}

template<class T> inline bool all_less_equal(const Vector<T,1>& v0, const Vector<T,1>& v1) {
  return v0.x<=v1.x;
}

//#####################################################################
// Functions clamp, clamp_min, clamp_max, in_bounds
//#####################################################################
template<class T> inline Vector<T,1>
clamp(const Vector<T,1>& v,const Vector<T,1>& vmin,const Vector<T,1>& vmax)
{return Vector<T,1>(clamp(v.x,vmin.x,vmax.x));}

template<class T> inline Vector<T,1>
clamp(const Vector<T,1>& v,T min,T max)
{return Vector<T,1>(clamp(v.x,min,max));}

template<class T> inline Vector<T,1>
clamp_min(const Vector<T,1>& v,const Vector<T,1>& vmin)
{return Vector<T,1>(clamp_min(v.x,vmin.x));}

template<class T> inline Vector<T,1>
clamp_min(const Vector<T,1>& v,const T& min)
{return Vector<T,1>(clamp_min(v.x,min));}

template<class T> inline Vector<T,1>
clamp_max(const Vector<T,1>& v,const Vector<T,1>& vmax)
{return Vector<T,1>(clamp_max(v.x,vmax.x));}

template<class T> inline Vector<T,1>
clamp_max(const Vector<T,1>& v,const T& max)
{return Vector<T,1>(clamp_max(v.x,max));}

template<class T> inline bool
in_bounds(const Vector<T,1>& v,const Vector<T,1>& vmin,const Vector<T,1>& vmax)
{return in_bounds(v.x,vmin.x,vmax.x);}

template<class T> const int Vector<T,1>::dimension;
template<class T> const int Vector<T,1>::m;

}
