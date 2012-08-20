//#####################################################################
// Class Interval
//#####################################################################
#pragma once

#include <other/core/vector/ScalarPolicy.h>
#include <other/core/math/clamp.h>
#include <other/core/math/max.h>
#include <other/core/math/min.h>
#include <other/core/math/Zero.h>
#include <other/core/python/forward.h>
#include <other/core/utility/STATIC_ASSERT_SAME.h>
#include <limits>
#include <cfloat>
namespace other {

using std::numeric_limits;

template<class TV> struct IsScalarBlock<Interval<TV> >:public IsScalarBlock<TV>{};

template<class T> PyObject* to_python(const Interval<T>& self) OTHER_EXPORT;
template<class T> struct FromPython<Interval<T> >{OTHER_EXPORT static Interval<T> convert(PyObject* object);};

template<class T>
class Interval
{
public:
    template<class T2> struct Rebind{typedef Interval<T2> type;};
    typedef T Scalar;

    T min,max;

    Interval()
        :min(numeric_limits<T>::max()),max(-numeric_limits<T>::max())
    {}

    Interval(const T value)
        :min(value),max(value)
    {}

    Interval(const T min,const T max)
        :min(min),max(max)
    {}

    template<class T2> explicit Interval(const Interval<T2>& interval)
        :min(T(interval.min)),max(T(interval.max))
    {}

    static Interval unit_box()
    {return Interval((T)0,(T)1);}

    static Interval zero_box()
    {return Interval();}

    static Interval empty_box()
    {return Interval();}

    static Interval full_box()
    {return Interval(-numeric_limits<T>::max(),numeric_limits<T>::max());}

    bool empty() const
    {return min>max;}

    bool operator==(const Interval& r) const
    {return min==r.min && max==r.max;}

    bool operator!=(const Interval& r) const
    {return !(*this==r);}

    Interval operator-() const
    {return Interval(-max,-min);}

    Interval& operator+=(const Interval& r)
    {min+=r.min;max+=r.max;return *this;}

    Interval& operator-=(const Interval& r)
    {min-=r.max;max-=r.min;return *this;}

    Interval operator+(const Interval& r) const
    {return Interval(min+r.min,max+r.max);}

    Interval operator-(const Interval& r) const
    {return Interval(min-r.max,max-r.min);}

    Interval operator*(const T a) const
    {return a>=0?Interval(min*a,max*a):Interval(max*a,min*a);}

    Interval& operator*=(const T a)
    {return *this=*this*a;}

    Interval operator/(const T a) const
    {assert(a!=0);return *this*inverse(a);}

    Interval& operator/=(const T a)
    {return *this=*this/a;}

    T center() const
    {return (T).5*(min+max);}

    T size() const
    {return max-min;}

    T minabs() const
    {return min>0?min:max<0?-max:0;}

    T maxabs() const
    {return maxabs(min,max);}

    void enlarge(const T& point)
    {min=other::min(min,point);max=other::max(max,point);}

    void enlarge_nonempty(const T& point)
    {assert(!empty());if(point<min) min=point;else if(point>max) max=point;}

    void enlarge_nonempty(const T& p1,const T& p2)
    {enlarge_nonempty(p1);enlarge_nonempty(p2);}

    void enlarge_nonempty(const T& p1,const T& p2,const T& p3)
    {enlarge_nonempty(p1);enlarge_nonempty(p2);enlarge_nonempty(p3);}

    template<class TArray>
    void enlarge_nonempty(const TArray& points)
    {STATIC_ASSERT_SAME(typename TArray::Element,T);
    for(int i=0;i<points.size();i++) enlarge_nonempty(points(i));}

    void enlarge(const Interval& interval)
    {min=other::min(min,interval.min);max=other::max(max,interval.max);}

    void change_size(const T delta)
    {min-=delta;max+=delta;}

    Interval thickened(const T thickness_over_two) const
    {return Interval(min-thickness_over_two,max+thickness_over_two);}

    static Interval combine(const Interval& box1,const Interval& box2)
    {return Interval(other::min(box1.min,box2.min),other::max(box1.max,box2.max));}

    static Interval intersect(const Interval& box1,const Interval& box2)
    {return Interval(other::max(box1.min,box2.min),other::min(box1.max,box2.max));}

    void scale_about_center(const T factor)
    {T center=(T).5*(min+max),length_over_two=factor*(T).5*(max-min);min=center-length_over_two;max=center+length_over_two;}

    bool lazy_inside(const T& location) const
    {return min<=location && location<=max;}

    bool lazy_inside_half_open(const T& location) const
    {return min<=location && location<max;}

    bool inside(const T& location,const T thickness_over_two) const
    {return thickened(-thickness_over_two).lazy_inside(location);}

    bool inside(const T& location,const Zero thickness_over_two) const
    {return lazy_inside(location);}

    bool lazy_outside(const T& location) const
    {return !lazy_inside(location);}

    bool outside(const T& location,const T thickness_over_two) const
    {return thickened(thickness_over_two).lazy_outside(location);}

    bool outside(const T& location,const Zero thickness_over_two) const
    {return lazy_outside(location);}

    bool boundary(const T& location,const T thickness_over_two) const
    {bool strict_inside=min+thickness_over_two<location && location<max-thickness_over_two;
    return !strict_inside && !outside(location,thickness_over_two);}

    T clamp(const T& location) const
    {return other::clamp(location,min,max);}

    bool contains(const Interval& interval) const
    {return min<=interval.min && interval.max<=max;}

    bool lazy_intersection(const Interval& interval) const
    {return min<=interval.max && interval.min<=max;}

    bool intersection(const Interval& interval,const T thickness_over_two) const
    {return thickened(thickness_over_two).lazy_intersection(interval);}

    bool intersection(const Interval& interval,const Zero thickness_over_two) const
    {return lazy_intersection(interval);}

    bool intersection(const Interval& interval) const
    {return lazy_intersection(interval);}

    T signed_distance(const T& X) const
    {return abs(X-center())-size();}
};

template<class T> inline typename boost::enable_if<IsScalar<T>,Interval<T> >::type
bounding_box(const T& p1,const T& p2) {
  Interval<T> interval(p1);interval.enlarge_nonempty(p2);return interval;
}

template<class T> inline typename boost::enable_if<IsScalar<T>,Interval<T> >::type
bounding_box(const T& p1,const T& p2,const T& p3) {
  Interval<T> interval(p1);interval.enlarge_nonempty(p2,p3);return interval;
}

template<class T> inline typename boost::enable_if<IsScalar<T>,Interval<T> >::type
bounding_box(const T& p1,const T& p2,const T& p3,const T& p4) {
  Interval<T> interval(p1);interval.enlarge_nonempty(p2,p3,p4);return interval;
}

template<class TArray> inline typename boost::enable_if<IsScalar<typename TArray::Element>,Interval<typename TArray::Element> >::type
bounding_box(const TArray& points) {
  typedef typename TArray::Element T;
  if (!points.size()) return Interval<T>::empty_box();
  Interval<T> interval(points[0]);
  for (int i=1;i<points.size();i++) interval.enlarge_nonempty(points[i]);
  return interval;
}

template<class T>
inline Interval<T> operator+(const T& a,const Interval<T>& b)
{return Interval<T>(a+b.min,a+b.max);}

template<class T>
inline Interval<T> operator-(const T& a,const Interval<T>& b)
{return Interval<T>(a-b.max,a-b.min);}

template<class T> inline Interval<T> operator*(const T a,const Interval<T>& interval)
{return interval*a;}

template<class T> inline Interval<T> exp(const Interval<T>& x)
{return Interval<T>(exp(x.min),exp(x.max));}

template<class T> inline Interval<T> sqr(const Interval<T>& x) {
  T smin = sqr(x.min), smax = sqr(x.max);
  return x.min>=0?Interval<T>(smin,smax):x.max<=0?Interval<T>(smax,smin):Interval<T>(0,max(smin,smax));
}

template<class T>
inline std::ostream& operator<<(std::ostream& output,const Interval<T>& interval)
{output<<"("<<interval.min<<","<<interval.max<<")";return output;}
}
