//#####################################################################
// Class Box
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
#include <other/core/geometry/forward.h>
#include <other/core/math/clamp.h>
#include <other/core/math/Zero.h>
#include <other/core/vector/Vector.h>
#include <limits>
namespace other{

using std::numeric_limits;

template<class TV> struct IsScalarBlock<Box<TV> >:public IsScalarBlock<TV>{};
template<class TV> struct is_packed_pod<Box<TV> >:public is_packed_pod<TV>{};

template<class TV> PyObject* to_python(const Box<TV>& box) OTHER_EXPORT;
template<class TV> struct FromPython<Box<TV> >{OTHER_EXPORT static Box<TV> convert(PyObject* object);};

template<class TV>
class Box
{
    typedef typename TV::Scalar T;
    struct Unusable{};
public:
    template<class T2> struct Rebind{typedef Box<T2> type;};
    typedef T Scalar;
    typedef TV VectorT;
    enum Workaround {d=TV::dimension};

    TV min,max;

    Box()
        :min(numeric_limits<T>::max()*TV::ones()),max((T)-numeric_limits<T>::max()*TV::ones())
    {}

    Box(const T xmin,const T xmax)
        :min(xmin),max(xmax)
    {
        BOOST_STATIC_ASSERT(d==1);
    }

    Box(const T xmin,const T xmax,const T ymin,const T ymax)
        :min(xmin,ymin),max(xmax,ymax)
    {
        BOOST_STATIC_ASSERT(d==2);
    }

    Box(const T xmin,const T xmax,const T ymin,const T ymax,const T zmin,const T zmax)
        :min(xmin,ymin,zmin),max(xmax,ymax,zmax)
    {
        BOOST_STATIC_ASSERT(d==3);
    }

    Box(const TV& minimum_corner,const TV& maximum_corner)
        :min(minimum_corner),max(maximum_corner)
    {}

    template<class T2> explicit Box(const Box<T2>& box)
        :min(TV(box.min)),max(TV(box.max))
    {}

    explicit Box(const TV& point)
        :min(point),max(point)
    {}

    Box(const Box<TV>& box,const Frame<Vector<T,1> >& frame) // allow 1d boxes to be used as oriented boxes
        :min(box.min),max(box.max)
    {
        BOOST_STATIC_ASSERT(d==1);
    }

    Box<TV> axis_aligned_bounding_box() const
    {return *this;}

    static Box<TV> unit_box()
    {return Box<TV>(TV(),TV::ones());}

    static Box<TV> zero_box()
    {return Box<TV>(TV(),TV());}

    static Box<TV> empty_box()
    {return Box<TV>(numeric_limits<T>::max()*TV::ones(),(T)-numeric_limits<T>::max()*TV::ones());}
  
    static Box<TV> full_box()
    {return Box<TV>((T)-numeric_limits<T>::max()*TV::ones(),numeric_limits<T>::max()*TV::ones());}

    bool empty() const
    {return !min.all_less_equal(max);}

    bool operator==(const Box<TV>& r) const
    {return min==r.min && max==r.max;}

    bool operator!=(const Box<TV>& r) const
    {return !(*this==r);}

    Box<TV> operator-() const
    {return Box<TV>(-max,-min);}

    Box<TV>& operator+=(const Box<TV>& r)
    {min+=r.min;max+=r.max;return *this;}

    Box<TV>& operator-=(const Box<TV>& r)
    {min-=r.max;max-=r.min;return *this;}

    Box<TV> operator+(const Box<TV>& r) const
    {return Box<TV>(min+r.min,max+r.max);}

    Box<TV> operator-(const Box<TV>& r) const
    {return Box<TV>(min-r.max,max-r.min);}

    Box<TV> operator*(const T a) const
    {return a>=0?Box<TV>(min*a,max*a):Box<TV>(max*a,min*a);}

    Box<TV>& operator*=(const T a)
    {return *this=*this*a;}

    Box<TV> operator/(const T a) const
    {assert(a!=0);return *this*inverse(a);}

    Box<TV>& operator/=(const T a)
    {return *this=*this/a;}

    TV sizes() const
    {return max-min;}

    TV center() const
    {return (T).5*(min+max);}

    void corners(Array<TV>& corners) const
    {BOOST_STATIC_ASSERT(d==1);corners.resize(2);corners(0)=min;corners(1)=max;}

    void corners(Array<TV,2>& corners) const
    {BOOST_STATIC_ASSERT(d==2);corners.resize(2,2);
    for(int i=0;i<=1;i++) for(int j=0;j<=1;j++)
        corners(i,j)=TV(i?max.x:min.x,j?max.y:min.y);}

    void corners(Array<TV,3>& corners) const
    {BOOST_STATIC_ASSERT(d==3);corners.resize(2,2,2);
    for(int i=0;i<=1;i++) for(int j=0;j<=1;j++) for(int k=0;k<=1;k++)
        corners(i,j,k)=TV(i?max.x:min.x,j?max.y:min.y,k?max.z:min.z);}

    T volume() const
    {return empty()?(T)0:sizes().product();}

    T surface_area() const
    {BOOST_STATIC_ASSERT(d==3);Vector<T,3> size(sizes());return 2*(size.x*(size.y+size.z)+size.y*size.z);}

    void enlarge(const TV& point)
    {min=TV::componentwise_min(min,point);max=TV::componentwise_max(max,point);}

    void enlarge_nonempty(const TV& point)
    {assert(!empty());for(int i=0;i<d;i++) if(point[i]<min[i]) min[i]=point[i];else if(point[i]>max[i]) max[i]=point[i];}

    void enlarge_nonempty(const TV& p1,const TV& p2)
    {enlarge_nonempty(p1);enlarge_nonempty(p2);}

    void enlarge_nonempty(const TV& p1,const TV& p2,const TV& p3)
    {enlarge_nonempty(p1);enlarge_nonempty(p2);enlarge_nonempty(p3);}

    template<class TArray>
    void enlarge_nonempty(const TArray& points)
    {STATIC_ASSERT_SAME(typename TArray::Element,TV);
    for(int i=0;i<points.size();i++) enlarge_nonempty(points(i));}

    void enlarge(const Box<TV>& box)
    {min=TV::componentwise_min(min,box.min);max=TV::componentwise_max(max,box.max);}

    void enlarge_nonempty(const Box<TV>& box)
    {enlarge(box);}

    void change_size(const T delta)
    {min-=delta;max+=delta;}

    void change_size(const TV& delta)
    {min-=delta;max+=delta;}

    Box<TV> thickened(const T thickness_over_two) const
    {return Box<TV>(min-thickness_over_two,max+thickness_over_two);}

    Box<TV> thickened(const TV& thickness_over_two) const
    {return Box<TV>(min-thickness_over_two,max+thickness_over_two);}

    static Box<TV> combine(const Box<TV>& box1,const Box<TV>& box2)
    {return Box<TV>(TV::componentwise_min(box1.min,box2.min),TV::componentwise_max(box1.max,box2.max));}

    static Box<TV> intersect(const Box<TV>& box1,const Box<TV>& box2) // assumes nonnegative entries
    {return Box<TV>(TV::componentwise_max(box1.min,box2.min),TV::componentwise_min(box1.max,box2.max));}

    void scale_about_center(const T factor)
    {TV center=(T).5*(min+max),length_over_two=factor*(T).5*(max-min);min=center-length_over_two;max=center+length_over_two;}

    void scale_about_center(const TV factor)
    {TV center=(T).5*(min+max),length_over_two=factor*(T).5*(max-min);min=center-length_over_two;max=center+length_over_two;}

    void scale_about_center(const T x_factor,const T y_factor)
    {BOOST_STATIC_ASSERT(d==2);scale_about_center(TV(x_factor,y_factor));}

    void scale_about_center(const T x_factor,const T y_factor,const T z_factor)
    {BOOST_STATIC_ASSERT(d==3);scale_about_center(TV(x_factor,y_factor,z_factor));}

    bool lazy_inside(const TV& location) const
    {return location.all_greater_equal(min) && location.all_less_equal(max);}

    bool lazy_inside_half_open(const TV& location) const
    {return location.all_greater_equal(min) && location.all_less(max);}

    bool inside(const TV& location,const T thickness_over_two) const
    {return thickened(-thickness_over_two).lazy_inside(location);}

    bool inside(const TV& location,const Zero thickness_over_two) const
    {return lazy_inside(location);}

    bool lazy_outside(const TV& location) const
    {return !lazy_inside(location);}

    bool outside(const TV& location,const T thickness_over_two) const
    {return thickened(thickness_over_two).lazy_outside(location);}

    bool outside(const TV& location,const Zero thickness_over_two) const
    {return lazy_outside(location);}

    bool boundary(const TV& location,const T thickness_over_two) const
    {bool strict_inside=location.all_greater(min+thickness_over_two) && location.all_less(max-thickness_over_two);
    return !strict_inside && !outside(location,thickness_over_two);}

    TV clamp(const TV& location) const
    {return other::clamp(location,min,max);}

    T clamp(const T& location) const
    {BOOST_STATIC_ASSERT(d==1);return clamp(TV(location)).x;}

    void enlarge_by_sign(const TV& v)
    {for(int i=1;i<=d;i++) if(v(i)>0) max(i)+=v(i);else min(i)+=v(i);}

    void project_points_onto_line(const TV& direction,T& line_min,T& line_max) const
    {line_min=line_max=dot(direction,min);TV e=direction*(max-min);
    for(int i=0;i<d;i++) if(e(i)>0) line_max+=e(i);else line_min+=e(i);}

    TV point_from_normalized_coordinates(const TV& weights) const
    {return min+weights*(max-min);}

    bool contains(const Box<TV>& box) const
    {return min.all_less_equal(box.min) && max.all_greater_equal(box.max);}

    bool lazy_intersects(const Box<TV>& box) const
    {return min.all_less_equal(box.max) && max.all_greater_equal(box.min);}

    bool intersects(const Box<TV>& box,const T thickness_over_two) const
    {return thickened(thickness_over_two).lazy_intersects(box);}

    bool intersects(const Box<TV>& box,const Zero thickness_over_two) const
    {return lazy_intersects(box);}

    bool intersects(const Box<TV>& box) const
    {return lazy_intersects(box);}

    T intersection_area(const Box<TV>& box) const
    {return intersect(*this,box).robust_size();}

    Box<Vector<T,d-1> > horizontal_box() const
    {return Box<Vector<T,d-1> >(min.horizontal_vector(),max.horizontal_vector());}

    Box<Vector<T,d-1> > vertical_box() const
    {BOOST_STATIC_ASSERT(d==2);return Box<Vector<T,d-1> >(min.vertical_vector(),max.vertical_vector());}

    Box<Vector<T,d-1> > remove_dimension(int dimension) const
    {return Box<Vector<T,d-1> >(min.remove_index(dimension),max.remove_index(dimension));}

    const Box<TV>& bounding_box() const // for templatization purposes
    {return *this;}

    Vector<T,TV::dimension-1> principal_curvatures(const TV& X) const
    {return Vector<T,TV::dimension-1>();}

    T sqr_distance_bound(const TV& X) const
    {return sqr_magnitude(X-clamp(X));}

//#####################################################################
    TV normal(const TV& X) const;
    TV surface(const TV& X) const;
    T phi(const TV& X) const;
    static std::string name();
    string repr() const;
    bool lazy_intersects(const Ray<TV>& ray,T box_enlargement) const;
//#####################################################################
};
template<class TV>
inline Box<TV> operator+(const TV& a,const Box<TV>& b)
{return Box<TV>(a+b.min,a+b.max);}

template<class TV>
inline Box<TV> operator-(const TV& a,const Box<TV>& b)
{return Box<TV>(a-b.max,a-b.min);}

template<class TV> inline Box<TV> operator*(const typename TV::Scalar a,const Box<TV>& box)
{return box*a;}

template<class TV> inline Box<TV> bounding_box(const TV& p1,const TV& p2)
{Box<TV> box(p1);box.enlarge_nonempty(p2);return box;}

template<class TV> inline Box<TV> bounding_box(const TV& p1,const TV& p2,const TV& p3)
    {Box<TV> box(p1);box.enlarge_nonempty(p2,p3);return box;}

template<class TV> inline Box<TV> bounding_box(const TV& p1,const TV& p2,const TV& p3,const TV& p4)
{Box<TV> box(p1);box.enlarge_nonempty(p2,p3,p4);return box;}

template<class TArray> inline Box<Vector<typename TArray::Element::Scalar,TArray::Element::m> > bounding_box(const TArray& points)
{typedef typename TArray::Element TV;
if(!points.size()) return Box<TV>::empty_box();
Box<TV> box(points[0]);for(int i=1;i<points.size();i++) box.enlarge_nonempty(points[i]);return box;}

template<class TV>
inline std::ostream& operator<<(std::ostream& output,const Box<TV>& box)
{return output<<'['<<box.min<<','<<box.max<<']';}

template<class TV>
inline std::istream& operator>>(std::istream& input,Box<TV>& box)
{return input>>expect('[')>>box.min>>expect(',')>>box.max>>expect(']');}

}
