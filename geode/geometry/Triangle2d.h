//#####################################################################
// Class Triangle<T,2>
//#####################################################################
#pragma once

#include <geode/geometry/Box.h>
#include <geode/structure/Tuple.h>
#include <geode/vector/Vector.h>
#include <geode/vector/Matrix2x2.h>
namespace geode {

template<class T>
class Triangle<Vector<T,2> >
{
    typedef Vector<T,2> TV;
public:
    Vector<TV,3> X;

    Triangle() {}

    Triangle(const TV& x0,const TV& x1,const TV& x2)
      :X(x0,x1,x2) {}

    template<class TArray>
    Triangle(const TArray& X)
        :X(X)
    {
        BOOST_STATIC_ASSERT(TArray::m==3);
    }

    static T signed_area(const TV& x1,const TV& x2,const TV& x3)
    {return (T).5*TV::cross_product(x2-x1,x3-x1).x;}

    T signed_area() const
    {return signed_area(X[0],X[1],X[2]);}

    static T area(const TV& x1,const TV& x2,const TV& x3)
    {return abs(signed_area(x1,x2,x3));}

    T area() const
    {return abs(signed_area());}

    T size() const
    {return area();}

    T signed_size() const
    {return signed_area();}

    template<class TArray>
    static T signed_size(const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==3);return signed_area(X[0],X[1],X[2]);}

    template<class TArray>
    static T size(const TArray& X)
    {return abs(signed_size(X));}

    static bool check_orientation(const TV& x1,const TV& x2,const TV& x3)
    {return signed_area(x1,x2,x3)>=0;}

    bool check_orientation() const
    {return signed_area()>=0;}

    bool fix_orientation()
    {if(check_orientation()) return false;exchange(X[1],X[2]);return true;}

    template<class TArray>
    static T half_boundary_measure(const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==3);return (T).5*((X[0]-X[1]).magnitude()+(X[1]-X[2]).magnitude()+(X[2]-X[0]).magnitude());}

    T aspect_ratio() const
    {return aspect_ratio(X[0],X[1],X[2]);}

    static T aspect_ratio(const TV& x1_input,const TV& x2_input,const TV& x3_input)
    {TV u=x1_input-x2_input,v=x2_input-x3_input,w=x3_input-x1_input;
    T u2=TV::dot_product(u,u),v2=TV::dot_product(v,v),w2=TV::dot_product(w,w);
    return max(u2,v2,w2)/abs(TV::cross_product(u,v).x);}

    static T minimum_edge_length(const TV& x1,const TV& x2,const TV& x3)
    {return sqrt(minimum_edge_length_squared(x1,x2,x3));}

    static T minimum_edge_length_squared(const TV& x1,const TV& x2,const TV& x3)
    {return min((x2-x1).sqr_magnitude(),(x3-x1).sqr_magnitude(),(x3-x2).sqr_magnitude());}

    static T maximum_edge_length(const TV& x1,const TV& x2,const TV& x3)
    {return sqrt(maximum_edge_length_squared(x1,x2,x3));}

    static T maximum_edge_length_squared(const TV& x1,const TV& x2,const TV& x3)
    {return max((x2-x1).sqr_magnitude(),(x3-x1).sqr_magnitude(),(x3-x2).sqr_magnitude());}

    T minimum_altitude() const
    {return minimum_altitude(X[0],X[1],X[2]);}

    static T minimum_altitude(const TV& x1,const TV& x2,const TV& x3)
    {return 2*area(x1,x2,x3)/maximum_edge_length(x1,x2,x3);}

    static TV first_two_barycentric_coordinates(const TV& location,const TV& x1,const TV& x2,const TV& x3)
    {return Matrix<T,2>(x1-x3,x2-x3).robust_solve_linear_system(location-x3);}

    static Vector<T,3> barycentric_coordinates(const TV& location,const TV& x1,const TV& x2,const TV& x3)
    {TV w=first_two_barycentric_coordinates(location,x1,x2,x3);return Vector<T,3>(w.x,w.y,1-w.x-w.y);}

    template<class TArray>
    static Vector<T,3> barycentric_coordinates(const TV& location,const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==3);return barycentric_coordinates(location,X[0],X[1],X[2]);}

    Vector<T,3> barycentric_coordinates(const TV& location) const
    {return barycentric_coordinates(location,X[0],X[1],X[2]);}

    static TV point_from_barycentric_coordinates(const Vector<T,3>& weights,const TV& x1,const TV& x2,const TV& x3)
    {return weights.x*x1+weights.y*x2+weights.z*x3;}

    static TV point_from_barycentric_coordinates(const TV& weights,const TV& x1,const TV& x2,const TV& x3)
    {return weights.x*x1+weights.y*x2+(1-weights.x-weights.y)*x3;}

    TV point_from_barycentric_coordinates(const Vector<T,3>& weights) const
    {return point_from_barycentric_coordinates(weights,X[0],X[1],X[2]);}

    template<class TArray>
    static TV point_from_barycentric_coordinates(const Vector<T,3>& weights,const TArray& X)
    {BOOST_STATIC_ASSERT(TArray::m==3);return point_from_barycentric_coordinates(weights,X[0],X[1],X[2]);}

    Vector<T,3> sum_barycentric_coordinates(const Triangle& embedded_triangle) const
    {return barycentric_coordinates(embedded_triangle.X[0],X)+barycentric_coordinates(embedded_triangle.X[1],X)+barycentric_coordinates(embedded_triangle.X[2],X);}

    static TV center(const TV& x1,const TV& x2,const TV& x3) // centroid
    {return T(1./3)*(x1+x2+x3);}

    TV center() const // centroid
    {return center(X[0],X[1],X[2]);}

    TV incenter() const // intersection of angle bisectors
    {Vector<T,3> edge_lengths((X[2]-X[1]).magnitude(),(X[0]-X[2]).magnitude(),(X[1]-X[0]).magnitude());T perimeter=edge_lengths.x+edge_lengths.y+edge_lengths.z;assert(perimeter>0);
    return point_from_barycentric_coordinates(edge_lengths/perimeter);}

    static TV circumcenter(const TV& x1,const TV& x2,const TV& x3)
    {TV x1x2=x2-x1,x1x3=x3-x1,m1=(T).5*(x1+x2),m2=(T).5*(x1+x3),m1m2=m2-m1,x1x2_perp(-x1x2.y,x1x2.x);
    return m1+x1x2_perp*(TV::dot_product(m1m2,x1x3)/TV::dot_product(x1x2_perp,x1x3));}

    static Vector<T,3> circumcenter_barycentric_coordinates(const TV& x1,const TV& x2,const TV& x3)
    {TV a=x3-x2,b=x3-x1,c=x2-x1;T aa=a.sqr_magnitude(),bb=b.sqr_magnitude(),cc=c.sqr_magnitude();
    Vector<T,3> weights(aa*(bb+cc-aa),bb*(cc+aa-bb),cc*(aa+bb-cc));return weights/(weights.x+weights.y+weights.z);}

    T minimum_angle() const
    {TV s1=(X[0]-X[1]).normalized(),s2=(X[1]-X[2]).normalized(),s3=(X[2]-X[0]).normalized();
    return acos(max(TV::dot_product(s1,-s2),TV::dot_product(-s1,s3),TV::dot_product(s2,-s3)));}

    T maximum_angle() const
    {TV s1=(X[0]-X[1]).normalized(),s2=(X[1]-X[2]).normalized(),s3=(X[2]-X[0]).normalized();
    return acos(min(TV::dot_product(s1,-s2),TV::dot_product(-s1,s3),TV::dot_product(s2,-s3)));}

    bool outside(const TV& location,const T thickness_over_2=0)
    {return outside(location,X[0],X[1],X[2],thickness_over_2);}

    static bool outside(const TV& location,const TV& x1,const TV& x2,const TV& x3,const T thickness_over_2=0)
    {assert(check_orientation(x1,x2,x3));TV location_minus_x1=location-x1;
    TV edge1=x2-x1;if(TV::cross_product(location_minus_x1,edge1).x>thickness_over_2*edge1.magnitude()) return true;
    TV edge3=x1-x3;if(TV::cross_product(location_minus_x1,edge3).x>thickness_over_2*edge3.magnitude()) return true;
    TV edge2=x3-x2;if(TV::cross_product(location-x2,edge2).x>thickness_over_2*edge2.magnitude()) return true;
    return false;}

    Box<TV> bounding_box() const
    {return geode::bounding_box(X[0],X[1],X[2]);}

    static bool check_delaunay_criterion(TV a,TV b,TV c,TV d)
    {assert(check_orientation(a,b,c) && check_orientation(d,c,b));b-=a;c-=a;d-=a;
    return det(b.append(b.sqr_magnitude()),c.append(c.sqr_magnitude()),d.append(d.sqr_magnitude()))>=0;}

    // For templatization purposes
    static T min_weight(const Vector<T,3>& w) {
      return w.min();
    }

    GEODE_CORE_EXPORT Tuple<TV,Vector<T,3>> closest_point(const TV& location) const;
    GEODE_CORE_EXPORT T distance(const TV& location) const;
};

template<class T> std::ostream& operator<<(std::ostream& output,const Triangle<Vector<T,2> >& triangle)
{return output<<triangle.X;}

template<class T> static inline Vector<T,3> barycentric_coordinates(const Triangle<Vector<T,2>>& tri, const Vector<T,2>& p) {
  return tri.barycentric_coordinates(p);
}

}
