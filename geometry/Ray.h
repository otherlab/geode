//#####################################################################
// Class Ray
//#####################################################################
#pragma once

#include <other/core/geometry/Box.h>
#include <other/core/vector/Vector.h>
#include <other/core/vector/normalize.h>
#include <cmath>
#include <limits>
namespace other{
using std::numeric_limits;

template<class TV>
class Ray
{
public:
    typedef typename TV::Scalar T;
    typedef typename TV::template Rebind<int>::type TvInt;
    TV start; // start of the ray where t=0
    TV direction; // direction the ray sweeps out - unit vector
    T t_max; // maximum value of t allowed for the ray
    int aggregate_id; // indicates the piece of an aggregate object that is intersected by t_max
    enum Location {StartPoint,EndPoint,InteriorPoint,LocationUnknown};
    Location intersection_location; // indicates what type of intersection happened, LocationUnknown is used if not computed
    Box<TV> bounding_box;

    // used for triangle hierarchy fast lazy_box_intersection 
    mutable bool computed_lazy_box_intersection_acceleration_data;
    mutable TV inverse_direction;
    mutable TvInt direction_is_negative;

    Ray()
        :start(TV()),t_max(numeric_limits<T>::infinity()),aggregate_id(-1),intersection_location(LocationUnknown),computed_lazy_box_intersection_acceleration_data(false)
    {
         direction=TV::axis_vector(TV::dimension-1);
    }

    Ray(const TV& start_input,const TV& direction_input,const bool already_normalized=false)
        :start(start_input),direction(direction_input),t_max(numeric_limits<T>::infinity()),aggregate_id(-1),intersection_location(LocationUnknown),computed_lazy_box_intersection_acceleration_data(false)
    {if(!already_normalized) direction.normalize();}

    Ray(const Segment<TV>& segment)
      :start(segment.x0),direction(segment.vector()),aggregate_id(-1),intersection_location(LocationUnknown),computed_lazy_box_intersection_acceleration_data(false)
    {t_max=normalize(direction);}

    void initialize(const TV& start_input,const TV& direction_input,const bool already_normalized=false)
    {start=start_input;direction=direction_input;t_max=numeric_limits<T>::infinity();aggregate_id=-1;intersection_location=LocationUnknown;computed_lazy_box_intersection_acceleration_data=false;
    if(!already_normalized) direction.normalize();}

    void save_intersection_information(Ray<TV>& storage_ray) const
    {storage_ray.t_max=t_max;storage_ray.aggregate_id=aggregate_id;storage_ray.intersection_location=intersection_location;}

    void restore_intersection_information(const Ray<TV>& storage_ray)
    {t_max=storage_ray.t_max;aggregate_id=storage_ray.aggregate_id;intersection_location=storage_ray.intersection_location;}

    TV point(const T t) const // finds the point on the ray, given by the parameter t
    {return start+t*direction;}

    void compute_bounding_box()
    {bounding_box=Box<TV>::bounding_box(start,point(t_max));}

    T parameter(const TV& point) const // finds the parameter t, given a point that lies on the ray
    {int axis=direction.dominant_axis();return (point[axis]-start[axis])/direction[axis];}

    TV reflected_direction(const TV& normal) const
    {return 2*TV::dot_product(-direction,normal)*normal+direction;}

    static bool create_non_degenerate_ray(const TV& start,const TV& length_and_direction,Ray<TV>& ray)
    {T length_squared=length_and_direction.sqr_magnitude();
    if(length_squared>0){ray.t_max=sqrt(length_squared);ray.start=start;ray.direction=length_and_direction/ray.t_max;return true;}
    else return false;}

    void compute_lazy_box_intersection_acceleration_data() const;
};

template<class TV> std::ostream &operator<<(std::ostream &output,const Ray<TV> &ray)
{output<<"start = "<<ray.start<<", direction = "<<ray.direction<<", ";
output<<"t_max = "<<ray.t_max;output<<std::endl;
return output;
}

}
