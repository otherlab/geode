//#####################################################################
// Class RayIntersection
//#####################################################################
//
// A Ray with additional information about an intersection
//
//#####################################################################

#pragma once

#include <geode/geometry/Box.h>
#include <geode/geometry/Ray.h>
#include <geode/vector/Vector.h>
#include <geode/vector/normalize.h>
#include <cmath>
#include <limits>
namespace geode {
using std::numeric_limits;

template<class TV>
class RayIntersection : public Ray<TV>
{
public:
    using Ray<TV>::start;
    using Ray<TV>::direction;
    typedef typename TV::Scalar T;
    T t_max; // maximum value of t allowed for the ray
    int aggregate_id; // indicates the piece of an aggregate object that is intersected by t_max
    enum Location {StartPoint,EndPoint,InteriorPoint,LocationUnknown};
    Location intersection_location; // indicates what type of intersection happened, LocationUnknown is used if not computed
    Box<TV> bounding_box;

    RayIntersection()
     : Ray<TV>(),t_max(numeric_limits<T>::infinity()),aggregate_id(-1),intersection_location(LocationUnknown)
    { }

    RayIntersection(const TV& start_input,const TV& direction_input,const bool already_normalized=false)
     : Ray<TV>(start_input, direction_input, already_normalized), t_max(numeric_limits<T>::infinity()),aggregate_id(-1),intersection_location(LocationUnknown)
    { }

    explicit RayIntersection(const Segment<TV>& segment)
     : Ray<TV>(segment.x0,segment.vector(),true),aggregate_id(-1),intersection_location(LocationUnknown)
    {
        // We lie above and tell the base Ray constructor that direction is normalized
        // We normalize it here and save the length in t_max
        t_max=normalize(direction);
    }

    void initialize(const TV& start_input,const TV& direction_input,const bool already_normalized=false)
    {start=start_input;direction=direction_input;t_max=numeric_limits<T>::infinity();aggregate_id=-1;intersection_location=LocationUnknown;
    if(!already_normalized) direction.normalize();}

    void save_intersection_information(RayIntersection<TV>& storage_ray) const
    {storage_ray.t_max=t_max;storage_ray.aggregate_id=aggregate_id;storage_ray.intersection_location=intersection_location;}

    void restore_intersection_information(const RayIntersection<TV>& storage_ray)
    {t_max=storage_ray.t_max;aggregate_id=storage_ray.aggregate_id;intersection_location=storage_ray.intersection_location;}

    void compute_bounding_box()
    {bounding_box=Box<TV>::bounding_box(start,point(t_max));}

    static bool create_non_degenerate_ray(const TV& start,const TV& length_and_direction,RayIntersection<TV>& ray)
    {T length_squared=length_and_direction.sqr_magnitude();
    if(length_squared>0){ray.t_max=sqrt(length_squared);ray.start=start;ray.direction=length_and_direction/ray.t_max;return true;}
    else return false;}

    // for sorting in containers
    inline bool operator<(RayIntersection<TV> const &r) const {
      return t_max < r.t_max;
    }
};

template<class TV> std::ostream &operator<<(std::ostream &output,const RayIntersection<TV> &ray)
{output<<"start = "<<ray.start<<", direction = "<<ray.direction<<", ";
output<<"t_max = "<<ray.t_max;output<<std::endl;
return output;
}

} // geode namespace
