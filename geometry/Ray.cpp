//#####################################################################
// Class Ray
//#####################################################################
#include <other/core/geometry/Ray.h>
namespace other{

template<> void Ray<Vector<real,2> >::compute_lazy_box_intersection_acceleration_data() const {
  const T tolerance=(T)1e-10,large_number=(T)1e10;
  if(direction.x<0){direction_is_negative.x=1;inverse_direction.x=(direction.x<-tolerance)?1/direction.x:-large_number;}
  else{direction_is_negative.x=0;inverse_direction.x=(direction.x>tolerance)?1/direction.x:large_number;}
  if(direction.y<0){direction_is_negative.y=1;inverse_direction.y=(direction.y<-tolerance)?1/direction.y:-large_number;}
  else{direction_is_negative.y=0;inverse_direction.y=(direction.y>tolerance)?1/direction.y:large_number;}
  computed_lazy_box_intersection_acceleration_data=true;
}

#ifndef OTHER_FLOAT
template<> void Ray<Vector<float,2> >::compute_lazy_box_intersection_acceleration_data() const {
  const T tolerance=(T)1e-10,large_number=(T)1e10;
  if(direction.x<0){direction_is_negative.x=1;inverse_direction.x=(direction.x<-tolerance)?1/direction.x:-large_number;}
  else{direction_is_negative.x=0;inverse_direction.x=(direction.x>tolerance)?1/direction.x:large_number;}
  if(direction.y<0){direction_is_negative.y=1;inverse_direction.y=(direction.y<-tolerance)?1/direction.y:-large_number;}
  else{direction_is_negative.y=0;inverse_direction.y=(direction.y>tolerance)?1/direction.y:large_number;}
  computed_lazy_box_intersection_acceleration_data=true;
}
#endif

template<> void Ray<Vector<real,3> >::compute_lazy_box_intersection_acceleration_data() const {
  const T tolerance=(T)1e-10,large_number=(T)1e10;
  if(direction.x<0){direction_is_negative.x=1;inverse_direction.x=(direction.x<-tolerance)?1/direction.x:-large_number;}
  else{direction_is_negative.x=0;inverse_direction.x=(direction.x>tolerance)?1/direction.x:large_number;}
  if(direction.y<0){direction_is_negative.y=1;inverse_direction.y=(direction.y<-tolerance)?1/direction.y:-large_number;}
  else{direction_is_negative.y=0;inverse_direction.y=(direction.y>tolerance)?1/direction.y:large_number;}
  if(direction.z<0){direction_is_negative.z=1;inverse_direction.z=(direction.z<-tolerance)?1/direction.z:-large_number;}
  else{direction_is_negative.z=0;inverse_direction.z=(direction.z>tolerance)?1/direction.z:large_number;}
  computed_lazy_box_intersection_acceleration_data=true;
}

}
