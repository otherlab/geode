//
//
//  Tyson Brochu 2011
//

#pragma once

#include <other/core/utility/config.h>
#include <other/core/vector/Vector.h>
namespace other {

// True if the edges x01 and x23 intersect an odd or degenerate number of times during linear motion
OTHER_CORE_EXPORT bool edge_edge_collision_parity(const Vector<double,3>& x0old, const Vector<double,3>& x1old, const Vector<double,3>& x2old, const Vector<double,3>& x3old,
                                                  const Vector<double,3>& x0new, const Vector<double,3>& x1new, const Vector<double,3>& x2new, const Vector<double,3>& x3new);

// True if the point x0 intersects the triangle x123 an odd or degenerate number of times during linear motion
OTHER_CORE_EXPORT bool point_triangle_collision_parity(const Vector<double,3>& x0old, const Vector<double,3>& x1old, const Vector<double,3>& x2old, const Vector<double,3>& x3old,
                                                       const Vector<double,3>& x0new, const Vector<double,3>& x1new, const Vector<double,3>& x2new, const Vector<double,3>& x3new);
 
// True if edge x01 intersects triangle x234
OTHER_CORE_EXPORT bool edge_triangle_intersection(const Vector<double,3>& x0, const Vector<double,3>& x1, const Vector<double,3>& x2, const Vector<double,3>& x3, const Vector<double,3>& x4);

// Does triangle a012 intersect triangle b012?  Warning: This function is quite slow, and it is often faster to use edge_triangle_intersection directly.
OTHER_CORE_EXPORT bool triangle_triangle_intersection(const Vector<double,3>& a0, const Vector<double,3>& a1, const Vector<double,3>& a2,
                                                      const Vector<double,3>& b0, const Vector<double,3>& b1, const Vector<double,3>& b2);

}
