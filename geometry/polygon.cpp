#include "polygon.h"

#include <other/core/vector/Matrix.h>

#include <tr1/unordered_set>
#include <other/core/utility/Hasher.h>

#include <other/core/utility/stl.h>
#include <other/core/vector/Vector2d.h>
#include <other/core/vector/normalize.h>
#include <other/core/geometry/Box.h>
#include <other/core/geometry/Segment2d.h>
#include <other/core/array/NestedArray.h>
#include <other/core/array/Array.h>
#include <other/core/array/RawArray.h>

namespace other {
  
  Box<Vector<real,2> > bounding_box(Polygon const &poly) {
    Box<Vector<real,2> > box;
    for (int i = 0; i < (int) poly.size(); ++i) {
      box.enlarge(poly[i]);
    }    
    return box;
  }

  Box<Vector<real,2> > bounding_box(Polygons const &polys) {
    Box<Vector<real,2> > box;
    for (int i = 0; i < (int) polys.size(); ++i) {
      box.enlarge(bounding_box(polys[i]));
    }    
    return box;
  }
  
  Polygons polygons_from_index_list(Array<Vector<real,2> > const &positions, NestedArray<const int> indices) {
    Polygons polys;
    for (int i = 0; i < indices.size(); ++i) {
      polys.push_back(polygon_from_index_list(positions, indices[i]));
    }
    
    return polys;
  }
  
  Polygon polygon_from_index_list(Array<Vector<real,2> > const &positions, RawArray<const int> indices) {
    Polygon poly;
    
    for (int i = 0; i < indices.size(); ++i) {
      poly.push_back(positions[indices[i]]);
    }
    
    return poly;
  }

  // compute signed distance of non-overlapping (for example result of union) shape
  real polygon_area(Polygons const &polys) {
    real area = 0.;
    for (Polygon const &poly : polys) {
      area += polygon_area(poly);
    }
    return area;
  }

  // compute signed area of polygon
  real polygon_area(Polygon const &poly) {
    double area = 0;
    int n = (int)poly.size();
    if (n == 0) 
      return 0;
    int lim = (poly[n-1] == poly[0]) ? n-1 : n;
    for (int i = 0; i < lim; i++)
      area += cross(poly[i], poly[(i+1)%n]);
    return 0.5 * area;    
  }

  real polyline_length(Polygon const &poly) {
    double l = 0;
    int n = (int)poly.size();
    for (int i = 0; i < n-1; i++) {
      l += (poly[i] - poly[i+1]).magnitude();
    }
    
    return l;
  }
  
  
  real polygon_length(Polygon const &poly) {
    double l = 0;
    int n = (int)poly.size();
    for (int i = n-1, j =  0; j < n; i = j++) {
      l += (poly[i] - poly[j]).magnitude();
    }
    
    return l;
  }
  
  Polygon resample_polygon(Polygon poly, double maximum_edge_length) {
    typedef Vector<real,2> TV;
    
    if (poly.empty())
      return poly;
    
    // make this polygon explicitly closed
    poly.push_back(poly.front());
    
    OTHER_ASSERT(maximum_edge_length > 0);
    
    double sme = maximum_edge_length * maximum_edge_length;
    
    //std::cout << format("resampling polygon of length %f (%d points) using l = %f.", polygon_length(poly), poly.size(), maximum_edge_length) << std::endl;
    
    // make no edge longer than the maximum
    Polygon newpoly;
    bool too_far = false;
    int i = 0;
    while (i != (int)poly.size()) {
      
      if (too_far) {
        // compute point on this edge at the right distance
        TV dir = (poly[i] - newpoly.back()).normalized();
        newpoly.push_back(newpoly.back() + dir * maximum_edge_length);
      } else {
        // go to next segment
        newpoly.push_back(poly[i]);
        i++;
      }
      
      too_far = sme < (newpoly.back() - poly[i]).sqr_magnitude();
      
    }
    
    // remove the explicit closure from the resampled polygon
    assert(newpoly.front() == newpoly.back());
    newpoly.pop_back();
  
    return newpoly;
  }
  
  bool inside_polygon(Vector<real,2> const &p, Polygon const &poly) {
    int count = 0;
    real far = 1e3 * bounding_box(poly).sizes().max();
    Vector<real,2> outside = poly[0] + vec(far,far); // TODO: make random? 
    Segment<Vector<real,2> > S(p,outside);
    for (int i = 0, j = (int) poly.size()-1; i < (int) poly.size(); j = i++) {
      count += S.segment_segment_intersection(Segment<Vector<real,2> >(poly[i], poly[j])); 
    }
    return count & 1;    
  }
  
  // find a point inside the shape defined by polys, and inside the contour poly
  Vector<real,2> point_inside_polygon_component(Polygon const &poly, Polygons const &polys) {

    //std::cout << "point in polygon with " << poly.size() << " vertices, area = " << polygon_area(poly) << ", " << polys.size() << " polygons defining shape. " << std::endl;
    
    real drel = 1e-3;
    real dbox = bounding_box(polys).sizes().min();

    bool negative = polygon_area(poly) < 0.;
    
    // find the negative polygons to avoid
    std::vector<int> neg_polys;
    for (int i = 0; i < (int) polys.size(); ++i) {
      if (polygon_area(polys[i]) < 0) {
        neg_polys.push_back(i);
      }
    }
    
    do {   
      
      //std::cout << "drel = " << drel << std::endl;
      
      real d = dbox * drel;
      
      // try all edges
      for (int i = 0; i < (int) poly.size(); ++i) {
      
        // find a candidate point
        int i2 = (i + 1) % poly.size();
        Vector<real,2> mid = .5 * (poly[i] + poly[i2]);
        Vector<real,2> normal = normalized(poly[i2]-poly[i]).rotate_left_90();
        if (negative)
          normal *= -1;
        
        Vector<real,2> p = mid + d * normal;
              
        // make sure the point is actually inside our polygon
        if (!inside_polygon(p, poly)) {
          //std::cout << "point off edge " << i << " not in shape." << std::endl;
          continue;
        }
        
        // make sure the point is also inside the total shape
        bool no = false;
        for (int j = 0; j < (int) neg_polys.size(); ++j) {
          if (inside_polygon(p, polys[neg_polys[j]])) {
            //std::cout << "point off edge " << i << " in negative polygon " << j << " with " << polys[neg_polys[j]].size() << " vertices, area = " << polygon_area(polys[neg_polys[j]]) << std::endl;
            no = true;
            break;
          }
        }
          
        if (!no) 
          return p;
      }

      // maybe closer?
      drel /= 10.;
      
    } while (drel > 1e-10);
    
    std::cout << "Could not find any point inside contour " << poly << " and inside shape " << polys << std::endl;
    OTHER_ASSERT(false);
    return Vector<real,2>(0.,0.);
  } 
  
  
  
  Polygon polygon_simplify(Polygon const &poly, real max_angle_deg, real max_dist) {
    double mincos = cos(max_angle_deg / 180. * M_PI);
    double sqr_min_length = sqr(max_dist);
    
    // Repeatedly simplify until nothing changes
    Polygon result = poly;
    bool changed = true;
    while (changed) {
      changed = false;
      
      //std::cout << "simplifying polygon, currently " << result.size() << " nodes." << std::endl;
      
      // If u-v-w is collinear, remove v
      Polygon tmp;
      int m = (int)result.size();
      for (int i = 0; i < m; ++i) {
        int h = (i - 1 + m) % m;
        int j = (i + 1) % m;
        Vector<real,2> a = result[i]-result[h], b = result[j]-result[i];
        
        if (dot(a.normalized(), b.normalized()) < mincos) {
          tmp.push_back(result[j]);
        } else {
          changed = true;
        }
      }
            
      if (changed) {
        result = tmp;
        continue;
      }
      
      // collapse short edges
      tmp.clear();
      for (int i = 0; i < m; ++i) {
        int h = (i - 1 + m) % m;
        int j = (i + 1) % m;
        int k = (i + 2) % m;
          
        if ((result[i] - result[j]).sqr_magnitude() < sqr_min_length) {
          // check which end point to leave intact
          Vector<real,2> hi = (result[i] - result[h]).normalized();
          Vector<real,2> hj = (result[j] - result[h]).normalized();
          Vector<real,2> ik = (result[k] - result[i]).normalized();
          Vector<real,2> jk = (result[k] - result[j]).normalized();
          
          if (dot(hi,hj) > dot(ik,jk)) {
            // it's better to drop i
          } else {
            // it's better to drop j
            tmp.push_back(result[i]);
            i++;
          }
          
          changed = true;
        } else {
          tmp.push_back(result[i]);
        }
      }
      
      if (changed) {
        result = tmp;
      }
        
    }
    return result;
  }
  
  
  
  
  Tuple<Polygon, std::vector<int> > offset_polygon_with_correspondence(Polygon const &poly, real offset, real maxangle_deg, real minangle_deg) {
    OTHER_ASSERT(poly.size() > 1);
    
    real minangle = minangle_deg / 180. * M_PI;
    real maxangle = maxangle_deg / 180. * M_PI;
    
    int sign;
    if (offset < 0) {
      sign = -1;
    } else {
      sign = 1;
    }
    
    Polygon offset_poly;
    std::vector<int> correspondence;
    correspondence.reserve(poly.size());
    
    std::tr1::unordered_set<Vector<int,2>, Hasher> merged;
    
    for (int i = 0; i < (int)poly.size(); ++i) {
      
      Vector<real,2> const &last = poly[(i+poly.size()-1)%poly.size()];
      Vector<real,2> const &ours = poly[i];
      Vector<real,2> const &next = poly[(i+1)%poly.size()];
      
      Segment<Vector<real,2> > s0(last,ours);
      Segment<Vector<real,2> > s1(ours,next);
      
      Vector<real, 2> n0 = s0.normal();
      Vector<real, 2> n1 = s1.normal();
      
      real angle = -oriented_angle_between(s0.vector().normalized(), s1.vector().normalized());
      
      if (sign * angle > -minangle) {
        
        // the arc is (significantly) contracted 
        
        // add only one point at the intersection of the displaced segments
        Segment<Vector<real,2> > s0l(s0.x0 + offset * n0, s0.x1 + offset * n0);
        Segment<Vector<real,2> > s1l(s1.x0 + offset * n1, s1.x1 + offset * n1);
        real t0, t1;
        
        if (fabs(angle) < 0.01 || fabs(fabs(angle)-M_PI) < 0.01) {
          
          t0 = 1.;
          
        } else {
          
          s0l.segment_line_intersection(s1l.x0, n1, t0);
          s1l.segment_line_intersection(s0l.x0, n0, t1);
          
          // TODO: walk further here to avoid all local self-intersecions
          if (t0 < 0 || t1 > 1) {
            //std::cout << "WARNING: local self-intersection while making sausages (t0 = " << t0 << ", t1 = " << t1 << ")" << std::endl;
            if (t0 < 0)
              t0 = 0;
          }
          
        }
        
        Vector<real,2> p = s0l.point_from_barycentric_coordinates(t0);
        
        offset_poly.push_back(p);
        correspondence.push_back(i);
        
      } else {
        
        // the arc is possibly split into several segments
        int segments = (int) (fabs(angle) / maxangle);
        
        // make the first point 
        offset_poly.push_back(s0.x1 + offset * n0);
        
        if (segments > 0) {
          typedef Matrix<real,2,2> Mat22;
          Mat22 R = Mat22::rotation_matrix(-angle / (segments + 1));
          Vector<real,2> n = n0;
          
          // make the intermediate points 
          for (int k = 0; k < segments; ++k) {
            n = R * n;
            offset_poly.push_back(s0.x1 + offset * n);
          }
        }
        
        // make the last point
        offset_poly.push_back(s1.x0 + offset * n1);
        
        // all points we created belong to center point i
        while (correspondence.size() < offset_poly.size())
          correspondence.push_back(i);
      }       
    } 
    
    // walk over the offset polygon and get rid of local self-intersections by deleting points
    int m = (int)poly.size();
    int mo = (int)offset_poly.size();
    
    bool changed;
    int iter = 0;
    do {
      iter++;
      changed = false;
      
      //std::cout << "cleaning polygon with " << m << " points, iteration " << iter << std::endl;

      for (int i = 0; i < mo; ++i) {
        
        // ignore deleted points
        if (correspondence[i] == -1) 
          continue;
        
        // remember whether the current poly point is assigned another point prior to this one 
        // we cannot delete the first point -- only redirect it, we can delete all the others.
        bool can_delete = false;
        
        // find last non-deleted point
        int ilast = i;
        do {
          ilast = (ilast-1+mo)%mo;
        } while (correspondence[ilast] == -1);

        // find next non-deleted point
        int inext = i;
        do {
          inext = (inext+1)%mo;
        } while (correspondence[inext] == -1);
        
        // if last point is assigned to same base point, we can delete our point if we want to
        if (correspondence[ilast] == correspondence[i])
          can_delete = true;
        
        // this point was created by ci-1,ci and ci,ci+1

        Vector<real,2> p = offset_poly[i];
        int ci = correspondence[i];
        int cim1 = (ci-1 + m) % m;
        int cim2 = (ci-2 + m) % m;
        int cip1 = (ci+1) % m;
        int cip2 = (ci+2) % m;
        
        // check stuff that would lead us to merge with prior
        if (offset_poly[ilast] != offset_poly[i] && !merged.count(vec(i,ilast))) {

          bool merge = false;
          Vector<real,2> pnew = p;
          
          // check if it conflicts with ci-2,ci-1
          Segment<Vector<real,2> > sm2(poly[cim2], poly[cim1]);
          real d = sm2.distance(p);
          if (d < offset) {
            merge = true;
            //std::cout << "point conflicting with last segment " << cim2 << "--" << cim1 << " (d/offset = " << d / offset << "): " << i << " (poly " << ci << ", last " << ilast << ", next " << inext << ", can delete: " << can_delete << ")" << std::endl;
            pnew = offset_poly[ilast];
            merged.insert(vec(i, ilast));
          } else {
            
            int cil = correspondence[ilast];
            Segment<Vector<real,2> > sl(poly[cil], offset_poly[ilast]);
            Segment<Vector<real,2> > st(poly[ci], offset_poly[i]);
            
            if (cil != ci && sl.segment_segment_intersection(st)) {
              // the two connections intersect
              //std::cout << "connectors " << i << "->" << ci << " and " << ilast << "->" << cil << " intersect." << std::endl;
              
              /*
              // get the two outer generating segments
              Segment<Vector<real,2> > sl(poly[cil], poly[(cil-1+mo)%mo]);
              Segment<Vector<real,2> > st(poly[ci], poly[cip1]);
              
              real t; 
              if () {
              }
              */
              
              pnew = offset_poly[ilast];
              
              merge = true;
            }
             
          }
          
          if (merge) {
            changed = true;
            if (can_delete) {
              correspondence[i] = -1;
            } else {
              offset_poly[i] = pnew;
            }
            continue;
          }
        }
        
        // check stuff that would lead us to merge with next
        if (offset_poly[inext] != offset_poly[i] && !merged.count(vec(i,inext))) {
          bool merge = false; 
          Vector<real,2> pnew = p;

          // check if it conflicts with ci+1,ci+2
          Segment<Vector<real,2> > sp2(poly[cip1], poly[cip2]);
          real d = sp2.distance(p);
          if (d < offset) {          
            merge = true;
            //std::cout << "point conflicting with next segment " << cip1 << "--" << cip2 << " (d/offset = " << d / offset << "): " << i << " (poly " << ci << ", last " << ilast << ", next " << inext << ", can delete: " << can_delete << ")" << std::endl;
            pnew = offset_poly[inext];
            merged.insert(vec(i,inext));
          }

          if (merge) {
            changed = true;
            if (can_delete) {
              correspondence[i] = -1;
            } else {
              offset_poly[i] = pnew;
            }
            continue;
          }
        }
      }
            
    } while (changed);
    
    // clean offset_poly and correspondence (all points corresponding to -1 are removed)
    Polygon new_opoly;
    new_opoly.reserve(offset_poly.size());
    std::vector<int> new_corr;
    new_corr.reserve(correspondence.size());
    for (int i = 0; i < (int)offset_poly.size(); ++i) {
      if (correspondence[i] == -1) {
        continue;
      } else {
        new_opoly.push_back(offset_poly[i]);
        new_corr.push_back(correspondence[i]);
      }

    }
    
    return Tuple<Polygon, std::vector<int> >(new_opoly, new_corr);
  }
  
}
