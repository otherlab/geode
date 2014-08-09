#include <geode/geometry/polygon.h>
#include <geode/vector/Matrix.h>
#include <geode/math/constants.h>
#include <geode/utility/Hasher.h>
#include <geode/utility/str.h>
#include <geode/utility/tr1.h>
#include <geode/vector/Vector2d.h>
#include <geode/vector/normalize.h>
#include <geode/geometry/Box.h>
#include <geode/geometry/Segment.h>
#include <geode/mesh/SegmentSoup.h>
#include <geode/geometry/SimplexTree.h>
#include <geode/array/Array.h>
#include <geode/array/IndirectArray.h>
#include <geode/array/NdArray.h>
#include <geode/array/Nested.h>
#include <geode/geometry/Ray.h>
#include <geode/array/RawArray.h>
#include <geode/array/sort.h>
namespace geode {

typedef real T;

bool polygon_outlines_intersect(RawArray<const Vec2> p1, RawArray<const Vec2> p2, Ptr<SimplexTree<Vec2,1>> p2_tree) {
  if (!p2_tree) {
    auto mesh = to_segment_soup(make_nested(p2), false);
    p2_tree = new_<SimplexTree<Vec2,1>>(mesh.x, mesh.y, 4);
  }

  // walk p1, check for each segment's intersection
  for (int i = 0, j = p1.size()-1; i < (int)p1.size(); j = i++) {
    Vec2 dir = p1[i] - p1[j];
    Ray<Vec2> ray(p1[j], dir);
    ray.t_max = dir.magnitude();
    if (p2_tree->intersection(ray, 1e-10))
      return true;
  }
  return false;
}

Array<Vec2> polygon_from_index_list(RawArray<const Vec2> positions, RawArray<const int> indices) {
  return positions.subset(indices).copy();
}

Nested<Vec2> polygons_from_index_list(RawArray<const Vec2> positions, Nested<const int> indices) {
  Nested<Vec2> polys;
  polys.offsets = indices.offsets;
  polys.flat.copy(positions.subset(indices.flat));
  return polys;
}

T polygon_area(RawArray<const Vec2> poly) {
  const int n = poly.size();
  T area = 0;
  for (int i=n-1,j=0;j<n;i=j++)
    area += cross(poly[i],poly[j]);
  return .5*area;
}

T polygon_area(Nested<const Vec2> polys) {
  T area = 0;
  for (auto poly : polys)
    area += polygon_area(poly);
  return area;
}

T polygon_area_py(PyObject* object) {
  return polygon_area(polygons_from_python(object));
}

T open_polygon_length(RawArray<const Vec2> poly) {
  const int n = poly.size();
  T len = 0;
  for (int i=0;i<n-1;i++)
    len += magnitude(poly[i]-poly[i+1]);
  return len;
}

T polygon_length(RawArray<const Vec2> poly) {
  const int n = poly.size();
  T len = 0;
  for (int i=n-1,j=0;j<n;i=j++)
    len += magnitude(poly[i]-poly[j]);
  return len;
}

Array<Vec2> resample_polygon(RawArray<const Vec2> poly, const T maximum_edge_length) {
  GEODE_ASSERT(maximum_edge_length > 0);
  Array<Vec2> fine;
  if (!poly.size())
    return fine;
  fine.preallocate(poly.size());
  const T sqr_max_len = sqr(maximum_edge_length);
  Vec2 prev = poly.back();
  for (int i=0;i<poly.size();i++) {
    const T sqr_len = sqr_magnitude(prev-poly[i]);
    if (sqr_len > sqr_max_len) {
      // The edge is too long, so add some intermediate points
      const int extras = int(sqrt(sqr_len/sqr_max_len));
      const Vec2 step = (poly[i]-prev)/(extras+1);
      for (int j=0;j<extras;j++)
        fine.append(prev+(j+1)*step);
    }
    // Add the original point
    fine.append(poly[i]);
  }
  return fine;
}

bool inside_polygon(RawArray<const Vec2> poly, const Vec2 p) {
  int count = 0;
  T vfar = 1e3 * bounding_box(poly).sizes().max();
  Vec2 outside = poly[0] + vec(vfar,vfar); // TODO: make random?
  Segment<Vec2> S(p,outside);
  for (int i = 0, j = poly.size()-1; i < poly.size(); j = i++)
    count += segment_segment_distance(S,simplex(poly[i],poly[j]))==0;
  return count & 1;
}

// Find a point inside the shape defined by polys, and inside the contour poly
Vec2 point_inside_polygon_component(RawArray<const Vec2> poly, Nested<const Vec2> polys) {

  //std::cout << "point in polygon with " << poly.size() << " vertices, area = " << polygon_area(poly) << ", " << polys.size() << " polygons defining shape. " << std::endl;

  T drel = 1e-3;
  T dbox = bounding_box(polys).sizes().min();

  bool negative = polygon_area(poly) < 0.;

  // find the negative polygons to avoid
  Array<int> neg_polys;
  for (int i = 0; i < polys.size(); ++i)
    if (polygon_area(polys[i]) < 0)
      neg_polys.append(i);

  do {

    //std::cout << "drel = " << drel << std::endl;

    T d = dbox * drel;

    // try all edges
    for (int i = 0; i < poly.size(); ++i) {

      // find a candidate point
      int i2 = (i + 1) % poly.size();
      Vec2 mid = .5 * (poly[i] + poly[i2]);
      Vec2 normal = rotate_left_90(normalized(poly[i2]-poly[i]));
      if (negative)
        normal *= -1;

      Vec2 p = mid + d * normal;

      // make sure the point is actually inside our polygon
      if (!inside_polygon(poly,p)) {
        //std::cout << "point off edge " << i << " not in shape." << std::endl;
        continue;
      }

      // make sure the point is also inside the total shape
      bool no = false;
      for (int j = 0; j < neg_polys.size(); ++j) {
        if (inside_polygon(polys[neg_polys[j]],p)) {
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

  throw RuntimeError(format("point_inside_polygon_component: could not find a point inside contour and shape\n  contour = %s\n  shape = %s",str(poly),str(polys)));
}


Array<Vec2> polygon_simplify(RawArray<const Vec2> poly_, const T max_angle_deg, const T max_dist) {
  const T mincos = cos(pi/180*max_angle_deg);
  const T sqr_min_length = sqr(max_dist);
  Array<Vec2> poly = poly_.copy();
  Array<Vec2> tmp;

  // Repeatedly simplify until nothing changes
  for (;;) {
    bool changed = false;

    // If u-v-w is collinear, remove v
    tmp.clear();
    const int m = poly.size();
    for (int i = 0; i < m; ++i) {
      const int h = (i - 1 + m) % m;
      const int j = (i + 1) % m;
      const Vec2 a = poly[i]-poly[h],
                 b = poly[j]-poly[i];

      if (dot(normalized(a),normalized(b)) < mincos)
        tmp.append(poly[j]);
      else
        changed = true;
    }

    if (changed) {
      swap(poly,tmp);
      continue;
    }

    // Collapse short edges
    tmp.clear();
    for (int i = 0; i < m; ++i) {
      const int h = (i - 1 + m) % m;
      const int j = (i + 1) % m;
      const int k = (i + 2) % m;

      if ((poly[i] - poly[j]).sqr_magnitude() < sqr_min_length) {
        // Check which end point to leave intact
        Vec2 hi = normalized(poly[i] - poly[h]);
        Vec2 hj = normalized(poly[j] - poly[h]);
        Vec2 ik = normalized(poly[k] - poly[i]);
        Vec2 jk = normalized(poly[k] - poly[j]);

        if (dot(hi,hj) > dot(ik,jk)) {
          // It's better to drop i
        } else {
          // It's better to drop j
          tmp.append(poly[i]);
          i++;
        }

        changed = true;
      } else
        tmp.append(poly[i]);
    }

    if (changed)
      swap(poly,tmp);
    else
      break;
  }
  return poly;
}

// TODO: Move into Segment.h, possibly merging with other code
static bool segment_line_intersection(const Segment<Vector<T,2>>& segment, const Vector<T,2>& point_on_line,const Vector<T,2>& normal_of_line, T& interpolation_fraction) {
  const T denominator = dot(segment.x1-segment.x0,normal_of_line);
  if (!denominator) { // Parallel
    interpolation_fraction = FLT_MAX;
    return false;
  }
  interpolation_fraction = dot(point_on_line-segment.x0,normal_of_line)/denominator;
  return 0<=interpolation_fraction && interpolation_fraction<=1;
}

Tuple<Array<Vec2>,Array<int>> offset_polygon_with_correspondence(RawArray<const Vec2> poly, const T offset, const T maxangle_deg, const T minangle_deg) {
  GEODE_ASSERT(poly.size() > 1);

  const T minangle = pi/180*minangle_deg;
  const T maxangle = pi/180*maxangle_deg;
  const int sign = offset<0?-1:1;

  Array<Vec2> offset_poly;
  Array<int> correspondence;
  correspondence.preallocate(poly.size());

  unordered_set<Vector<int,2>,Hasher> merged;

  for (int i = 0; i < poly.size(); ++i) {
    Vec2 const &last = poly[(i+poly.size()-1)%poly.size()];
    Vec2 const &ours = poly[i];
    Vec2 const &next = poly[(i+1)%poly.size()];

    Segment<Vec2> s0(last,ours);
    Segment<Vec2> s1(ours,next);

    Vector<T, 2> n0 = s0.normal();
    Vector<T, 2> n1 = s1.normal();

    T angle = -angle_between(s0.vector().normalized(), s1.vector().normalized());

    if (sign * angle > -minangle) {

      // the arc is (significantly) contracted

      // add only one point at the intersection of the displaced segments
      Segment<Vec2> s0l(s0.x0 + offset * n0, s0.x1 + offset * n0);
      Segment<Vec2> s1l(s1.x0 + offset * n1, s1.x1 + offset * n1);
      T t0, t1;

      if (fabs(angle) < 0.01 || fabs(fabs(angle)-pi) < 0.01) {

        t0 = 1.;

      } else {

        segment_line_intersection(s0l,s1l.x0, n1, t0);
        segment_line_intersection(s1l,s0l.x0, n0, t1);

        // TODO: walk further here to avoid all local self-intersecions
        if (t0 < 0 || t1 > 1) {
          //std::cout << "WARNING: local self-intersection while making sausages (t0 = " << t0 << ", t1 = " << t1 << ")" << std::endl;
          if (t0 < 0)
            t0 = 0;
        }

      }

      Vec2 p = s0l.interpolate(t0);

      offset_poly.append(p);
      correspondence.append(i);

    } else {

      // the arc is possibly split into several segments
      int segments = int(abs(angle) / maxangle);

      // make the first point
      offset_poly.append(s0.x1 + offset * n0);

      if (segments > 0) {
        typedef Matrix<T,2,2> Mat22;
        Mat22 R = Mat22::rotation_matrix(-angle / (segments + 1));
        Vec2 n = n0;

        // make the intermediate points
        for (int k = 0; k < segments; ++k) {
          n = R * n;
          offset_poly.append(s0.x1 + offset * n);
        }
      }

      // make the last point
      offset_poly.append(s1.x0 + offset * n1);

      // all points we created belong to center point i
      while (correspondence.size() < offset_poly.size())
        correspondence.append(i);
    }
  }

  // walk over the offset polygon and get rid of local self-intersections by deleting points
  int m = poly.size();
  int mo = offset_poly.size();

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

      Vec2 p = offset_poly[i];
      int ci = correspondence[i];
      int cim1 = (ci-1 + m) % m;
      int cim2 = (ci-2 + m) % m;
      int cip1 = (ci+1) % m;
      int cip2 = (ci+2) % m;

      // check stuff that would lead us to merge with prior
      if (offset_poly[ilast] != offset_poly[i] && !merged.count(vec(i,ilast))) {

        bool merge = false;
        Vec2 pnew = p;

        // check if it conflicts with ci-2,ci-1
        Segment<Vec2> sm2(poly[cim2], poly[cim1]);
        T d = segment_point_distance(sm2,p);
        if (d < offset) {
          merge = true;
          //std::cout << "point conflicting with last segment " << cim2 << "--" << cim1 << " (d/offset = " << d / offset << "): " << i << " (poly " << ci << ", last " << ilast << ", next " << inext << ", can delete: " << can_delete << ")" << std::endl;
          pnew = offset_poly[ilast];
          merged.insert(vec(i, ilast));
        } else {

          int cil = correspondence[ilast];
          Segment<Vec2> sl(poly[cil], offset_poly[ilast]);
          Segment<Vec2> st(poly[ci], offset_poly[i]);

          if (cil != ci && segment_segment_distance(sl,st)==0) { // TODO: Not robust
            // the two connections intersect
            //std::cout << "connectors " << i << "->" << ci << " and " << ilast << "->" << cil << " intersect." << std::endl;

            /*
            // get the two outer generating segments
            Segment<Vec2> sl(poly[cil], poly[(cil-1+mo)%mo]);
            Segment<Vec2> st(poly[ci], poly[cip1]);

            T t;
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
        Vec2 pnew = p;

        // check if it conflicts with ci+1,ci+2
        Segment<Vec2> sp2(poly[cip1], poly[cip2]);
        T d = segment_point_distance(sp2,p);
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
  Array<Vec2> new_opoly;
  new_opoly.preallocate(offset_poly.size());
  Array<int> new_corr;
  new_corr.preallocate(correspondence.size());
  for (int i = 0; i < offset_poly.size(); ++i) {
    if (correspondence[i] == -1) {
      continue;
    } else {
      new_opoly.append(offset_poly[i]);
      new_corr.append(correspondence[i]);
    }

  }

  return tuple(new_opoly,new_corr);
}

Ref<SegmentSoup> nested_array_offsets_to_segment_soup(RawArray<const int> offsets, bool open) {
  GEODE_ASSERT(offsets.size() && !offsets[0]); // Not a complete check, but may catch a few bugs

  // empty?
  if (offsets.back() == 0) {
    return new_<SegmentSoup>(Array<Vector<int,2>>());
  }

  const int count = offsets.size()-1;
  Array<Vector<int,2>> segments(offsets.back()-count*open,uninit);
  if (open) {
    int s = 0;
    for (int p=0;p<count;p++)
      for (int i=offsets[p];i<offsets[p+1]-1;i++)
        segments[s++] = vec(i,i+1);
  } else {
    for (int i=0;i<offsets.back();i++)
      segments[i] = vec(i,i+1);
    // Fix wrap around segments
    for (int i=0;i<count;i++)
      segments[offsets[i+1]-1].y = offsets[i];
  }
  return new_<SegmentSoup>(segments);
}

Nested<const Vec2> polygons_from_python(PyObject* object) {
#ifdef GEODE_PYTHON
  try {
    const auto polys = from_python<NdArray<const Vec2>>(object);
    if (!polys.rank() || polys.rank()>2)
      throw TypeError(format("polygons_from_python: expected rank 1 or 2 array, got rank %d",polys.rank()));
    const int count = polys.rank()==1?1:polys.shape[0];
    Nested<const Vec2> nested;
    nested.offsets = (polys.shape.back()*arange(count+1)).copy();
    nested.flat = polys.flat;
    return nested;
  } catch (const exception&) {
    PyErr_Clear();
    // numpy conversion failed, try a nested array
    return from_python<Nested<const Vec2>>(object);
  }
#else
  GEODE_NOT_IMPLEMENTED("No python support");
#endif
}

// Compare two rotated views of a polygon checking for repeated points
static bool rotation_less(const int start0, const int start1, const RawArray<const Vec2>& poly) {
  assert(start0 != start1);
  const int n = poly.size();
  for(const int d : range(n)) {
    const auto p0 = poly[wrap(start0 + d, n)];
    const auto p1 = poly[wrap(start1 + d, n)];
    if(lex_less(p0, p1))
      return true;
    if(lex_less(p1, p0))
      return false;
  }
  return false;
}

Nested<Vec2> canonicalize_polygons(Nested<const Vec2> polys) {
  // Find the minimal point in each polygon under lexicographic order
  Array<int> mins(polys.size());
  for (int p=0;p<polys.size();p++) {
    const auto poly = polys[p];
    for (int i=1;i<poly.size();i++)
      if (rotation_less(i,mins[p],poly))
        mins[p] = i;
  }

  // Sort the polygons
  struct Order {
    Nested<const Vec2> polys;
    RawArray<const int> mins;
    Order(Nested<const Vec2> polys, RawArray<const int> mins)
      : polys(polys), mins(mins) {}
    bool operator()(int i,int j) const {
      return lex_less(polys(i,mins[i]),polys(j,mins[j]));
    }
  };
  Array<int> order = arange(polys.size()).copy();
  sort(order,Order(polys,mins));

  // Copy into new array
  Nested<Vec2> new_polys(polys.sizes().subset(order).copy(),uninit);
  for (int p=0;p<polys.size();p++) {
    const int base = mins[order[p]];
    const auto poly = polys[order[p]];
    const auto new_poly = new_polys[p];
    for (int i=0;i<poly.size();i++)
      new_poly[i] = poly[(i+base)%poly.size()];
  }
  return new_polys;
}

Array<int> closed_contours_next_from_offsets(RawArray<const int> offsets) {
  const int n = offsets.back();
  if(n == 0) // Catch empty arrays to avoid trying to iterate over an inverted range
    return Array<int>();
  Array<int> next(n,uninit);
  for (const int i : range(1,n))
    next[i-1] = i;
  for (const int j : range(offsets.size()-1)) {
    const int lo = offsets[j],
              hi = offsets[j+1];
    if (lo < hi)
      next[hi-1] = lo;
  }
  return next;
}

}
using namespace geode;

#include <geode/python/wrap.h>
#include <geode/python/function.h>
#include <geode/python/from_python.h>
#include <geode/vector/convert.h>

void wrap_polygon() {
  GEODE_FUNCTION_2(polygon_area,polygon_area_py)
  GEODE_FUNCTION(polygons_from_index_list)
  GEODE_FUNCTION(canonicalize_polygons)
}
