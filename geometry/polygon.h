#pragma once

#include <other/core/structure/Tuple.h>
#include <other/core/geometry/Segment2d.h>
#include <other/core/array/Array.h>
#include <vector>

namespace other {

class SegmentMesh;

typedef std::vector<Vector<real, 2> > Polygon;
typedef std::vector<Polygon> Polygons;

OTHER_CORE_EXPORT Box<Vector<real,2> > bounding_box(Polygon const &poly);
OTHER_CORE_EXPORT Box<Vector<real,2> > bounding_box(Polygons const &polys);

OTHER_CORE_EXPORT Polygon polygon_from_index_list(RawArray<const Vector<real,2> > const &positions, RawArray<const int> indices);
OTHER_CORE_EXPORT Polygons polygons_from_index_list(RawArray<const Vector<real,2> > const &positions, NestedArray<const int> indices);

// compute signed area of polygon(s)
OTHER_CORE_EXPORT real polygon_area(Polygons const &polys);
OTHER_CORE_EXPORT real polygon_area(Polygon const &poly);

// compute the length of an (open) polyline
OTHER_CORE_EXPORT real polyline_length(Polygon const &poly);

// compute the circumference of a closed polygon
OTHER_CORE_EXPORT real polygon_length(Polygon const &poly);

// enforce maximum edge length along the polygon
OTHER_CORE_EXPORT Polygon resample_polygon(Polygon poly, double maximum_edge_length);

// return whether the point is inside the polygon
OTHER_CORE_EXPORT bool inside_polygon(Vector<real,2> const &p, Polygon const &poly);

// find a point inside the shape defined by polys, and inside the contour poly
OTHER_CORE_EXPORT Vector<real,2> point_inside_polygon_component(Polygon const &poly, Polygons const &polys);

OTHER_CORE_EXPORT Tuple<Polygon, std::vector<int> > offset_polygon_with_correspondence(Polygon const &poly, real offset, real maxangle_deg = 20., real minangle_deg = 10.);

template<int d>
OTHER_CORE_EXPORT Tuple<Ref<SegmentMesh>, Array<Vector<real,d>>> to_segment_mesh(std::vector<std::vector<Vector<real,d>>> const &polys, bool open = false);

}
