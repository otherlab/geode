//#####################################################################
// Header mesh/forward
//#####################################################################
#pragma once

#include <geode/array/forward.h>
namespace geode {

class PolygonSoup;
class SegmentSoup;
class TriangleSoup;
class HalfedgeMesh;
class HalfedgeGraph;
class TriangleMesh;
class TriangleTopology;
class TriangleSubdivision;

template<int d> struct SimplexMesh;
template<> struct SimplexMesh<1>{typedef SegmentSoup type;};
template<> struct SimplexMesh<2>{typedef TriangleSoup type;};

}
