//#####################################################################
// Header mesh/forward
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
namespace other{

class PolygonSoup;
class SegmentSoup;
class TriangleSoup;
class HalfedgeMesh;

template<int d> struct SimplexMesh;
template<> struct SimplexMesh<1>{typedef SegmentSoup type;};
template<> struct SimplexMesh<2>{typedef TriangleSoup type;};

}
