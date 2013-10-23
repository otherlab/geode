//#####################################################################
// Header mesh/forward
//#####################################################################
#pragma once

#include <geode/array/forward.h>
namespace geode {

class PolygonMesh;
class SegmentMesh;
class TriangleMesh;
class HalfedgeMesh;

template<int d> struct SimplexMesh;
template<> struct SimplexMesh<1>{typedef SegmentMesh type;};
template<> struct SimplexMesh<2>{typedef TriangleMesh type;};

}
