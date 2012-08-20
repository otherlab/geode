//#####################################################################
// Header mesh/forward
//#####################################################################
#pragma once

#include <other/core/array/forward.h>
namespace other{

class PolygonMesh;
class SegmentMesh;
class TriangleMesh;

template<int d> struct SimplexMesh;
template<> struct SimplexMesh<1>{typedef SegmentMesh type;};
template<> struct SimplexMesh<2>{typedef TriangleMesh type;};

}
