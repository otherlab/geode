// Convenience functions for generating platonic solids

#include <other/core/geometry/platonic.h>
#include <other/core/mesh/TriangleSubdivision.h>
#include <other/core/python/wrap.h>
#include <other/core/vector/normalize.h>
namespace other {

typedef real T;
typedef Vector<T,3> TV;
typedef Vector<int,3> IV;

Tuple<Ref<TriangleMesh>,Array<TV>> icosahedron_mesh() {
  static const T p = (1+sqrt(5))/2;
  static const IV tris[] = {IV(0,1,2),IV(0,2,4),IV(1,3,6),IV(0,3,1),IV(0,4,7),IV(2,5,8),IV(1,5,2),IV(1,6,5),IV(3,7,9),IV(0,7,3),IV(4,8,10),
                            IV(2,8,4),IV(5,6,11),IV(6,9,11),IV(3,9,6),IV(4,10,7),IV(5,11,8),IV(7,10,9),IV(8,11,10),IV(9,10,11)};
  static const TV X[] = {TV(1,p,0),TV(0,1,p),TV(p,0,1),TV(-1,p,0),TV(p,0,-1),TV(0,-1,p),TV(-p,-0,1),TV(0,1,-p),TV(1,-p,0),TV(-p,0,-1),TV(0,-1,-p),TV(-1,-p,0)};
  static const auto mesh = new_<TriangleMesh>(Array<const IV>(sizeof(tris)/sizeof(IV),tris,&*Ref<>(new_<Object>())));
  return tuple(mesh,RawArray<const TV>(sizeof(X)/sizeof(TV),X).copy());
}

Tuple<Ref<TriangleMesh>,Array<TV>> sphere_mesh(const int refinements, const TV center, const T radius) {
  auto mesh_X = icosahedron_mesh();
  for (int r=0;r<refinements;r++) {
    for (auto& x : mesh_X.y)
      normalize(x);
    const auto sub = new_<TriangleSubdivision>(mesh_X.x);
    mesh_X.x = sub->fine_mesh;
    mesh_X.y = sub->linear_subdivide(mesh_X.y);
  }
  for (auto& x : mesh_X.y)
    x = center+radius*normalized(x);
  return mesh_X;
}

}
using namespace other;

void wrap_platonic() {
  OTHER_FUNCTION(icosahedron_mesh)
  OTHER_FUNCTION_2(sphere_mesh_py,sphere_mesh)
}
