// Convenience functions for generating platonic solids

#include <geode/geometry/platonic.h>
#include <geode/mesh/TriangleSubdivision.h>
#include <geode/python/wrap.h>
#include <geode/vector/normalize.h>
namespace geode {

typedef real T;
typedef Vector<T,3> TV;
typedef Vector<int,3> IV;

Tuple<Ref<TriangleSoup>,Array<TV>> icosahedron_mesh() {
  static const T p = (1+sqrt(5))/2;
  static const IV tris[] = {IV(0,1,2),IV(0,2,4),IV(1,3,6),IV(0,3,1),IV(0,4,7),IV(2,5,8),IV(1,5,2),IV(1,6,5),IV(3,7,9),IV(0,7,3),IV(4,8,10),
                            IV(2,8,4),IV(5,6,11),IV(6,9,11),IV(3,9,6),IV(4,10,7),IV(5,11,8),IV(7,10,9),IV(8,11,10),IV(9,10,11)};
  static const TV X[] = {TV(1,p,0),TV(0,1,p),TV(p,0,1),TV(-1,p,0),TV(p,0,-1),TV(0,-1,p),TV(-p,-0,1),TV(0,1,-p),TV(1,-p,0),TV(-p,0,-1),TV(0,-1,-p),TV(-1,-p,0)};
  static const auto mesh = new_<TriangleSoup>(Array<const IV>(sizeof(tris)/sizeof(IV),tris,&*Ref<>(new_<Object>())));
  return tuple(mesh,RawArray<const TV>(sizeof(X)/sizeof(TV),X).copy());
}

Tuple<Ref<TriangleSoup>,Array<TV>> sphere_mesh(const int refinements, const TV center, const T radius) {
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

Ref<TriangleSoup> double_torus_mesh() {
  // This is N_1 from Basudeb Datta and Ashish Kumar Upadhyay, Degree-regular triangulations of the double-torus, originally from
  // F. H. Lutz, Triangulated manifolds with few vertices and vertex-transitive group actions".
  static const IV tris[] = {IV(3,5,4),IV(3,9,5),IV(7,9,8),IV(1,5,9),IV(1,9,7),IV(1,11,5),IV(2,8,10),IV(8,9,10),IV(0,10,11),IV(0,11,1),IV(0,1,2),IV(0,2,6),IV(2,10,6),IV(4,6,10),
                            IV(0,4,10),IV(0,8,4),IV(0,6,8),IV(6,7,8),IV(5,7,6),IV(4,5,6),IV(2,3,4),IV(2,4,8),IV(5,11,7),IV(1,7,3),IV(3,7,11),IV(1,3,2),IV(3,11,9),IV(9,11,10)};
  static const auto mesh = new_<TriangleSoup>(Array<const IV>(sizeof(tris)/sizeof(IV),tris,&*Ref<>(new_<Object>())));
  return mesh;
}

}
using namespace geode;

void wrap_platonic() {
  GEODE_FUNCTION(icosahedron_mesh)
  GEODE_FUNCTION_2(sphere_mesh_py,sphere_mesh)
  GEODE_FUNCTION(double_torus_mesh)
}
