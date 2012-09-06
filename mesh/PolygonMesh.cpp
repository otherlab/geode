//#####################################################################
// Class PolygonMesh
//#####################################################################
#include <other/core/mesh/PolygonMesh.h>
#include <other/core/mesh/SegmentMesh.h>
#include <other/core/mesh/TriangleMesh.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/const_cast.h>
#include <other/core/python/Class.h>
namespace other {

OTHER_DEFINE_TYPE(PolygonMesh)

PolygonMesh::PolygonMesh(Array<const int> counts, Array<const int> vertices)
  : counts(counts)
  , vertices(vertices)
  , node_count(0)
  , half_edge_count(counts.sum()) {
  // Assert validity and compute counts
  OTHER_ASSERT(half_edge_count==vertices.size());
  for (int i=0;i<counts.size();i++)
    OTHER_ASSERT(counts[i]>=3); // Polygons must be at least triangles
  for (int i=0;i<vertices.size();i++) {
    OTHER_ASSERT(vertices[i]>=0);
    const_cast_(node_count) = max(node_count,vertices[i]+1);
  }
}

PolygonMesh::~PolygonMesh() {}

Ref<SegmentMesh> PolygonMesh::segment_mesh() const {
  if (!segment_mesh_) {
    Hashtable<Vector<int,2> > hash;
    Array<Vector<int,2> > segments;
    int offset = 0;
    for (int p=0;p<counts.size();p++) {
      for (int i=0,j=counts[p]-1;i<counts[p];j=i,i++) {
        Vector<int,2> segment=vec(vertices[offset+i],vertices[offset+j]).sorted();
        if(hash.set(segment)) segments.append(segment);
      }
      offset += counts[p];
    }
    segment_mesh_=new_<SegmentMesh>(segments);
  }
  return ref(segment_mesh_);
}

Ref<TriangleMesh> PolygonMesh::triangle_mesh() const {
  if (!triangle_mesh_) {
    Array<Vector<int,3> > triangles(half_edge_count-2*counts.size());
    int offset=0, t=0;
    for (int p=0;p<counts.size();p++) {
      for (int i=0;i<counts[p]-2;i++)
        triangles[t++].set(vertices[offset],vertices[offset+i+1],vertices[offset+i+2]);
      offset+=counts[p];
    }
    triangle_mesh_=new_<TriangleMesh>(triangles);
  }
  return ref(triangle_mesh_);
}

}
using namespace other;

void wrap_polygon_mesh() {
  typedef PolygonMesh Self;
  Class<Self>("PolygonMesh")
    .OTHER_INIT(Array<const int>,Array<const int>)
    .OTHER_FIELD(counts)
    .OTHER_FIELD(vertices)
    .OTHER_METHOD(segment_mesh)
    .OTHER_METHOD(triangle_mesh)
    .OTHER_METHOD(nodes)
    ;
}
