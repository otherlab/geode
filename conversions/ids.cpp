#include <geode/mesh/ids.h>
#include <geode/python/Class.h>
#include <geode/vector/convert.h>
#include <geode/python/pyrange.h>

namespace geode {

// Add numpy conversion support
#ifdef GEODE_PYTHON

GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,VertexId)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,HalfedgeId)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,FaceId)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,VertexId)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,HalfedgeId)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,FaceId)
NESTED_CONVERSIONS(HalfedgeId)

GEODE_DEFINE_TYPE(PyFieldId);

template<> GEODE_DEFINE_TYPE(PyRange<IdIter<VertexId>>);
template<> GEODE_DEFINE_TYPE(PyRange<IdIter<FaceId>>);
template<> GEODE_DEFINE_TYPE(PyRange<IdIter<HalfedgeId>>);
#endif

}

using namespace geode;

void wrap_ids() {
  GEODE_OBJECT(invalid_id);
  GEODE_OBJECT(erased_id);
  GEODE_OBJECT(vertex_position_id);
  GEODE_OBJECT(vertex_color_id);
  GEODE_OBJECT(vertex_texcoord_id);
  GEODE_OBJECT(face_color_id);
  GEODE_OBJECT(halfedge_color_id);
  GEODE_OBJECT(halfedge_texcoord_id);

  GEODE_PYTHON_RANGE(IdIter<VertexId>, "VertexIter")
  GEODE_PYTHON_RANGE(IdIter<FaceId>, "FaceIter")
  GEODE_PYTHON_RANGE(IdIter<HalfedgeId>, "HalfedgeIter")
}
