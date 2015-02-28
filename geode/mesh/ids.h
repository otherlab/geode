// Strongly typed ids for use in meshes
#pragma once

#include <geode/python/numpy-types.h>
#include <geode/python/from_python.h>
#include <geode/python/to_python.h>
#include <geode/vector/Vector.h>

namespace geode {

using std::ostream;
using std::numeric_limits;

// Special id values
const int invalid_id = numeric_limits<int>::min();
const int erased_id = numeric_limits<int>::max();

// Special property ID values (all values <100 are reserved for special use and
// won't be used if not explicitly requested)
const int vertex_position_id = 0;
const int vertex_color_id = 1;
const int vertex_texcoord_id = 2;

const int face_color_id = 1;

const int halfedge_color_id = 1;
const int halfedge_texcoord_id = 2;

#define GEODE_DEFINE_ID_INTERNAL(Name, full_name, templates, template_args) \
  GEODE_REMOVE_PARENS(templates) struct Name { \
    int id; \
    Name() : id(invalid_id) {} \
    explicit Name(int id) : id(id) {} \
    int idx() const { return id; } \
    bool valid() const { return id!=invalid_id; } \
    bool operator==(Name i) const { return id==i.id; } \
    bool operator!=(Name i) const { return id!=i.id; } \
    bool operator< (Name i) const { return id< i.id; } \
    bool operator<=(Name i) const { return id<=i.id; } \
    bool operator> (Name i) const { return id> i.id; } \
    bool operator>=(Name i) const { return id>=i.id; } \
    explicit operator int() const { return id; } \
  }; \
  template<GEODE_REMOVE_PARENS(template_args)> struct is_packed_pod<GEODE_REMOVE_PARENS(full_name)> : mpl::true_ {}; \
  GEODE_REMOVE_PARENS(templates) GEODE_UNUSED static inline ostream& \
  operator<<(ostream& output, GEODE_REMOVE_PARENS(full_name) i) { return output<<((#Name)[0])<<i.id; }
// When we print ids, we use first character of Name as a prefix to make it easier to differentiate ids
// (i.g. V0 for VertexId(0), E0 for EdgeId(0), vs 0 for int(0))

#define GEODE_DEFINE_ID_CONVERSIONS(Name, full_name, templates, template_args) \
  GEODE_REMOVE_PARENS(templates) GEODE_UNUSED static inline PyObject* \
  to_python(GEODE_REMOVE_PARENS(full_name) i) { return to_python(i.id); } \
  namespace { \
  template<GEODE_REMOVE_PARENS(template_args)> struct NumpyIsScalar<GEODE_REMOVE_PARENS(full_name)>:public mpl::true_{};\
  template<GEODE_REMOVE_PARENS(template_args)> struct NumpyScalar<GEODE_REMOVE_PARENS(full_name)>{enum{value=NPY_INT};};\
  } \
  template<GEODE_REMOVE_PARENS(template_args)> struct FromPython<GEODE_REMOVE_PARENS(full_name)>{static GEODE_REMOVE_PARENS(full_name) convert(PyObject* o) { return GEODE_REMOVE_PARENS(full_name)(FromPython<int>::convert(o)); }};

#define GEODE_DEFINE_ID(Name)\
  GEODE_DEFINE_ID_INTERNAL(Name, (Name), (), ()) \
  GEODE_ONLY_PYTHON(GEODE_DEFINE_ID_CONVERSIONS(Name, (Name), (), ()))

GEODE_DEFINE_ID(VertexId)
GEODE_DEFINE_ID(HalfedgeId)
GEODE_DEFINE_ID(EdgeId)
GEODE_DEFINE_ID(FaceId)
GEODE_DEFINE_ID(BorderId)
GEODE_DEFINE_ID(ComponentId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,VertexId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,FaceId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,HalfedgeId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,VertexId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,FaceId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,HalfedgeId)
GEODE_DEFINE_ID_INTERNAL(FieldId, (FieldId<T,Id>), (template<class T,class Id>), (class T,class Id))

#ifdef GEODE_PYTHON

class PyFieldId : public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;

  const int id;
  const type_info* type;
  const enum Primitive {Vertex, Face, Halfedge} prim;

protected:
  template<class T> PyFieldId(FieldId<T,VertexId> id)   : id(id.id), type(&typeid(T)), prim(Vertex) {}
  template<class T> PyFieldId(FieldId<T,FaceId> id)     : id(id.id), type(&typeid(T)), prim(Face) {}
  template<class T> PyFieldId(FieldId<T,HalfedgeId> id) : id(id.id), type(&typeid(T)), prim(Halfedge) {}

  // making stuff from python without access to a type
  PyFieldId(Primitive prim, int id): id(id), type(NULL), prim(prim) {}
};

template<class T, class Id> static inline PyObject* to_python(FieldId<T,Id> i) {
  return to_python(new_<PyFieldId>(i));
}

#endif

template<class Id> struct IdIter {
  Id i;
  IdIter() = default;
  IdIter(Id i) : i(i) {}
  IdIter &operator++() { i.id++; return *this; }
  IdIter operator++(int) { IdIter<Id> old(*this); i.id++; return old; } // postfix
  bool operator!=(IdIter o) const { return i!=o.i; }
  bool operator==(IdIter o) const { return i==o.i; }
  Id operator*() const { return i; }
  IdIter operator+(int d) const { return Id(i.id+d);}
  IdIter operator-(int d) const { return Id(i.id-d);}
  int operator-(IdIter o) const { return i.id-o.i.id; }
};

template<class Id> Range<IdIter<Id>> id_range(const Id min, const Id end) {
  assert(min == end || min.idx() <= end.idx()); // Catch unbounded ranges
  return range(IdIter<Id>(min),IdIter<Id>(end));
}
template<class Id> Range<IdIter<Id>> id_range(const int n) { return id_range(Id(0),Id(n)); }
template<class Id> Range<IdIter<Id>> id_range(const int lo, const int hi) { return id_range(Id(lo),Id(hi)); }

#ifdef OTHER_PYTHON
template<class Id> static inline PyObject* to_python(IdIter<Id> i) { return to_python(i.i); }
template<class Id> struct FromPython<IdIter<Id>> {
static IdIter<Id> convert(PyObject* o) { return IdIter<Id>(FromPython<Id>::convert(o)); }};
#endif

}
