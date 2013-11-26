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
  GEODE_REMOVE_PARENS(templates) static inline ostream& operator<<(ostream& output, GEODE_REMOVE_PARENS(full_name) i) { return output<<i.id; }

#define GEODE_DEFINE_ID_CONVERSIONS(Name, full_name, templates, template_args) \
  GEODE_REMOVE_PARENS(templates) static inline PyObject* to_python(GEODE_REMOVE_PARENS(full_name) i) { return to_python(i.id); } \
  namespace {\
  template<GEODE_REMOVE_PARENS(template_args)> struct NumpyIsScalar<GEODE_REMOVE_PARENS(full_name)>:public mpl::true_{};\
  template<GEODE_REMOVE_PARENS(template_args)> struct NumpyScalar<GEODE_REMOVE_PARENS(full_name)>{enum{value=NPY_INT};};\
  }\
  template<GEODE_REMOVE_PARENS(template_args)> struct FromPython<GEODE_REMOVE_PARENS(full_name)>{static GEODE_REMOVE_PARENS(full_name) convert(PyObject* o) { return GEODE_REMOVE_PARENS(full_name)(FromPython<int>::convert(o)); }};

#define GEODE_DEFINE_ID(Name)\
  GEODE_DEFINE_ID_INTERNAL(Name, (Name), (), ()) \
  GEODE_ONLY_PYTHON(GEODE_DEFINE_ID_CONVERSIONS(Name, (Name), (), ()))

GEODE_DEFINE_ID(VertexId)
GEODE_DEFINE_ID(HalfedgeId)
GEODE_DEFINE_ID(EdgeId)
GEODE_DEFINE_ID(FaceId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,VertexId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,FaceId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,2,HalfedgeId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,VertexId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,FaceId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,HalfedgeId)

GEODE_DEFINE_ID_INTERNAL(PropertyId, (PropertyId<T, Id, Fancy>), (template<class T, class Id, bool Fancy=false>), (class T, class Id, bool Fancy))

#ifdef GEODE_PYTHON

class PyPropertyId: public Object {
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  typedef Object Base;

  const int id;
  const string type_id;
  const enum {idVertex, idFace, idHalfedge} id_type;
  const bool fancy;

  template<class T, bool Fancy>
  PyPropertyId(PropertyId<T,VertexId,Fancy> const &id)
  : id(id.id), type_id(typeid(T).name()), id_type(idVertex), fancy(Fancy) {}
  template<class T, bool Fancy>
  PyPropertyId(PropertyId<T,FaceId,Fancy> const &id)
  : id(id.id), type_id(typeid(T).name()), id_type(idFace), fancy(Fancy) {}
  template<class T, bool Fancy>
  PyPropertyId(PropertyId<T,HalfedgeId,Fancy> const &id)
  : id(id.id), type_id(typeid(T).name()), id_type(idHalfedge), fancy(Fancy) {}
};

template<class T, class id_type, bool Fancy>
static inline PyObject* to_python(PropertyId<T,id_type,Fancy> const &i) {
  return to_python(new_<PyPropertyId>(i));
}

#endif

template<class Id> struct IdIter {
  Id i;
  IdIter(Id i) : i(i) {}
  IdIter &operator++() { i.id++; return *this; }
  IdIter operator++(int) { IdIter<Id> old(*this); i.id++; return old; } // postfix
  bool operator!=(IdIter o) const { return i!=o.i; }
  bool operator==(IdIter o) const { return i==o.i; }
  Id operator*() const { return i; }
  IdIter operator+(int d) const { return Id(i.id+d);}
  IdIter operator-(int d) const { return Id(i.id-d);}
};

#ifdef OTHER_PYTHON
template<class Id> static inline PyObject* to_python(IdIter<Id> i) { return to_python(i.i); }
template<class Id> struct FromPython<IdIter<Id>> {
static IdIter<Id> convert(PyObject* o) { return IdIter<Id>(FromPython<Id>::convert(o)); }};
#endif

}
