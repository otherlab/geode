// Strongly typed ids for use in meshes
#pragma once

#include <geode/python/from_python.h>
#include <geode/python/to_python.h>
#include <geode/vector/Vector.h>
namespace geode {

using std::ostream;
using std::numeric_limits;

// Special id values
const int invalid_id = numeric_limits<int>::min();
const int erased_id = numeric_limits<int>::max();

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
  GEODE_ONLY_PYTHON(GEODE_REMOVE_PARENS(templates) static inline PyObject* to_python(GEODE_REMOVE_PARENS(full_name) i) { return to_python(i.id); }) \
  GEODE_ONLY_PYTHON(template<GEODE_REMOVE_PARENS(template_args)> struct FromPython<GEODE_REMOVE_PARENS(full_name)>{static GEODE_REMOVE_PARENS(full_name) convert(PyObject* o) { return GEODE_REMOVE_PARENS(full_name)(FromPython<int>::convert(o)); }};) \
  GEODE_REMOVE_PARENS(templates) static inline ostream& operator<<(ostream& output, GEODE_REMOVE_PARENS(full_name) i) { return output<<i.id; }

#define GEODE_DEFINE_TEMPLATE_ID(Name, decl, spec, use)\
  GEODE_DEFINE_ID_INTERNAL(Name, (Name<GEODE_REMOVE_PARENS(use)>), (template<GEODE_REMOVE_PARENS(decl)>), spec)

#define GEODE_DEFINE_ID(Name)\
  GEODE_DEFINE_ID_INTERNAL(Name, (Name), (), ())

GEODE_DEFINE_ID(VertexId)
GEODE_DEFINE_ID(HalfedgeId)
GEODE_DEFINE_ID(EdgeId)
GEODE_DEFINE_ID(FaceId)
GEODE_DEFINE_TEMPLATE_ID(PropertyId, (class T, class Id, bool Fancy=false), (class T, class Id, bool Fancy), (T, Id, Fancy))
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,VertexId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,FaceId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,HalfedgeId)

template<class Id> struct IdIter {
  Id i;
  IdIter(Id i) : i(i) {}
  void operator++() { i.id++; }
  bool operator!=(IdIter o) const { return i!=o.i; }
  Id operator*() const { return i; }
  IdIter operator+(int d) const { return Id(i.id+d);}
  IdIter operator-(int d) const { return Id(i.id-d);}
};

}
