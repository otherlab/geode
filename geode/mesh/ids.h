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

#define GEODE_DEFINE_ID(Name) \
  struct Name { \
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
  template<> struct is_packed_pod<Name> : mpl::true_ {}; \
  GEODE_ONLY_PYTHON(static inline PyObject* to_python(Name i) { return to_python(i.id); }) \
  GEODE_ONLY_PYTHON(template<> struct FromPython<Name>{static Name convert(PyObject* o) { return Name(FromPython<int>::convert(o)); }};) \
  static inline ostream& operator<<(ostream& output, Name i) { return output<<i.id; }
GEODE_DEFINE_ID(VertexId)
GEODE_DEFINE_ID(HalfedgeId)
GEODE_DEFINE_ID(EdgeId)
GEODE_DEFINE_ID(FaceId)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_CORE_EXPORT,3,VertexId)

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
