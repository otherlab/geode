// Strongly typed ids for use in meshes
#pragma once

#include <other/core/python/from_python.h>
#include <other/core/python/to_python.h>
#include <other/core/vector/Vector.h>
namespace other {

using std::ostream;
using std::numeric_limits;

// Special id values
const int invalid_id = numeric_limits<int>::min();
const int deleted_id = numeric_limits<int>::max();

#define OTHER_DEFINE_ID(Name) \
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
  static inline PyObject* to_python(Name i) { return to_python(i.id); } \
  template<> struct FromPython<Name>{static Name convert(PyObject* o) { return Name(FromPython<int>::convert(o)); }}; \
  static inline ostream& operator<<(ostream& output, Name i) { return output<<i.id; }
OTHER_DEFINE_ID(VertexId)
OTHER_DEFINE_ID(HalfedgeId)
OTHER_DEFINE_ID(EdgeId)
OTHER_DEFINE_ID(FaceId)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_CORE_EXPORT,3,VertexId)

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
