// Strongly typed ids for use in meshes
#pragma once

#include <geode/utility/numpy.h>
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
  operator<<(ostream& output, GEODE_REMOVE_PARENS(full_name) i) { return output<<i.id; }

#define GEODE_DEFINE_ID_CONVERSIONS(Name, full_name, templates, template_args) \
  template<GEODE_REMOVE_PARENS(template_args)> struct NumpyIsScalar<GEODE_REMOVE_PARENS(full_name)> \
    : public mpl::true_{}; \
  template<GEODE_REMOVE_PARENS(template_args)> struct NumpyScalar<GEODE_REMOVE_PARENS(full_name)> { \
    enum {value=NPY_INT}; }; \

#define GEODE_DEFINE_ID(Name) \
  GEODE_DEFINE_ID_INTERNAL(Name, (Name), (), ()) \
  GEODE_DEFINE_ID_CONVERSIONS(Name, (Name), (), ())

GEODE_DEFINE_ID(VertexId)
GEODE_DEFINE_ID(HalfedgeId)
GEODE_DEFINE_ID(EdgeId)
GEODE_DEFINE_ID(FaceId)
GEODE_DEFINE_ID(BorderId)
GEODE_DEFINE_ID(ComponentId)
GEODE_DEFINE_ID_INTERNAL(FieldId, (FieldId<T,Id>), (template<class T,class Id>), (class T,class Id))

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

template<class Id> Range<IdIter<Id>> id_range(const int n) { return range(IdIter<Id>(Id(0)),IdIter<Id>(Id(n))); }

}
