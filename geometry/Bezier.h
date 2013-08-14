#pragma once

#include <other/core/vector/Vector.h>
#include <other/core/structure/Tuple.h>
#include <other/core/geometry/Box.h>
#include <other/core/python/Object.h>
#include <other/core/python/Ref.h>
#include <other/core/math/constants.h>
#include <other/core/utility/tr1.h>
#include <map>
namespace other {

using std::map;
using std::pair;
using std::make_pair;

// Box<real> uses max < min to represent empty boxes. This can cause problems for closed Beziers.
// When specifying a range of t values on a closed Bezier that starts before and ends after the minimum/maximum t values (i.e. wraps around)
// the max value might be less than the min value. This class is used instead to avoid accidentally calling functions like size that wouldn't
// be meaningful in this case.
class InvertableBox {
 public:
  InvertableBox(real _begin, real _end) : begin(_begin), end(_end) {}
  real begin;
  real end;
};
OTHER_CORE_EXPORT PyObject* to_python(const InvertableBox& self);
template<> struct FromPython<InvertableBox>{OTHER_CORE_EXPORT static InvertableBox convert(PyObject* object);};

std::ostream& operator<<(std::ostream& os, const InvertableBox& ib);


template <int d> struct Knot : public Object{
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;
  typedef Vector<real,d> TV;
  TV pt;
  TV tangent_in;
  TV tangent_out;
  Knot(){}
  Knot(const Knot<d>& k):pt(k.pt),tangent_in(k.tangent_in),tangent_out(k.tangent_out){}
  Knot(const TV& p):pt(p),tangent_in(p),tangent_out(p){}
  Knot(const TV& p,const TV& tin,const TV& tout):pt(p),tangent_in(tin),tangent_out(tout){}
  inline Knot operator+(const TV& v){
    return Knot(pt+v,tangent_in+v,tangent_out+v);
  }
  inline void operator+=(const TV& v){
    pt+=v;
    tangent_in+=v;
    tangent_out+=v;
  }

  //equals operator ok via construction
  inline bool singular_in(){
    return tangent_in == pt;
  }
  inline bool singular_out(){
    return tangent_out == pt;
  }
  inline bool singular(){
    return singular_in() && singular_out();
  }

  inline bool colinear(){
    real delta = pi*1e-1;
    return dot((tangent_out-pt).normalized(),(tangent_in-pt).normalized())<(-1+delta);
  }

  inline bool symmetric(){
    real delta = 1e-5;
    return abs((tangent_in-pt).magnitude() - (tangent_out-pt).magnitude()) < delta;
  }

  inline bool symmetric(real delta){
    return abs((tangent_in-pt).magnitude() - (tangent_out-pt).magnitude()) < delta;
  }
};

template<int d> struct Span{
  Ref<Knot<d> > start, end;
  real start_t, end_t;
  Span(const Ref<Knot<d>> s, const Ref<Knot<d> > e, real st, real et):start(s),end(e),start_t(st),end_t(et){}
};

template<int d> class Bezier : public Object{
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;
private:
  typedef real T;
  typedef Vector<T,d> TV;
  Box<real> t_range;
  bool b_closed;
protected:
  OTHER_CORE_EXPORT Bezier();
  OTHER_CORE_EXPORT Bezier(const Bezier<d>& b);
public:
  map<real,Ref<Knot<d>>> knots;
  OTHER_CORE_EXPORT Array<TV> segment(const InvertableBox& range, int res) const ;
  OTHER_CORE_EXPORT Array<TV> alen_segment(const InvertableBox& range, int res) const ;
  OTHER_CORE_EXPORT Array<TV> evaluate(int res) const ;
  OTHER_CORE_EXPORT Array<TV> alen_evaluate(int res) const ;
  OTHER_CORE_EXPORT void append_knot(const TV& pt, TV tin = T(inf)*TV::ones(), TV tout = T(inf)*TV::ones());
  OTHER_CORE_EXPORT void insert_knot(const real t) ;
  OTHER_CORE_EXPORT Span<d> segment(real t) const ;
  OTHER_CORE_EXPORT TV point(real t) const;
  OTHER_CORE_EXPORT TV tangent(Span<d> const &seg, real t) const;
  OTHER_CORE_EXPORT TV tangent(real t) const;
  // left and right tangent are always the same, except at knots
  OTHER_CORE_EXPORT TV left_tangent(real t) const;
  OTHER_CORE_EXPORT TV right_tangent(real t) const;
  // mostly 0, except (maybe) at knots (in radians)
  OTHER_CORE_EXPORT real angle_at(real t) const;
  inline real t_max() const { return t_range.max; }
  inline real t_min() const { return t_range.min; }
  OTHER_CORE_EXPORT void close() ;
  OTHER_CORE_EXPORT void fuse_ends() ;
  OTHER_CORE_EXPORT void erase(real t) ;
  inline bool closed() const { return b_closed; }
  OTHER_CORE_EXPORT ~Bezier();

  OTHER_CORE_EXPORT Box<TV> bounding_box() const;

  OTHER_CORE_EXPORT void translate(const TV& t);

};

template<int d>
inline std::ostream &operator<<(std::ostream &os, Bezier<d> const &b) {
  return os << "bezier" << d << "d" << std::endl;
}

template<int d> struct PointSolve {
  const Vector<real,d> pt;
  const pair<real,Ref<Knot<d>> > k1, k2;
  PointSolve(const Vector<real,d> pt, const pair<real,Ref<Knot<d>>>& k1, const pair<real,Ref<Knot<d>> >& k2)
    : pt(pt), k1(k1), k2(k2) {}

  OTHER_CORE_EXPORT real distance(real t);
};

}
