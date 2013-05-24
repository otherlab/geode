#include "Bezier.h"
#include <other/core/vector/Matrix4x4.h>
#include <other/core/vector/Matrix.h>
#include <other/core/utility/stl.h>
#include <other/core/python/Class.h>
#include <other/core/python/stl.h>
#include <other/core/geometry/polygon.h>
#include <iostream>

namespace other{

using std::cout;
using std::endl;

template<> OTHER_DEFINE_TYPE(Bezier<2>)
template<> OTHER_DEFINE_TYPE(Knot<2>)

static double interpolate(double v0, double v1, double v2, double v3, double t) {
  Vector<real,4> i(v0,v1,v2,v3);
  Matrix<real,4> mx( -1, 3, -3, 1,
                     3, -6, 3, 0,
                     -3,3, 0, 0,
                     1, 0,0, 0);

  Vector<real,4> a = mx*i; // transpose because of legacy constructor
  return a.w + t*(a.z + t*(a.y + t*a.x));
}

template<int d> static Vector<real,d> point(Vector<real,d> v0, Vector<real,d> v1, Vector<real,d> v2, Vector<real,d> v3, double t) {
  Vector<real,d> result;
  for (int i = 0; i < d; i++)
    result[i] = interpolate(v0[i],v1[i],v2[i],v3[i],t);
  return result;
}

template<int d> Array<Vector<real,d>> Bezier<d>::segment(const InvertableBox& range, int res) const{
  Array<Vector<real,d>> path;
  if(knots.size()<=1) return path;
  Vector<real,d> p1, p2, p3, p4;
  auto it = knots.upper_bound(range.begin);
  auto end = knots.lower_bound(range.end);

  if(end == knots.end()) end--;
  OTHER_ASSERT(it!=knots.end() && end!=knots.end());
  if(it!=knots.begin()) it--;

  while(it!=end){
    if(b_closed && it->first == t_max()){
      if(range.end == t_min() || range.end == t_max()) break;
      it = knots.begin();
    }
    p1 = it->second->pt;
    p2 = it->second->tangent_out;
    it++;
    if(it == knots.end()) it = knots.begin(); // wrap around at end

    p3 = it->second->tangent_in;
    p4 = it->second->pt;

    for(int j=0;j<res;j++){
      real t = j/(real)(res); // [0,1)
      path.append(other::point(p1,p2,p3,p4,t));
    }
  }
  path.append(end->second->pt);

  return path;
}

template<int d> Array<Vector<real,d>> Bezier<d>::alen_segment(const InvertableBox& range, int res) const{
  Array<Vector<real,d>> path, d_path;
  d_path = segment(range,200);
  real len = open_polygon_length(d_path);
  real step = len/res;
  real tstep = .001; //TODO: multi-resolution/binary-search?

  Vector<real,d> p1, p2, p3, p4;
  auto it = knots.upper_bound(range.begin);
  auto end = knots.lower_bound(range.end);
  //Replaced this to support wrapping around: if(end == knots.end()) end--;
  if(end == knots.end()) end = knots.begin();

  OTHER_ASSERT(it!=knots.end() && end!=knots.end());

  //Replaced this to support wrapping around: if(it->first != t_min()) it--;
  if(it == knots.begin()) it = knots.end();
  it--;

  path.append(it->second->pt);
  real t = tstep;
  real dst = tstep;

  do {
    p1 = it->second->pt;
    p2 = it->second->tangent_out;
    it++;
    if(it == knots.end()) {
      OTHER_ASSERT(range.end < range.begin, "Attempting to wrap around, but range is positive! Likely an infinite loop.");
      it = knots.begin(); // wrap around if we go past end
      it++; // and skip past duplicate first knot
    }
    OTHER_ASSERT(it != knots.end());

    p3 = it->second->tangent_in;
    p4 = it->second->pt;

    TV pt;
    TV prev = p1;
    while(t<1){
      while(dst < step){
        pt = other::point(p1,p2,p3,p4,t);
        dst += (pt-prev).magnitude();
        prev = pt;
        t+=tstep;
        if(t>=1) break;
      }
      if(t<1) {
        path.append(pt);
        dst = 0;
      }
    }
    t-=1;
  } while(it->second != end->second);

  path.append(end->second->pt);

  OTHER_ASSERT(path.size() >= 2 && (int)path.size() == res +1);
  return path;
}

template<int d> Span<d> Bezier<d>::segment(real t) const{
  auto it = knots.lower_bound(t);
  if(it==knots.end()) return Span<d>(it->second,it->second,it->first,it->first);
  if(it != knots.begin() ) --it;
  auto iit = it; ++iit;
  return Span<d>(it->second,iit->second,it->first,iit->first);
}

template<int d> Vector<real,d> Bezier<d>::point(real t) const{
  if(knots.count(t)) return knots.find(t)->second->pt;
  Span<d> seg = segment(t);
  if(seg.start_t==seg.end_t) {Vector<real,d> v; v.fill(numeric_limits<real>::infinity()); return v;}
  return other::point(seg.start->pt,seg.start->tangent_out,seg.end->tangent_in,seg.end->pt,(t-seg.start_t)/(seg.end_t-seg.start_t));
}

template<int d> Vector<real,d> Bezier<d>::tangent(real t) const{
  Span<d> seg = segment(t);
  if(seg.start_t==seg.end_t) {Vector<real,d> v; v.fill(numeric_limits<real>::infinity()); return v;}
  real subt = (t-seg.start_t)/(seg.end_t-seg.start_t);

  /*
  real u = 1.-subt;
  TV out = (seg.start.pt*-3*u*u + seg.start.tangent_out*3*(u*u - 2*u*subt) + seg.end.tangent_in*3*(2*u*subt - subt*subt) + seg.end.pt*3*subt*subt).normalized();
  */

  Matrix<real,4,d> P;
  for(int i =0; i < d; i++){
    P.set_column(i,Vector<real,4>(seg.start->pt[i],seg.start->tangent_out[i],seg.end->tangent_in[i],seg.end->pt[i]));
  }

  Matrix<real,4> A( -1, 3, -3, 1,
                     3, -6, 3, 0,
                     -3,3, 0, 0,
                     1, 0,0, 0);
  Matrix<real,4,d> AP = A*P;
  Matrix<real,4,1> tt; tt.set_column(0,Vector<real,4>(3*subt*subt,2*subt,1,0));
  return (tt.transposed()*AP).transposed().column(0).normalized();
}

template<int d> Array<Vector<real,d>> Bezier<d>::evaluate(int res) const{
  if(t_range == Box<real>(0)) return Array<Vector<real,d>>();
  return segment(InvertableBox(t_range.min, t_range.max),res);
}

template<int d> Array<Vector<real,d>> Bezier<d>::alen_evaluate(int res) const{
  if(t_range == Box<real>(0)) return Array<Vector<real,d>>();
  return segment(InvertableBox(t_range.min, t_range.max),res);
}

template<int d> void Bezier<d>::append_knot(const TV& pt, TV tin, TV tout){
  if(!isfinite(tin)) tin = pt;
  if(!isfinite(tout)) tout = pt;

  if(knots.size()) t_range.max+=1.;
  knots.insert(make_pair(t_range.max,new_<Knot<d>>(pt,tin,tout)));
}

template<int d> void Bezier<d>::insert_knot(const real t){
  if(knots.count(t)) return;
  auto it = knots.upper_bound(t);
  OTHER_ASSERT(it!=knots.begin() && it!=knots.end());
  // have we accounted for all collision cases?

  Knot<d>& pt2 = *it->second; real t1 = it->first;
  --it;
  Knot<d>& pt1 = *it->second; real t2 = it->first;
  real isub_t = (t-t1)/(t2-t1);
  real sub_t = 1-isub_t;

  //via de casteljau algorithm http://en.wikipedia.org/wiki/De_Casteljau's_algorithm
  TV prev_out = isub_t*pt1.pt + sub_t*pt1.tangent_out;
  TV sub_pt = isub_t*pt1.tangent_out + sub_t*pt2.tangent_in;
  TV new_in = isub_t*prev_out + sub_t*sub_pt;
  TV next_in = isub_t*pt2.tangent_in + sub_t*pt2.pt;
  TV new_out = isub_t*sub_pt + sub_t*next_in;

  TV new_pt = isub_t*new_in + sub_t*new_out;
  knots.insert(make_pair(t,new_<Knot<d>>(new_pt)));
  pt1.tangent_out = prev_out;
  Ref<Knot<d> > k = knots.find(t)->second;
  k->tangent_in = new_in;
  k->tangent_out = new_out;
  pt2.tangent_in = next_in;
}

template<int d> Box<Vector<real,d> > Bezier<d>::bounding_box() const{
  Box<Vector<real,d> > bx;
  for(auto& pr : knots){
    const Knot<d>& knot = *pr.second;
    bx.enlarge(knot.pt);
    bx.enlarge(knot.tangent_in);
    bx.enlarge(knot.tangent_out);
  }
  return bx;
}

template<int d> void Bezier<d>::translate(const TV& t){
  for(auto& k : knots){
    if(!(k.first == t_max() && closed())) (*k.second)+=t;
  }
}

template<int d> void Bezier<d>::fuse_ends(){
  if(knots.size()<=2) return;
  TV b = knots.begin()->second->pt;
  auto it = knots.end(); --it;
  if(b_closed) --it; //jump back one because a closed bezier has a doubled beginning
  real sz =  1e-5*bounding_box().sizes().magnitude();
  OTHER_ASSERT(it!=knots.end());
  TV e = it->second->pt;
  if((b-e).sqr_magnitude() < sz*sz){
    if(knots.find(t_range.max)->second == knots.begin()->second){
      knots.erase(t_range.max);
      auto end = knots.end(); --end;
      t_range.max = end->first;
    }
    TV tin = knots.find(t_range.max)->second->tangent_in;
    knots.erase(t_range.max);
    knots.insert(make_pair(t_range.max,knots.find(t_range.min)->second));
    knots.begin()->second->tangent_in = tin;
    b_closed = true;
  }
}

template<int d> void Bezier<d>::close(){
  if(b_closed) return;
  OTHER_ASSERT(knots.size()>2);
  t_range.max += (T)1;
  knots.insert(make_pair(t_range.max,knots.find(t_range.min)->second));
  b_closed = true;
}

template<int d> void Bezier<d>::erase(real t){
  auto it = knots.find(t);
  OTHER_ASSERT(it!=knots.end());
  if(it==knots.begin()) {
    t_range.min=(++it)->first;
    if(b_closed){
      auto end = knots.end(); --end;
      knots.erase(end);
      end = knots.end(); --end;
      t_range.max = end->first;
      b_closed = false;
    }
  }else if(++it==knots.end()){
    --it;
    t_range.max=(--it)->first;
    if(b_closed){
      auto b = knots.begin();
      knots.erase(b);
      b = knots.begin();
      t_range.min = b->first;
      b_closed = false;
    }
  }
  knots.erase(t);
}

template<int d> Bezier<d>::Bezier() : t_range(0),b_closed(false){}
template<int d> Bezier<d>::Bezier(const Bezier<d>& b) : t_range(b.t_range),b_closed(b.b_closed)
{
  for(auto& k : b.knots)
    knots.insert(make_pair(k.first,(k.first==b.t_max() && b.closed()) ? knots.begin()->second : new_<Knot<d>>(*k.second)));
}
template<int d> Bezier<d>::~Bezier(){}

template<int d> real PointSolve<d>::distance(real t){
  Vector<real,d> p = other::point(k1.second->pt,k1.second->tangent_out,k2.second->tangent_in,k2.second->pt,t/(k2.first-k1.first));
  return (p-pt).magnitude();
}

template class Bezier<2>;

template struct PointSolve<2>;
template struct PointSolve<3>;

template struct Span<2>;
template struct Span<3>;

#ifdef OTHER_PYTHON

PyObject* to_python(const InvertableBox& self) {
  return to_python(tuple(self.begin,self.end));
}

InvertableBox FromPython<InvertableBox>::convert(PyObject* object) {
  const auto extents = from_python<Tuple<real,real>>(object);
  return InvertableBox(extents.x,extents.y);
}

#endif

std::ostream& operator<<(std::ostream& os, const InvertableBox& ib) {
  os << "[" << ib.begin << ", " << ib.end << "]";
  return os;
}

}
using namespace other;

void wrap_bezier() {
  {
    typedef Knot<2> Self;
    Class<Self>("Knot")
      .OTHER_INIT()
      .OTHER_FIELD(pt)
      .OTHER_FIELD(tangent_in)
      .OTHER_FIELD(tangent_out)
      ;
  }
  {
    typedef Bezier<2> Self;
    Class<Self>("Bezier")
      .OTHER_INIT()
      .OTHER_FIELD(knots)
      .OTHER_METHOD(t_max)
      .OTHER_METHOD(t_min)
      .OTHER_METHOD(closed)
      .OTHER_METHOD(close)
      .OTHER_METHOD(fuse_ends)
      .OTHER_METHOD(evaluate)
      .OTHER_METHOD(append_knot)
      ;
  }
}
