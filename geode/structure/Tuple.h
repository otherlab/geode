// Tuples with convenient x,y,z,... fields
#pragma once

#include <geode/structure/Empty.h>
#include <geode/structure/Singleton.h>
#include <geode/structure/Pair.h>
#include <geode/structure/Triple.h>
#include <geode/structure/Quad.h>
#include <geode/structure/Quintuple.h>
#include <geode/utility/enumerate.h>
#include <geode/vector/forward.h>
namespace geode {

template<class T> static inline Tuple<T>       as_tuple(const Vector<T,1>& v) { return Tuple<T>      (v.x); }
template<class T> static inline Tuple<T,T>     as_tuple(const Vector<T,2>& v) { return Tuple<T,T>    (v.x,v.y); }
template<class T> static inline Tuple<T,T,T>   as_tuple(const Vector<T,3>& v) { return Tuple<T,T,T>  (v.x,v.y,v.z); }
template<class T> static inline Tuple<T,T,T,T> as_tuple(const Vector<T,4>& v) { return Tuple<T,T,T,T>(v.x,v.y,v.z,v.w); }

#ifdef GEODE_VARIADIC

// Convenience and conversion

template<class... Args> static inline Tuple<Args...> tuple(const Args&... args) {
  return Tuple<Args...>(args...);
}

// Tuples of unusual size

template<class T> struct make_reference_const;
template<class T> struct make_reference_const<T&> { typedef const T& type; };

template<class T0,class T1,class T2,class T3,class T4,class... Rest> class Tuple<T0,T1,T2,T3,T4,Rest...> {
  static_assert(sizeof...(Rest)>0,"");
public:
  enum { m = 5+sizeof...(Rest) };
  Tuple<T0,T1,T2,T3,T4> left;
  Tuple<Rest...> right;

  Tuple() {}

  Tuple(const T0& x0, const T1& x1, const T2& x2, const T3& x3, const T4& x4, const Rest&... rest)
    : left(x0,x1,x2,x3,x4), right(rest...) {}

  bool operator==(const Tuple& t) const {
    return left==t.left && right==t.right;
  }

  bool operator!=(const Tuple& t) const {
    return !(*this==t);
  }

  template<int i> auto get()
    -> decltype(choice_helper(mpl::int_<(i>=5)>(),left,right).template get<(i>=5?i-5:i)>()) {
    return choice_helper(mpl::int_<(i>=5)>(),left,right).template get<(i>=5?i-5:i)>();
  }

  template<int i> auto get() const
    -> typename make_reference_const<decltype(choice_helper(mpl::int_<(i>=5)>(),left,right).template get<(i>=5?i-5:i)>())>::type {
    return choice_helper(mpl::int_<(i>=5)>(),left,right).template get<(i>=5?i-5:i)>();
  }
};

#else // Unpleasant nonvariadic versions

template<class A0, class A1, class A2, class A3, class A4, class A5, class A6> struct has_to_python<Tuple<A0,A1,A2,A3,A4,A5,A6>> : public mpl::and_<mpl::and_<has_to_python<A0>, has_to_python<A1>, has_to_python<A2>>, mpl::and_<has_to_python<A3>, has_to_python<A4>, has_to_python<A5>, has_to_python<A6>> > {};
template<class A0, class A1, class A2, class A3, class A4, class A5, class A6> struct has_from_python<Tuple<A0,A1,A2,A3,A4,A5,A6>> : public mpl::and_<mpl::and_<has_from_python<A0>, has_from_python<A1>, has_from_python<A2>>, mpl::and_<has_from_python<A3>, has_from_python<A4>, has_from_python<A5>, has_from_python<A6>> > {};

static inline Tuple<> tuple() { return Tuple<>(); }
template<class A0> static inline Tuple<A0> tuple(const A0& a0) { return Tuple<A0>(a0); }
template<class A0,class A1> static inline Tuple<A0,A1> tuple(const A0& a0,const A1& a1) { return Tuple<A0,A1>(a0,a1); }
template<class A0,class A1,class A2> static inline Tuple<A0,A1,A2> tuple(const A0& a0,const A1& a1,const A2& a2) { return Tuple<A0,A1,A2>(a0,a1,a2); }
template<class A0,class A1,class A2,class A3> static inline Tuple<A0,A1,A2,A3> tuple(const A0& a0,const A1& a1,const A2& a2,const A3& a3) { return Tuple<A0,A1,A2,A3>(a0,a1,a2,a3); }
template<class A0,class A1,class A2,class A3,class A4> static inline Tuple<A0,A1,A2,A3,A4> tuple(const A0& a0,const A1& a1,const A2& a2,const A3& a3,const A4& a4) { return Tuple<A0,A1,A2,A3,A4>(a0,a1,a2,a3,a4); }

#endif

}
