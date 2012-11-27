//#####################################################################
// Class AnalyticImplicit
//#####################################################################
#pragma once

#include <other/core/geometry/Implicit.h>
namespace other{

template<class Shape>
class AnalyticImplicit:public Implicit<typename Shape::VectorT>,public Shape
{
  typedef real T;
  typedef typename Shape::VectorT TV;
  BOOST_MPL_ASSERT((boost::is_same<T,typename TV::Scalar>));
public:
  OTHER_DECLARE_TYPE
  typedef Implicit<TV> Base;

protected:
#ifdef OTHER_VARIADIC

  template<class... Args>
  AnalyticImplicit(const Args&... args)
      : Shape(args...) {}

#else // Unpleasant nonvariadic versions

  template<class A0> AnalyticImplicit(const A0& a0) : Shape(a0) {}
  template<class A0,class A1> AnalyticImplicit(const A0& a0,const A1& a1) : Shape(a0,a1) {}
  template<class A0,class A1,class A2> AnalyticImplicit(const A0& a0,const A1& a1,const A2& a2) : Shape(a0,a1,a2) {}

#endif
public:
  ~AnalyticImplicit();

  virtual T phi(const TV& X) const;
  virtual TV normal(const TV& X) const;
  virtual TV surface(const TV& X) const;
  virtual bool lazy_inside(const TV& X) const;
  virtual Box<TV> bounding_box() const;
  virtual string repr() const;
};
}
