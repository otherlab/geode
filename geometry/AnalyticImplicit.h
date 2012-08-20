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
  template<class... Args>
  AnalyticImplicit(const Args&... args)
      :Shape(args...) {}
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
