//#####################################################################
// Class AnalyticImplicit
//#####################################################################
#include <other/core/geometry/AnalyticImplicit.h>
#include <other/core/geometry/Box.h>
#include <other/core/geometry/Sphere.h>
#include <other/core/geometry/Capsule.h>
#include <other/core/geometry/Cylinder.h>
#include <other/core/geometry/Plane.h>
#include <other/core/python/Class.h>
namespace other{

typedef real T;

template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Box<Vector<T,1>>>)
template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Box<Vector<T,2>>>)
template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Box<Vector<T,3>>>)
template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Sphere<Vector<T,2>>>)
template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Sphere<Vector<T,3>>>)
template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Capsule<Vector<T,2>>>)
template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Capsule<Vector<T,3>>>)
template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Cylinder>)
template<> OTHER_DEFINE_TYPE(AnalyticImplicit<Plane<T>>)

template<class Shape> AnalyticImplicit<Shape>::
~AnalyticImplicit() {}

template<class Shape> T AnalyticImplicit<Shape>::
phi(const TV& X) const
{
    return Shape::phi(X);
}

template<class Shape> typename Shape::VectorT AnalyticImplicit<Shape>::
normal(const TV& X) const
{
    return Shape::normal(X);
}

template<class Shape> typename Shape::VectorT AnalyticImplicit<Shape>::
surface(const TV& X) const
{
    return Shape::surface(X);
}

template<class Shape> bool AnalyticImplicit<Shape>::
lazy_inside(const TV& X) const
{
    return Shape::lazy_inside(X);
}

template<class Shape> Box<typename Shape::VectorT> AnalyticImplicit<Shape>::
bounding_box() const
{
    return Shape::bounding_box();
}

template<class Shape> string AnalyticImplicit<Shape>::
repr() const
{
    return Shape::repr();
}

}
using namespace other;
using namespace python;

template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;

  {typedef AnalyticImplicit<Sphere<TV> > Self;
  Class<Self>(d==2?"Sphere2d":"Sphere3d")
    .OTHER_INIT(TV,T)
    ;}

  {typedef AnalyticImplicit<Capsule<TV> > Self;
  Class<Self>(d==2?"Capsule2d":"Capsule3d")
    .OTHER_INIT(TV,TV,T)
    ;}
}

template<int d> static void wrap_box_helper() {
  typedef Vector<T,d> TV;
  typedef AnalyticImplicit<Box<TV> > Self;
  Class<Self>(d==1?"Box1d":d==2?"Box2d":"Box3d")
    .OTHER_INIT(TV,TV)
    .OTHER_FIELD(min)
    .OTHER_FIELD(max)
    .OTHER_METHOD(sizes)
    .template method<Box<TV>(Box<TV>::*)(T)const>("thickened",&Self::thickened)
    .template method<void(Box<TV>::*)(const TV&)>("enlarge",&Self::enlarge)
    ;

  function(d==1?"empty_box_1d":d==2?"empty_box_2d":"empty_box_3d",Box<TV>::empty_box);
}

void wrap_analytic_implicit() {
  wrap_helper<2>();
  wrap_helper<3>();
  wrap_box_helper<1>();
  wrap_box_helper<2>();
  wrap_box_helper<3>();

  typedef Vector<T,3> TV;

  {typedef AnalyticImplicit<Plane<T>> Self;
  Class<Self>("Plane")
    .OTHER_INIT(TV,TV)
    ;}

  {typedef AnalyticImplicit<Cylinder> Self;
  Class<Self>("Cylinder")
    .OTHER_INIT(TV,TV,T)
    ;}
}
