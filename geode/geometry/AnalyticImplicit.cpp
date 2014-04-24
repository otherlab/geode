//#####################################################################
// Class AnalyticImplicit
//#####################################################################
#include <geode/geometry/AnalyticImplicit.h>
#include <geode/geometry/Box.h>
#include <geode/geometry/Sphere.h>
#include <geode/geometry/Capsule.h>
#include <geode/geometry/Cylinder.h>
#include <geode/geometry/Plane.h>
#include <geode/python/Class.h>
namespace geode {

typedef real T;

template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Box<Vector<T,1>>>)
template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Box<Vector<T,2>>>)
template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Box<Vector<T,3>>>)
template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Sphere<Vector<T,2>>>)
template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Sphere<Vector<T,3>>>)
template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Capsule<Vector<T,2>>>)
template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Capsule<Vector<T,3>>>)
template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Cylinder>)
template<> GEODE_DEFINE_TYPE(AnalyticImplicit<Plane<T>>)

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

template AnalyticImplicit<Box<Vector<T,1>>>::~AnalyticImplicit();
template AnalyticImplicit<Box<Vector<T,2>>>::~AnalyticImplicit();
template AnalyticImplicit<Box<Vector<T,3>>>::~AnalyticImplicit();
template AnalyticImplicit<Sphere<Vector<T,2>>>::~AnalyticImplicit();
template AnalyticImplicit<Sphere<Vector<T,3>>>::~AnalyticImplicit();
template AnalyticImplicit<Capsule<Vector<T,2>>>::~AnalyticImplicit();
template AnalyticImplicit<Capsule<Vector<T,3>>>::~AnalyticImplicit();
template AnalyticImplicit<Cylinder>::~AnalyticImplicit();
template AnalyticImplicit<Plane<T>>::~AnalyticImplicit();

}
using namespace geode;
using namespace python;

template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;

  {typedef AnalyticImplicit<Sphere<TV> > Self;
  Class<Self>(d==2?"Sphere2d":"Sphere3d")
    .GEODE_INIT(TV,T)
    .GEODE_METHOD(volume)
    ;}

  {typedef AnalyticImplicit<Capsule<TV> > Self;
  Class<Self>(d==2?"Capsule2d":"Capsule3d")
    .GEODE_INIT(TV,TV,T)
    .GEODE_METHOD(volume)
    ;}
}

template<int d> static void wrap_box_helper() {
  typedef Vector<T,d> TV;
  typedef AnalyticImplicit<Box<TV> > Self;
  Class<Self>(d==1?"Box1d":d==2?"Box2d":"Box3d")
    .GEODE_INIT(TV,TV)
    .GEODE_FIELD(min)
    .GEODE_FIELD(max)
    .GEODE_METHOD(sizes)
    .GEODE_METHOD(clamp)
    .GEODE_METHOD(center)
    .GEODE_METHOD(volume)
    .template method<Box<TV>(Box<TV>::*)(T)const>("thickened",&Self::thickened)
    .template method<void(Box<TV>::*)(const TV&)>("enlarge",&Self::enlarge)
    ;

  geode::python::function(d==1?"empty_box_1d":d==2?"empty_box_2d":"empty_box_3d",Box<TV>::empty_box);
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
    .GEODE_INIT(TV,TV)
    ;}

  {typedef AnalyticImplicit<Cylinder> Self;
  Class<Self>("Cylinder")
    .GEODE_INIT(TV,TV,T)
    ;}
}
