//#####################################################################
// Class AnalyticImplicit
//#####################################################################
#include <geode/geometry/AnalyticImplicit.h>
#include <geode/geometry/Box.h>
#include <geode/geometry/Sphere.h>
#include <geode/geometry/Capsule.h>
#include <geode/geometry/Cylinder.h>
#include <geode/geometry/Plane.h>
namespace geode {

typedef real T;

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
