//#####################################################################
// Class Implicit
//#####################################################################
#pragma once

#include <geode/geometry/Box.h>
#include <geode/python/Object.h>
#include <geode/vector/Vector.h>
namespace geode {

template<class TV>
class Implicit:public Object
{
  typedef typename TV::Scalar T;
public:
  GEODE_DECLARE_TYPE(GEODE_CORE_EXPORT)
  static const int d = TV::m;

  Implicit();
  virtual ~Implicit();

  virtual T phi(const TV& X) const=0;
  virtual TV normal(const TV& X) const=0;
  virtual TV surface(const TV& X) const=0;
  virtual bool lazy_inside(const TV& X) const=0;
  virtual Box<TV> bounding_box() const=0;
  virtual string repr() const=0;
};
}
