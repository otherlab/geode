//#####################################################################
// Class Implicit
//#####################################################################
#pragma once

#include <other/core/geometry/Box.h>
#include <other/core/python/Object.h>
#include <other/core/vector/Vector.h>
namespace other{

template<class TV>
class Implicit:public Object
{
  typedef typename TV::Scalar T;
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
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
