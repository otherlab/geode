//#####################################################################
// Class Implicit
//#####################################################################
#pragma once

#include <geode/geometry/Box.h>
#include <geode/utility/Object.h>
#include <geode/vector/Vector.h>
namespace geode {

template<class TV> class Implicit : public Object {
  typedef typename TV::Scalar T;
public:
  GEODE_NEW_FRIEND
  static const int d = TV::m;

protected:
  Implicit();
public:
  virtual ~Implicit();

  virtual T phi(const TV& X) const=0;
  virtual TV normal(const TV& X) const=0;
  virtual TV surface(const TV& X) const=0;
  virtual bool lazy_inside(const TV& X) const=0;
  virtual Box<TV> bounding_box() const=0;
  virtual string repr() const=0;
};
}
