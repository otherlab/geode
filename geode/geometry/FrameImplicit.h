//#####################################################################
// Class FrameImplicit
//#####################################################################
#pragma once

#include <geode/geometry/Implicit.h>
#include <geode/vector/Frame.h>
namespace geode {

template<class TV>
class FrameImplicit:public Implicit<TV>
{
  typedef typename TV::Scalar T;
public:
  GEODE_NEW_FRIEND
  typedef Implicit<TV> Base;

  Frame<TV> frame;
  const Ref<const Implicit<TV> > object;

  FrameImplicit(Frame<TV> frame, const Implicit<TV>& object);
  virtual ~FrameImplicit();

  virtual T phi(const TV& X) const;
  virtual TV normal(const TV& X) const;
  virtual TV surface(const TV& X) const;
  virtual bool lazy_inside(const TV& X) const;
  virtual Box<TV> bounding_box() const;
  virtual string repr() const;
};
}
