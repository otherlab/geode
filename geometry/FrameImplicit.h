//#####################################################################
// Class FrameImplicit
//#####################################################################
#pragma once

#include <other/core/geometry/Implicit.h>
#include <other/core/vector/Frame.h>
namespace other{

template<class TV>
class FrameImplicit:public Implicit<TV>
{
  typedef typename TV::Scalar T;
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
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
