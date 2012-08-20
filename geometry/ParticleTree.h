//#####################################################################
// Class ParticleTree
//#####################################################################
#pragma once

#include <other/core/geometry/forward.h>
#include <other/core/geometry/BoxTree.h>
#include <other/core/utility/using.h>
namespace other{

template<class TV> class ParticleTree : public BoxTree<TV>
{
  typedef typename TV::Scalar T;
public:
  OTHER_DECLARE_TYPE
  typedef BoxTree<TV> Base;
  OTHER_USING(leaves,prims,boxes,update_nonleaf_boxes)

  const Array<const TV> X;

protected:
  ParticleTree(Array<const TV> X, int leaf_size) OTHER_EXPORT;
public:
  ~ParticleTree();

  void update() OTHER_EXPORT; // Call whenever X changes
  Array<int> remove_duplicates(T tolerance) const OTHER_EXPORT; // Returns map from point to component index
  template<class Shape> void intersection(const Shape& box, Array<int>& hits) const OTHER_EXPORT;
};

}
