//#####################################################################
// Class ParticleTree
//#####################################################################
#include <other/core/geometry/ParticleTree.h>
#include <other/core/geometry/Sphere.h>
#include <other/core/geometry/traverse.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/python/Class.h>
#include <other/core/structure/UnionFind.h>
namespace other{
using std::cout;
using std::endl;

typedef real T;
template<> OTHER_DEFINE_TYPE(ParticleTree<Vector<T,2>>)
template<> OTHER_DEFINE_TYPE(ParticleTree<Vector<T,3>>)

template<class TV> ParticleTree<TV>::
ParticleTree(Array<const TV> X,int leaf_size)
  : Base(X.raw(),leaf_size), X(X) {}

template<class TV> ParticleTree<TV>::
~ParticleTree() {}

template<class TV> void ParticleTree<TV>::
update() {
  for (int n : leaves)
    boxes[n] = other::bounding_box(X.subset(prims(n)));
  update_nonleaf_boxes();
}

namespace {
template<class TV> struct DuplicatesVisitor {
  const ParticleTree<TV>& tree;
  UnionFind components;
  T tolerance;

  DuplicatesVisitor(const ParticleTree<TV>& tree, T tolerance)
    : tree(tree), components(tree.X.size()), tolerance(tolerance) {}

  bool cull(int n) const { return false; }
  bool cull(int n0, int n1) const { return false; }

  void leaf(int n) {
    auto prims = tree.prims(n);
    for (int i=0;i<prims.size();i++) for (int j=i+1;j<prims.size();j++)
      if((tree.X[prims[i]]-tree.X[prims[j]]).sqr_magnitude()<=sqr(tolerance))
        components.merge(prims[i],prims[j]);
  }

  void leaf(int n0, int n1) {
    for (int i : tree.prims(n0)) for (int j : tree.prims(n1))
      if ((tree.X[i]-tree.X[j]).sqr_magnitude()<=sqr(tolerance))
        components.merge(i,j);
  }
};
}

template<class TV> Array<int> ParticleTree<TV>::
remove_duplicates(T tolerance) const {
  DuplicatesVisitor<TV> visitor(*this,tolerance);
  traverse(*this,visitor,tolerance);
  Array<int> map(X.size(),false);
  int count=0;
  for(int i=0;i<X.size();i++)
    if(visitor.components.is_root(i))
      map[i] = count++;
  for(int i=0;i<X.size();i++)
    map[i] = map[visitor.components.find(i)];
  return map;
}

template<class TV,class Shape> static void intersection_helper(const ParticleTree<TV>& self, const Shape& shape, Array<int>& hits, int node) {
  if (shape.lazy_intersects(self.boxes[node])) {
    if(!self.is_leaf(node)) {
      intersection_helper(self,shape,hits,2*node+1);
      intersection_helper(self,shape,hits,2*node+2);
    } else
      for (int i : self.prims(node))
        if (shape.lazy_inside(self.X[i]))
          hits.append(i);
  }
}

template<class TV> template<class Shape> void ParticleTree<TV>::
intersection(const Shape& shape, Array<int>& hits) const {
  hits.clear();
  intersection_helper(*this,shape,hits,0);
}

#define INSTANTIATE(d) \
  template class ParticleTree<Vector<T,d>>; \
  template OTHER_CORE_EXPORT void ParticleTree<Vector<T,d>>::intersection(const Box<Vector<T,d>>&,Array<int>&) const; \
  template OTHER_CORE_EXPORT void ParticleTree<Vector<T,d>>::intersection(const Sphere<Vector<T,d>>&,Array<int>&) const;
INSTANTIATE(2)
INSTANTIATE(3)
}
using namespace other;

template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef ParticleTree<TV> Self;
  Class<Self>(d==2?"ParticleTree2d":"ParticleTree3d")
    .OTHER_INIT(Array<const TV>,int)
    .OTHER_FIELD(X)
    .OTHER_METHOD(update)
    .OTHER_METHOD(remove_duplicates)
    ;
}

void wrap_particle_tree() {
  wrap_helper<2>();
  wrap_helper<3>();
}
