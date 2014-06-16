//#####################################################################
// Class ParticleTree
//#####################################################################
#include <geode/geometry/ParticleTree.h>
#include <geode/geometry/Sphere.h>
#include <geode/geometry/traverse.h>
#include <geode/array/IndirectArray.h>
#include <geode/python/Class.h>
#include <geode/structure/UnionFind.h>
namespace geode {
using std::cout;
using std::endl;

typedef real T;
template<> GEODE_DEFINE_TYPE(ParticleTree<Vector<T,2>>)
template<> GEODE_DEFINE_TYPE(ParticleTree<Vector<T,3>>)

template<class TV> ParticleTree<TV>::
ParticleTree(Array<const TV> X,int leaf_size)
  : Base(X.raw(),leaf_size), X(X) {}

template<class TV> ParticleTree<TV>::
~ParticleTree() {}

template<class TV> void ParticleTree<TV>::
update() {
  for (int n : leaves)
    boxes[n] = geode::bounding_box(X.subset(prims(n)));
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
  double_traverse(*this,visitor,tolerance);
  Array<int> map(X.size(),uninit);
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
  if (X.size())
    intersection_helper(*this,shape,hits,0);
}

template<class TV> static void closest_point_helper(const ParticleTree<TV>& self, TV point, int& index, T& sqr_distance, int node, int ignore) {
  if (!self.is_leaf(node)) {
    Vector<T,2> bounds(self.boxes[2*node+1].sqr_distance_bound(point),
                       self.boxes[2*node+2].sqr_distance_bound(point));
    int c = bounds.argmin();
    if (bounds[c]<sqr_distance)
      closest_point_helper(self,point,index,sqr_distance,2*node+1+c,ignore);
    if (bounds[1-c]<sqr_distance)
      closest_point_helper(self,point,index,sqr_distance,2*node+2-c,ignore);
  } else
    for (int t : self.prims(node)) {
      T sqr_d = sqr_magnitude(point-self.X[t]);
      if (sqr_distance>sqr_d && t != ignore) {
        sqr_distance = sqr_d;
        index = t;
      }
    }
}

template<class TV> TV ParticleTree<TV>::
closest_point(TV point, int& index, T max_distance, int ignore) const {
  index = -1;
  if (nodes()) {
    T sqr_distance = sqr(max_distance);
    closest_point_helper(*this,point,index,sqr_distance,0,ignore);
  }
  if (index == -1) {
    TV x;
    x.fill(inf);
    return x;
  } else
    return X[index];
}

template<class TV> TV ParticleTree<TV>::
closest_point(TV point, T max_distance) const {
  int index;
  return closest_point(point, index, max_distance);
}

template<class TV> Tuple<TV,int> ParticleTree<TV>::closest_point_py(TV point, T max_distance) const {
  int index = -1;
  const TV p = closest_point(point,index,max_distance);
  return tuple(p,index);
}

#define INSTANTIATE(d) \
  template class ParticleTree<Vector<T,d>>; \
  template GEODE_CORE_EXPORT void ParticleTree<Vector<T,d>>::intersection(const Box<Vector<T,d>>&,Array<int>&) const; \
  template GEODE_CORE_EXPORT void ParticleTree<Vector<T,d>>::intersection(const Sphere<Vector<T,d>>&,Array<int>&) const;
INSTANTIATE(2)
INSTANTIATE(3)
}
using namespace geode;

template<int d> static void wrap_helper() {
  typedef Vector<T,d> TV;
  typedef ParticleTree<TV> Self;
  Class<Self>(d==2?"ParticleTree2d":"ParticleTree3d")
    .GEODE_INIT(Array<const TV>,int)
    .GEODE_FIELD(X)
    .GEODE_METHOD(update)
    .GEODE_METHOD(remove_duplicates)
    .GEODE_METHOD_2("closest_point",closest_point_py)
    ;
}

void wrap_particle_tree() {
  wrap_helper<2>();
  wrap_helper<3>();
}
