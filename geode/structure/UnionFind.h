// Union-find algorithm for disjoint sets
#pragma once

// Our implementation uses path halving rather than path compression, and reuses the parents
// array to store ranks as well.  This combination is LRPH-MS in the terminology of
//
//   Patwary, Blair, Manne (2010), "Experiments on Union-Find Algorithms for the Disjoint-Set Data Structure".
//
// Although Patwary et al. claim that Rem is actually faster despite its inferior asymptotic
// complexity.  Our experients did not confirm this: Rem switched to slower for sparse random
// graphs between 10M and 100M vertices.  Therefore, we stick with reassuring O(n alpha(n)).

#include <geode/structure/Tuple.h>
#include <geode/array/ConstantMap.h>
#include <geode/array/Array.h>
namespace geode {

template<class Parents> struct UnionFindBase {
  // For roots, parents[i] = -rank-1
  mutable Parents parents;

protected:
  template<class... Args> UnionFindBase(Args&&... args)
    : parents(args...) {}
public:

  int size() const {
    return parents.size();
  }

  bool is_root(const int i) const {
    return parents[i]<0;
  }

  bool same(const int i, const int j) const {
    return find(i)==find(j);
  }

  int find(int i) const {
    // Use path halving rather than full path compression for speed.
    // Path halving changes x -> y -> z -> ... to x,y -> z -> ...
    for (;;) {
      const int pi = parents[i];
      if (pi < 0)
        return i;
      const int ppi = parents[pi];
      if (ppi < 0)
        return pi;
      parents[i] = ppi;
      i = ppi;
    }
  }

  int merge(const int i, const int j) {
    return merge_roots(find(i),find(j));
  }

  // Convenience merging for vectors of indices
  int merge(const Vector<int,1> v) {
    return find(v.x);
  }
  int merge(const Vector<int,2> v) {
    return merge(v.x,v.y);
  }
  int merge(const Vector<int,3> v) {
    const int xy = merge(v.x,v.y);
    return merge_roots(xy,find(v.z));
  }
  int merge(const Vector<int,4> v) {
    const int xy = merge(v.x,v.y);
    const int zw = merge(v.z,v.w);
    return merge_roots(find(xy),zw);
  }

  // Merge any nonempty list of indices
  template<class TA> int merge(const TA& indices) {
    STATIC_ASSERT_SAME(typename TA::Element,int);
    const int n = indices.size();
    int r = find(indices[0]);
    for (int i=1;i<n;i++)
      r = merge_roots(r,find(indices[i]));
    return r;
  }

  void merge(const UnionFindBase& union_find) {
    assert(size()==union_find.size());
    for (int i=0;i<size();i++) {
      const int j = union_find.parents[i];
      if (j >= 0)
        merge(i,j);
    }
  }

  int roots() const {
    int roots = 0;
    for (int i=0;i<parents.size();i++)
      roots += is_root(i);
    return roots;
  }

private:
  int merge_roots(int i, int j) {
    assert(parents[i]<0 && parents[j]<0);
    if (i != j) {
      if (parents[i] > parents[j])
        swap(i,j);
      else if (parents[i] == parents[j])
        parents[i]--;
      parents[j] = i;
    }
    return i; 
  }
};

class UnionFind : public UnionFindBase<Array<int>> {
public:
  explicit UnionFind(const int entries=0) {
    initialize(entries);
  }

  void initialize(const int entries) {
    parents.copy(constant_map(entries,-1));
  }

  void clear() {
    parents.clear();
  }

  int add_entry() {
    return parents.append(-1);
  }

  void extend(const int n) {
    parents.extend(constant_map(n,-1));
  }

  // Okay for map to yield invalid indices for isolated elements
  template<class TA> void mapped_merge(const UnionFind& union_find, const TA& map) {
    for (int i=0;i<union_find.size();i++) {
      const int root = union_find.find(i);
      if (i != root)
        merge(map(i),map(root));
    }
  }

  void forest_edges(Array<Tuple<int,int>>& pairs) const {
    pairs.clear();
    for (int i=0;i<size();i++) {
      int j = find(i);
      if (i!=j) pairs.append(tuple(i,j));
    }
  }

  void merge_forest_edges(RawArray<const Tuple<int,int>> pairs) {
    for (int i=0;i<pairs.m;i++)
      merge(pairs(i).x,pairs(i).y);
  }
};

template<int n> struct SmallUnionFind : public UnionFindBase<Vector<signed char,n>> {
  static_assert(0<=n && n<=128,"Large union finds should use UnionFind, not SmallUnionFind");
  
  SmallUnionFind() {
    this->parents.fill(-1);
  }
};

struct RawUnionFind : public UnionFindBase<RawArray<int>> {
  RawUnionFind(RawArray<int> parents)
    : UnionFindBase<RawArray<int>>(parents) {
    this->parents.fill(-1);
  }
};

}
