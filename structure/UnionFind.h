//#####################################################################
// Class UnionFind
//#####################################################################
#pragma once

#include <other/core/structure/Tuple.h>
#include <other/core/array/ConstantMap.h>
#include <other/core/array/Array.h>
namespace other{

template<class Derived,class Parents,class Ranks>
struct UnionFindBase {
  typedef typename Ranks::Element Rank;

  mutable Parents parents;
  Ranks ranks;

protected:
  UnionFindBase() {} // It's the derived class's responsibility to initialize parents and ranks
public:

  int size() const {
    return parents.size();
  }

  void clear_connectivity() {
    parents.fill(-1);
    ranks.fill(0);
  }

  bool is_root(const int i) const {
    return parents[i]<0;
  }

  int find(const int i) const {
    int root = find_without_path_compression(i);path_compress(i,root);
    return root;
  }

  int merge(const int i,const int j) {
    int root_i = find_without_path_compression(i),
        root_j = find_without_path_compression(j);
    int root = ranks[root_i]>=ranks[root_j]?root_i:root_j;
    path_compress(i,root);
    path_compress(j,root);
    if (ranks[root_i]==ranks[root_j] && root_i!=root_j) ranks[root]++;
    return root;
  }

  template<class TArray> int merge(const TArray& indices) {
    STATIC_ASSERT_SAME(typename TArray::Element,int);
    int root = find_without_path_compression(indices[0]);
    Rank max_rank = ranks[root];
    bool max_tie=false;
    for (int i=1;i<indices.size();i++) {
      int root_i = find_without_path_compression(indices[i]);
      if (max_rank<ranks[root_i]) {
        max_rank = ranks[root_i];
        root = root_i;
        max_tie = false;
      } else if (max_rank==ranks[root_i] && root_i!=root)
        max_tie=true;
    }
    for (int i=0;i<indices.size();i++) path_compress(indices[i],root);
    if (max_tie) ranks[root]++;
    return root;
  }

  void merge(const UnionFindBase& union_find) {
    assert(size()==union_find.size());
    for (int i=0;i<size();i++) {
      int j = union_find.parents[i];
      if (j>=0) merge(i,j);
    }
  }

  int roots() const {
    int roots = 0;
    for (int i=0;i<parents.size();i++)
      roots += is_root(i);
    return roots;
  }

private:
  int find_without_path_compression(const int i) const {
    int j = i;
    while (parents[j]>=0) j = parents[j];
    return j;
  }

  void path_compress(const int i,const int root) const {
    int j = i;
    while (j!=root) {
      int parent = parents[j];
      parents[j] = root;
      j = parent;
      if (j<0) break;
    }
  }
};

class UnionFind : public UnionFindBase<UnionFind,Array<int>,Array<unsigned char> > {
public:
  explicit UnionFind(const int entries=0) {
    initialize(entries);
  }

  void initialize(const int entries) {
    parents.copy(ConstantMap<int>(entries,-1));
    ranks.copy(ConstantMap<Rank>(entries,0));
  }

  int add_entry() {
    parents.append(-1);
    return ranks.append(0);
  }

  // Okay for map to yield invalid indices for isolated elements
  template<class TArray> void mapped_merge(const UnionFind& union_find,const TArray& map) {
    for (int i=0;i<union_find.size();i++) {
      int root = union_find.find(i);
      if (i!=root) merge(map(i),map(root));
    }
  }

  void forest_edges(Array<Tuple<int,int> >& pairs) const {
    pairs.clear();
    for (int i=0;i<size();i++) {
      int j = find(i);
      if (i!=j) pairs.append(tuple(i,j));
    }
  }

  void merge_forest_edges(RawArray<const Tuple<int,int> > pairs) {
    for (int i=0;i<pairs.m;i++)
      merge(pairs(i).x,pairs(i).y);
  }
};

template<int n> struct SmallUnionFind : public UnionFindBase<SmallUnionFind<n>,Vector<signed char,n>,Vector<unsigned char,n> > {
  BOOST_STATIC_ASSERT(0<=n && n<=128);
  
  SmallUnionFind() {
    this->parents.fill(-1);
  }
};

}
