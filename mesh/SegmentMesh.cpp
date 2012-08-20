//#####################################################################
// Class SegmentMesh
//#####################################################################
#include <other/core/mesh/SegmentMesh.h>
#include <other/core/array/NestedArray.h>
#include <other/core/array/sort.h>
#include <other/core/array/view.h>
#include <other/core/python/Class.h>
namespace other{

typedef Vector<real,2> TV2;
OTHER_DEFINE_TYPE(SegmentMesh)
const int SegmentMesh::d;

SegmentMesh::
SegmentMesh(Array<const Vector<int,2> > elements)
    :vertices(scalar_view_own(elements)),elements(elements)
{
    // assert validity and compute counts
    node_count=0;
    for(int i=0;i<vertices.size();i++){
        OTHER_ASSERT(vertices[i]>=0);
        node_count=max(node_count,vertices[i]+1);}
}

SegmentMesh::
~SegmentMesh()
{}

NestedArray<const int> SegmentMesh::
neighbors() const {
  if (nodes() && !neighbors_.size()) {
    Array<int> lengths(nodes());
    for(int s=0;s<elements.size();s++)
      for(int a=0;a<2;a++)
        lengths[elements[s][a]]++;
    neighbors_ = NestedArray<int>(lengths);
    for(int s=0;s<elements.size();s++) {
      int i,j;elements[s].get(i,j);
      neighbors_(i,neighbors_.size(i)-lengths[i]--) = j;
      neighbors_(j,neighbors_.size(j)-lengths[j]--) = i;
    }
    // Sort and remove duplicates if necessary
    bool need_copy = false;
    for(int i=0;i<nodes();i++) {
      RawArray<int> n = neighbors_[i]; 
      sort(n);
      int* last = std::unique(n.begin(),n.end());
      if(last!=n.end())
        need_copy = true;
      lengths[i] = int(last-n.begin());
    }
    if (need_copy) {
      NestedArray<int> copy(lengths);
      for(int i=0;i<nodes();i++)
        copy[i] = neighbors_[i].slice(0,lengths[i]);
      neighbors_ = copy;
    }
  }
  return neighbors_;
}

NestedArray<const int> SegmentMesh::
incident_elements() const
{
    if(nodes() && !incident_elements_.size()){
        Array<int> lengths(nodes());
        for(int i=0;i<vertices.size();i++)
            lengths[vertices[i]]++;
        incident_elements_=NestedArray<int>(lengths);
        for(int s=0;s<elements.size();s++) for(int i=0;i<2;i++){
            int p=elements[s][i];
            incident_elements_(p,incident_elements_.size(p)-lengths[p]--)=s;}}
    return incident_elements_;
}

Array<const Vector<int,2> > SegmentMesh::
adjacent_elements() const
{
  if (!adjacent_elements_.size() && nodes()) {
    adjacent_elements_.resize(elements.size(),false,false);
    NestedArray<const int> incident = incident_elements(); 
    for (int s=0;s<elements.size();s++) {
      Vector<int,2> seg = elements[s];
      for (int i=0;i<2;i++) {
        for (int s2 : incident[seg[i]])
          if (elements[s2][i]!=seg[i]) {
            adjacent_elements_[s][i] = s2;
            goto found;
          }
        adjacent_elements_[s][i] = -1;
        found:; 
      }
    }
  }
  return adjacent_elements_;
}

Array<TV2> SegmentMesh::
element_normals(RawArray<const TV2> X) const
{
    OTHER_ASSERT(X.size()>=nodes());
    Array<TV2> normals(elements.size(),false);
    for(int t=0;t<elements.size();t++){
        int i,j;elements[t].get(i,j);
        normals[t] = (X[j]-X[i]).rotate_right_90().normalized();}
    return normals;
}

Array<int> SegmentMesh::
nonmanifold_nodes(bool allow_boundary) const {
  Array<int> nonmanifold;
  NestedArray<const int> incident_elements = this->incident_elements();
  for (int i=0;i<incident_elements.size();i++) {
    RawArray<const int> incident = incident_elements[i];
    if (   incident.size()>2 // Too many segments
        || (incident.size()==1 && !allow_boundary) // Disallowed boundary
        || (incident.size()==2 && (elements[incident[0]][0]==i)==(elements[incident[1]][0]==i))) // Inconsistent orientations
      nonmanifold.append(i);
  }
  return nonmanifold;
}

}
using namespace other;

void wrap_segment_mesh()
{
    typedef SegmentMesh Self;
    Class<Self>("SegmentMesh")
        .OTHER_INIT(Array<const Vector<int,2> >)
        .OTHER_FIELD(d)
        .OTHER_FIELD(vertices)
        .OTHER_FIELD(elements)
        .OTHER_METHOD(segment_mesh)
        .OTHER_METHOD(incident_elements)
        .OTHER_METHOD(adjacent_elements)
        .OTHER_METHOD(nodes)
        .OTHER_METHOD(neighbors)
        .OTHER_METHOD(element_normals)
        .OTHER_METHOD(nonmanifold_nodes)
        ;
}
