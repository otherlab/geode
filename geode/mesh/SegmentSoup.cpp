//#####################################################################
// Class SegmentSoup
//#####################################################################
#include <geode/mesh/SegmentSoup.h>
#include <geode/array/Nested.h>
#include <geode/array/sort.h>
#include <geode/array/view.h>
#include <geode/python/Class.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/stl.h>
#include <geode/vector/convert.h>
#include <geode/vector/normalize.h>
#include <vector>
namespace geode {

using std::vector;
typedef Vector<real,2> TV2;

GEODE_DEFINE_TYPE(SegmentSoup)

#ifndef _WIN32
const int SegmentSoup::d;
#endif

SegmentSoup::SegmentSoup(Array<const Vector<int,2> > elements)
  : vertices(scalar_view_own(elements))
  , elements(elements)
  , node_count(compute_node_count())
  , bending_tuples_valid(false) {}

int SegmentSoup::compute_node_count() const {
  // Assert validity and compute counts
  int c = 0;
  for (int i=0;i<vertices.size();i++) {
    if (vertices[i] < 0)
      throw ValueError(format("SegmentSoup: invalid negative vertex %d",vertices[i]));
    c = max(c,vertices[i]+1);
  }
  return c;
}

SegmentSoup::~SegmentSoup() {}

const Tuple<Nested<const int>,Nested<const int>>& SegmentSoup::polygons() const {
  if (nodes() && !polygons_.x.size() && !polygons_.y.size()) {
    const auto incident = incident_elements();
    // Start from each segment, compute the contour that contains it and classify as either closed or open
    vector<vector<int>> closed, open;
    vector<bool> traversed(elements.size()); // Which segments have we covered already?
    for (int seed : range(elements.size())) {
      if (traversed[seed])
        continue;
      const int start = elements[seed].x;
      vector<int> poly;
      poly.push_back(start);
      for (int node=start,segment=seed;;) {
        traversed[segment] = true;
        const int other = elements[segment][elements[segment].x==node?1:0];
        if (other==start) { // Found a closed contour
          closed.push_back(poly);
          break;
        }
        if (incident[other].size()!=2) { // Found the end of a closed contour, or a nonmanifold vertex
          // Traverse backwards to fill in the entire open contour
          poly.clear();
          poly.push_back(other);
          for (int node2=other,segment2=segment;;) {
            traversed[segment2] = true;
            node2 = elements[segment2][elements[segment2].x==node2?1:0];
            poly.push_back(node2);
            if (incident[node2].size()!=2)
              break;
            segment2 = incident[node2][incident[node2][0]==segment2?1:0];
          }
          // If segments are oriented, preserve this order
          reverse(poly.begin(),poly.end());
          open.push_back(poly);
          break;
        }
        // Node other is manifold, so continue to the next segment
        node = other;
        poly.push_back(node);
        segment = incident[node][incident[node][0]==segment?1:0];
      }
    }
    // Store results
    polygons_ = tuple(Nested<const int>::copy(closed),Nested<const int>::copy(open));
  }
  return polygons_;
}

Nested<const int> SegmentSoup::neighbors() const {
  if (nodes() && !neighbors_.size()) {
    Array<int> lengths(nodes());
    for(int s=0;s<elements.size();s++)
      for(int a=0;a<2;a++)
        lengths[elements[s][a]]++;
    neighbors_ = Nested<int>(lengths);
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
      Nested<int> copy(lengths);
      for(int i=0;i<nodes();i++)
        copy[i] = neighbors_[i].slice(0,lengths[i]);
      neighbors_ = copy;
    }
  }
  return neighbors_;
}

Nested<const int> SegmentSoup::incident_elements() const {
  if (nodes() && !incident_elements_.size()) {
    Array<int> lengths(nodes());
    for (int i=0;i<vertices.size();i++)
      lengths[vertices[i]]++;
    incident_elements_=Nested<int>(lengths);
    for (int s=0;s<elements.size();s++) for(int i=0;i<2;i++) {
      int p=elements[s][i];
      incident_elements_(p,incident_elements_.size(p)-lengths[p]--)=s;
    }
  }
  return incident_elements_;
}

Array<const Vector<int,2> > SegmentSoup::adjacent_elements() const {
  if (!adjacent_elements_.size() && nodes()) {
    adjacent_elements_.resize(elements.size(),uninit);
    Nested<const int> incident = incident_elements();
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

Array<TV2> SegmentSoup::element_normals(RawArray<const TV2> X) const {
  GEODE_ASSERT(X.size()>=nodes());
  Array<TV2> normals(elements.size(),uninit);
  for (int t=0;t<elements.size();t++) {
    int i,j;elements[t].get(i,j);
    normals[t] = rotate_right_90(normalized(X[j]-X[i]));
  }
  return normals;
}

Array<int> SegmentSoup::nonmanifold_nodes(bool allow_boundary) const {
  Array<int> nonmanifold;
  Nested<const int> incident_elements = this->incident_elements();
  for (int i=0;i<incident_elements.size();i++) {
    RawArray<const int> incident = incident_elements[i];
    if (   incident.size()>2 // Too many segments
        || (incident.size()==1 && !allow_boundary) // Disallowed boundary
        || (incident.size()==2 && (elements[incident[0]][0]==i)==(elements[incident[1]][0]==i))) // Inconsistent orientations
      nonmanifold.append(i);
  }
  return nonmanifold;
}

Array<const Vector<int,3>> SegmentSoup::bending_tuples() const {
  if (!bending_tuples_valid) {
    Nested<const int> neighbors = this->neighbors();
    Array<Vector<int,3>> tuples;
    for (const int p : range(nodes())) {
      RawArray<const int> near = neighbors[p];
      for (int i=0;i<near.size();i++) for(int j=i+1;j<near.size();j++)
        tuples.append(vec(near[i],p,near[j]));
    }
    bending_tuples_ = tuples;
  }
  return bending_tuples_;
}

}
using namespace geode;

void wrap_segment_soup() {
  typedef SegmentSoup Self;
  Class<Self>("SegmentSoup")
    .GEODE_INIT(Array<const Vector<int,2> >)
    .GEODE_FIELD(d)
    .GEODE_FIELD(vertices)
    .GEODE_FIELD(elements)
    .GEODE_METHOD(segment_soup)
    .GEODE_METHOD(incident_elements)
    .GEODE_METHOD(adjacent_elements)
    .GEODE_METHOD(nodes)
    .GEODE_METHOD(neighbors)
    .GEODE_METHOD(element_normals)
    .GEODE_METHOD(nonmanifold_nodes)
    .GEODE_METHOD(polygons)
    .GEODE_METHOD(bending_tuples)
    ;
}
