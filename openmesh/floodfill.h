// Flood fills on TriMeshes
#pragma once

#include <other/core/openmesh/TriMesh.h>
#include <other/core/structure/Tuple.h>
namespace other {

using std::vector;

// Generic floodfill on vertices (across edges), starting at vh, returns the vertices reached.  EdgeStop = bool(EdgeHandle)
template<class EdgeStop> vector<VertexHandle> floodfill(const TriMesh& mesh, VertexHandle seed, const EdgeStop& stop);
template<class EdgeStop> vector<VertexHandle> floodfill(const TriMesh& mesh, RawArray<const VertexHandle> seeds, const EdgeStop& stop) OTHER_NEVER_INLINE;

// Generic floodfill starting at fh, returns the faces filled or the boundary or both
template<class EdgeStop> vector<FaceHandle> floodfill(const TriMesh& mesh, FaceHandle seed, const EdgeStop& stop);
template<class EdgeStop> vector<TriMesh::FaceHandle> floodfill(const TriMesh& mesh, RawArray<const FaceHandle> seeds, const EdgeStop& stop) OTHER_NEVER_INLINE;
template<class EdgeStop> unordered_set<HalfedgeHandle,Hasher> floodfill_boundary(const TriMesh& mesh, FaceHandle seed, const EdgeStop& stop) OTHER_NEVER_INLINE;
template<class EdgeStop> Tuple<vector<FaceHandle>,unordered_set<HalfedgeHandle,Hasher> > floodfill_data(const TriMesh& mesh, FaceHandle seed, const EdgeStop& stop) OTHER_NEVER_INLINE;

// Implementations follow

template<class EdgeStop> vector<VertexHandle> floodfill(const TriMesh& mesh, VertexHandle seed, const EdgeStop& stop) {
  return floodfill(mesh, RawArray<const VertexHandle>(1,&seed), stop);
}

template<class EdgeStop> vector<VertexHandle> floodfill(const TriMesh& mesh, RawArray<const VertexHandle> seeds, const EdgeStop& stop) {
  unordered_set<VertexHandle,Hasher> seen;
  vector<VertexHandle> front;
  for (auto seed : seeds)
    if (seen.insert(seed).second)
      front.push_back(seed);
  while (front.size()) {
    auto v = front.back();
    front.pop_back();
    for (auto voh = mesh.cvoh_iter(v); voh; ++voh) {
      VertexHandle v2 = mesh.to_vertex_handle(voh.handle());
      if (!stop(mesh.edge_handle(voh.handle())) && seen.insert(v2).second)
        front.push_back(v2);
    }
  }
  return vector<VertexHandle>(seen.begin(),seen.end());
}

template<class EdgeStop> vector<FaceHandle> floodfill(const TriMesh& mesh, FaceHandle seed, const EdgeStop& stop) {
  return floodfill(mesh, RawArray<const FaceHandle>(1,&seed), stop);
}

template<class EdgeStop> vector<FaceHandle> floodfill(const TriMesh& mesh, RawArray<const FaceHandle> seeds, const EdgeStop& stop) {
  unordered_set<FaceHandle,Hasher> seen;
  vector<FaceHandle> front;
  for (auto seed : seeds)
    if (seen.insert(seed).second)
      front.push_back(seed);
  while (front.size()) {
    auto f = front.back();
    front.pop_back();
    for (auto e : mesh.halfedge_handles(f)) {
      FaceHandle f2 = mesh.opposite_face_handle(e);
      if (f2.is_valid() && !stop(mesh.edge_handle(e)) && seen.insert(f2).second)
        front.push_back(f2);
    }
  }
  return vector<FaceHandle>(seen.begin(),seen.end());
}

template<class EdgeStop> void floodfill_helper(const TriMesh& mesh, FaceHandle seed, const EdgeStop& stop, unordered_set<FaceHandle,Hasher>& seen, unordered_set<HalfedgeHandle,Hasher>& boundary) {
  vector<FaceHandle> front;
  seen.insert(seed);
  front.push_back(seed);
  while (front.size()) {
    auto f = front.back();
    front.pop_back();
    for (auto e : mesh.halfedge_handles(f)) {
      FaceHandle f2 = mesh.opposite_face_handle(e);
      if (f2.is_valid() && !stop(mesh.edge_handle(e))) {
        if (seen.insert(f2).second)
          front.push_back(f2);
      } else
        boundary.insert(e);
    }
  }
}

template<class EdgeStop> unordered_set<HalfedgeHandle,Hasher> floodfill_boundary(const TriMesh& mesh, FaceHandle seed, const EdgeStop& stop) {
  unordered_set<FaceHandle,Hasher> seen;
  unordered_set<HalfedgeHandle,Hasher> boundary;
  floodfill_helper(mesh,seed,stop,seen,boundary);
  return boundary;
}

template<class EdgeStop> Tuple<vector<FaceHandle>,unordered_set<HalfedgeHandle,Hasher> > floodfill_data(const TriMesh& mesh, FaceHandle seed, const EdgeStop& stop) {
  unordered_set<FaceHandle,Hasher> seen;
  unordered_set<HalfedgeHandle,Hasher> boundary;
  floodfill_helper(mesh,seed,stop,seen,boundary);
  return tuple(vector<FaceHandle>(seen.begin(),seen.end()),boundary);
}

}
