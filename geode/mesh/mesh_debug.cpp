#include <geode/mesh/mesh_debug.h>
#include <geode/geometry/ParticleTree.h>

namespace geode {

int count_degenerate_edges(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X, const real epsilon) {
  int degenerate_edges = 0;
  for(const HalfedgeId hid : mesh.halfedges()) {
    if(mesh.reverse(hid) > hid) continue; // Only visit each edge once
    if(mesh.edge_length(X, hid) < epsilon) {
      degenerate_edges += 1;
    }
  }
  return degenerate_edges;
}

int count_degenerate_faces(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X, const real epsilon) {
  int degenerate_faces = 0;
  for(const FaceId fid : mesh.faces()) {
    if(mesh.area(X, fid) < epsilon) {
      degenerate_faces += 1;
    }
  }
  return degenerate_faces;
}


real mesh_volume(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X) {
  assert(!mesh.has_boundary());
  real sum = 0;
  for(const FaceId f : mesh.faces()) {
    const auto verts = mesh.vertices(f);
    sum += det(X[verts[0]],X[verts[1]],X[verts[2]]);
  }
  return real(1./6)*sum;
}

real mesh_volume(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X, const RawArray<const FaceId> component_faces) {
  real sum = 0;
  for(const FaceId f : component_faces) {
    const auto verts = mesh.vertices(f);
    sum += det(X[verts[0]],X[verts[1]],X[verts[2]]);
  }
  return real{1./6.}*sum;
}

NestedField<FaceId, ComponentId> get_component_faces(const TriangleTopology& mesh) {
  Field<ComponentId, FaceId> labels{mesh.allocated_faces()};
  Nested<FaceId,false> result;
  result.flat.preallocate(mesh.n_faces());

  Array<FaceId> queue;
  for(const FaceId seed : mesh.faces()) {
    if(labels[seed].valid()) continue; // Skip any that are already set
    const ComponentId current_component = ComponentId{result.size()};
    labels[seed] = current_component;
    queue.append(seed);
    result.append_empty();
    do {
      for(const FaceId f : mesh.faces(queue.pop())) {
        if(!f.valid()) continue; // Watch out for boundary
        if(!labels[f].valid()) {
          labels[f] = current_component;
          queue.append(f);
          result.append_to_back(f);
        }
      }
    } while(!queue.empty());
  }
  return {result.freeze()};
}

Nested<VertexId> get_unconnected_clusters(const TriangleTopology& mesh, const RawField<const Vector<real,3>, VertexId> X, const real epsilon) {
  Array<Vector<real,3>> non_erased_verts; // Dense packing of vertex positions (needed for ParticleTree)
  Array<VertexId> ids; // Mapping from dense indexes back to original ides 
  for(const VertexId vid : mesh.vertices()) {
    non_erased_verts.append(X[vid]);
    ids.append(vid);
  }

  const int n = non_erased_verts.size();
  GEODE_ASSERT(n == mesh.n_vertices());
  // This builds a mapping from points to clusters of adjacent points
  Array<int> duplicates = new_<ParticleTree<Vector<real,3>>>(non_erased_verts, 4)->remove_duplicates(epsilon);
  GEODE_ASSERT(non_erased_verts.size() == n);
  GEODE_ASSERT(duplicates.size() == n);

  // We invert mapping to get groups of VertexIds that are all in the same cluster
  // We also build map from VertexIds to clusters
  Array<int> cluster_sizes;
  for(const int cluster : duplicates) {
    if(cluster_sizes.size() <= cluster) {
      cluster_sizes.resize(cluster+1); // Ensure sizes array is big enough
    }
    cluster_sizes[cluster] += 1; // Count additional duplicate
  }
  Nested<VertexId> clusters = {cluster_sizes,uninit};
  const auto get_cluster = Field<int,VertexId>{mesh.allocated_vertices(),uninit};
  get_cluster.flat.fill(-1);
  for(const int i : range(n)) {
    const int cluster = duplicates[i];
    const VertexId id = ids[i];
    int& cluster_index = cluster_sizes[cluster];
    --cluster_index;
    clusters[cluster][cluster_index] = id;
    get_cluster[id] = cluster;
  }
  // We want to find any clusters that contain disjoint components
  // I.e. For each cluster, does traversing edges between vertices in the same cluster reach every vertex in that cluster?
  // We flood from first vertex in a cluster along edges within that cluster to see if entire cluster gets marked
  const auto cluster_roots = Field<int,VertexId>{mesh.allocated_vertices(),uninit};
  cluster_roots.flat.fill(-1);
  Array<VertexId> queue;
  Nested<VertexId,false> result;
  for(const int cluster_index : range(clusters.size())) {
    const auto& cluster_verts = clusters[cluster_index];
    const VertexId root = cluster_verts.front();
    cluster_roots[root] = cluster_index;
    queue.append(root);
    while(!queue.empty()) {
      const VertexId src = queue.pop();
      for(const HalfedgeId e : mesh.outgoing(src)) {
        const VertexId dst = mesh.dst(e);
        if(get_cluster[src] != get_cluster[dst]) continue; // Only consider edges that remain in the cluster
        assert(cluster_roots[dst] == -1 || cluster_roots[dst] == cluster_index); // Should have same root or be uninitialized
        if(cluster_roots[dst] == -1) {
          // Found another edge in the cluster
          cluster_roots[dst] = cluster_index;
          queue.append(dst);
        }
      }
    }
    // Now that we've flooded as far as possible from first vertex in cluster, see if we covered the entire cluster
    for(const VertexId v : cluster_verts) {
      if(cluster_roots[v] != cluster_index) {
        assert(cluster_roots[v] == -1); // Should just be uninitialized
        result.append(cluster_verts); // This as a new unconnected cluster
        break;
      }
    }
  }
  return result.freeze();
}

bool has_duplicate_faces(const TriangleTopology& mesh) {
  Hashtable<Vector<VertexId,3>,FaceId> face_map;
  for(const FaceId f : mesh.faces()) {
    const auto f_verts = mesh.vertices(f).sorted();
    if(face_map.contains(f_verts)) {
      return true;
    }
    face_map.set(f_verts, f);
  }
  return false;
}



} // geode namespace