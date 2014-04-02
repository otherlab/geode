#include <geode/config.h>
#ifdef GEODE_OPENMESH

#include <geode/openmesh/triangulator.h>
#include <geode/structure/Tuple.h>
#include <geode/mesh/TriangleSoup.h>
#include <geode/array/Array.h>
#include <geode/utility/stl.h>
#include <geode/geometry/Segment.h>
namespace geode {

// make a reasonably well-triangulated face from a set of vertices.
int triangulate_face(TriMesh &mesh, std::vector<VertexHandle> const &face,
                     std::vector<FaceHandle> &faces,
                     EdgePriority &ep, bool debug, int depth) {

  assert(face.size() >= 3);

  if (face.size() == 3) {
    FaceHandle fh = mesh.add_face(face[0], face[1], face[2]);
    if (fh.is_valid()) {
      faces.push_back(fh);
      return 1;
    } else {
      if (debug) {
        std::cout << "inserting face " << face << " failed." << std::endl;
        fh = mesh.add_face(face[0], face[2], face[1]);

        if (fh.is_valid()) {
          std::cout << "but inserting " << vec(face[0], face[2], face[1]) << " worked. " << std::endl;
          return 1;
        }

        std::cout << "inserting as isolated face." << std::endl;
        mesh.request_face_colors();
        mesh.request_vertex_colors();
        Vector<VertexHandle, 3> verts;
        for (int i = 0; i < 3; ++i) {
          TriMesh::Point p = mesh.point(face[i]);
          verts[i] = mesh.add_vertex(p);
          mesh.set_color(verts[i], TriMesh::Color(0, 0, 0, 255));
        }
        faces.push_back(mesh.add_face(verts[0], verts[1], verts[2]));
        mesh.set_color(faces.back(), TriMesh::Color(0, 0, 0, 255));
      }
      return 0;
    }
  }

  // find best non-edge (must not be consecutive, and must not exist in the mesh)
  double best = -1;
  int best1 = -1, best2 = -1;
  for (unsigned int i = 0; i < face.size() - 1; ++i) {
    for (unsigned int j = i+2; j < face.size() - (i == 0 ? 1 : 0); ++j) {
      double priority = ep(face[i], face[j]);

      // negative priority -> forbidden
      if (priority < 0)
        continue;

      if (priority > best) {
        best = priority;
        best1 = i;
        best2 = j;
      }
    }
  }

  if (best1 == -1)
    return 0;

  // check if either of the end points is ambiguous
  std::vector<int> copies1, copies2;
  for (unsigned int i = 0; i < face.size(); ++i) {
    if (face[i] == face[best1])
      copies1.push_back(i);
    if (face[i] == face[best2])
      copies2.push_back(i);
  }

  assert(!copies1.empty() && !copies2.empty());

  // chose the two vertices that are closest together
  int d = std::numeric_limits<int>::max();
  for (unsigned int i = 0; i < copies1.size(); ++i) {
    for (unsigned int j = 0; j < copies2.size(); ++j) {
      if (abs(copies1[i] - copies2[j]) < d && abs(copies1[i] - copies2[j]) > 1) {
        d = abs(copies1[i] - copies2[j]);
        best1 = std::min(copies1[i], copies2[j]);
        best2 = std::max(copies1[i], copies2[j]);
      }
    }
  }

  assert(best1 < best2);

  // make two new faces, separated by the shortest edge
  std::vector<VertexHandle> face1, face2;

  face1.insert(face1.end(), face.begin(), face.begin() + best1 + 1);
  face1.insert(face1.end(), face.begin() + best2, face.end());

  face2.insert(face2.end(), face.begin() + best1, face.begin() + best2 + 1);

  if (debug) {
    std::cout << "splitting at edge " << face[best1] << " -- " << face[best2] << " with priority " << best << ", new sizes " << face1.size() << " and " << face2.size() << std::endl;
    std::cout << "  boundary indices: " << copies1 << " and " << copies2 << ", chose " << best1 << " and " << best2 << std::endl;
    assert(mesh.edge_handle(face[best1], face[best2]) == TriMesh::InvalidEdgeHandle);
  }

  ep.used_edge(face[best1], face[best2]);

  int n1 = 0, n2 = 0;

  n1 = triangulate_face(mesh, face1, faces, ep, debug, depth+1);
  n2 = triangulate_face(mesh, face2, faces, ep, debug, depth+1);

  return n1 + n2;
}

int triangulate_cylinder(TriMesh &mesh, std::vector<VertexHandle> const &ring1,
                         std::vector<VertexHandle> const &ring2,
                         std::vector<FaceHandle> &faces,
                         EdgePriority &ep, bool debug) {

  int nbegin = (int)faces.size();

  // find the best edge starting at ring1[0]
  int besti1 = -1, besti2 = -1;
  double bestedgequality = 0;
  //for (int i = 0; i < (int)loops[0].size(); ++i) {
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < (int)ring2.size(); ++j) {
      double q = ep(ring1[i],
                    ring2[j]);

      if (bestedgequality < q) {
        bestedgequality = q;
        besti1 = i;
        besti2 = j;
      }
    }
  }

  if (besti1 == -1 || besti2 == -1) {
    assert(false);
    return 0;
  }

  if (true || debug)
    std::cout << "connecting cylinder with edge " << besti1 << " -- " << besti2 << ", quality " << bestedgequality << ", l = " << mesh.point(ring1[besti1]) - mesh.point(ring2[besti2]) << std::endl;

  bool atstart[2] = {true, true};
  int lasti1 = besti1, lasti2 = besti2;
  do {
    Vector<real, 2> qualities(-1., -1.);

    int nexti1 = (lasti1+1) % ring1.size();
    int nexti2 = (lasti2+ring2.size()-1) % (int)ring2.size();

    // allowed to move i1
    if (atstart[0] || lasti1 != besti1) {
      qualities[0] = ep(ring1[nexti1], ring2[lasti2]);
    }

    // allowed to move i2
    if (atstart[1] || lasti2 != besti2) {
      qualities[1] = ep(ring1[lasti1], ring2[nexti2]);
    }

    if (debug)
      std::cout << "walking, qualities " << qualities[0] << ", " << qualities[1] << std::endl;

    // create the face with higher quality
    if (qualities[0] > qualities[1]) {
      if (debug) {
        std::cout << "walking loop 0 (idx " << nexti1 << " " << lasti2 << ", v " << ring1[nexti1] << " -- " << ring2[lasti2] << ", qualities " << qualities[0] << " " << qualities[1] << ")" << std::endl;
        std::cout << "adding face " << vec(ring1[lasti1], ring1[nexti1], ring2[lasti2]) << ", quality " << qualities[0] << " (other " << qualities[1] << ") on boundary 0" << std::endl;
      }
      faces.push_back(mesh.add_face(ring1[lasti1], ring1[nexti1], ring2[lasti2]));
      lasti1 = nexti1;
      atstart[0] = false;
    } else if (qualities[1] != -1) {
      if (debug) {
        std::cout << "walking loop 1 (idx " << lasti1 << " " << nexti2 << ", v " << ring1[lasti1] << " -- " << ring2[nexti2] << ", qualities " << qualities[0] << " " << qualities[1] << ")" << std::endl;
        std::cout << "adding face " << vec(ring2[lasti2], ring1[lasti1], ring2[nexti2]) << ", quality " << qualities[1] << " (other " << qualities[0] << ") on boundary 1" << std::endl;
      }
      faces.push_back(mesh.add_face(ring2[lasti2], ring1[lasti1], ring2[nexti2]));
      lasti2 = nexti2;
      atstart[1] = false;
    } else {

      std::cout << "no valid edge found, advancing both" << std::endl;
      // no allowed edge found, advance both
      if (atstart[0] || lasti1 != besti1) {
        lasti1 = nexti1;
        atstart[0] = false;
      }

      if (atstart[1] || lasti2 != besti2) {
        lasti2 = nexti2;
        atstart[1] = false;
      }

      //assert(false);
      //return faces.size() - nbegin;
    }

    if (!faces.empty() && !faces.back().is_valid()) {
      std::cout << "failed (ring1: " << ring1.size() << ", ring2: " << ring2.size() << ")" << std::endl;
      faces.pop_back();
      //assert(false);
    }

    // one may be still at the start in the first step, but not both
  } while (lasti1 != besti1 || lasti2 != besti2);

  return (int)faces.size() - nbegin;
}

EdgePriority::EdgePriority() {}
EdgePriority::~EdgePriority() {}

CachedEdgePriority::CachedEdgePriority() {}
CachedEdgePriority::~CachedEdgePriority() {}

void CachedEdgePriority::used_edge(VertexHandle v1, VertexHandle v2) {
  cache[vec(v1,v2)] = -1;
  cache[vec(v2,v1)] = -1;
}

double CachedEdgePriority::operator()(VertexHandle v1, VertexHandle v2) {
  if (cache.count(vec(v1, v2)))
    return cache[vec(v1,v2)];
  if (cache.count(vec(v2, v1)))
    return cache[vec(v2,v1)];

  // not in cache. Must compute...
  cache[vec(v1,v2)] = computePriority(v1,v2);
  return cache[vec(v1,v2)];
}

ShortEdgePriority::ShortEdgePriority(TriMesh const &mesh): mesh(mesh) {}
ShortEdgePriority::~ShortEdgePriority() {}

double ShortEdgePriority::computePriority(VertexHandle v1, VertexHandle v2) {
  if (v1 == v2)
    return -1;

  if (mesh.edge_handle(v1, v2) != TriMesh::InvalidEdgeHandle)
    return -1;

  return 1./(mesh.point(v1) - mesh.point(v2)).magnitude();
}

}
#endif // GEODE_OPENMESH
