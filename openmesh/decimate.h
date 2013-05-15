#pragma once

#include <other/core/openmesh/TriMesh.h>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>

#ifdef USE_OPENMESH
namespace other {

// Decimater type
typedef OpenMesh::Decimater::DecimaterT<TriMesh> DecimaterT;

template<class Decimater>
class FaceQualityModule: public OpenMesh::Decimater::ModBaseT<Decimater> {

public:
  DECIMATING_MODULE(FaceQualityModule, Decimater, FaceQuality);

  // prevents collapse of edges that would create bad (degenerate) triangles

  // this is a binary module only

protected:
  double minquality;

  // quality measure for faces -- 0 for degenerate, 1 for equilateral
  static double modified_face_quality(Mesh const &mesh, typename Mesh::FaceHandle const &fh, typename Mesh::VertexHandle const &oldv, typename Mesh::VertexHandle const &newv) {

    assert(mesh.is_trimesh());

    Triangle<Vector<real,3> > T;
    int i = 0;
    for (typename Mesh::ConstFaceVertexIter it = mesh.cfv_iter(fh); it; ++it) {
      if (it.handle() == oldv)
        T.X(i) = mesh.point(newv);
      else
        T.X(i) = mesh.point(it.handle());

      i++;
    }

    return T.quality();
  }

public:
  FaceQualityModule(Decimater& dec) : OpenMesh::Decimater::ModBaseT<Decimater>(dec, true), minquality(1e-5) {}

  double min_quality() { return minquality; }
  void min_quality(double quality) { minquality = quality; }

  virtual float collapse_priority(const OpenMesh::Decimater::CollapseInfoT<Mesh> &ci) {

    bool collapse_allowed = true;

    // check if any of the faces incident to the moved vertex will be of bad quality after the collapse
    Mesh const &mesh = OpenMesh::Decimater::ModBaseT<Decimater>::mesh();
    for (typename Mesh::ConstVertexFaceIter vf = mesh.cvf_iter(ci.v0); vf; ++vf) {
      if (vf.handle() != ci.fl && vf.handle() != ci.fr) {
        if (modified_face_quality(mesh, vf.handle(), ci.v0, ci.v1) < minquality) {
          collapse_allowed = false;
          break;
        }
      }
    }

    if (collapse_allowed)
      return OpenMesh::Decimater::ModBaseT<Decimater>::LEGAL_COLLAPSE;
    else
      return OpenMesh::Decimater::ModBaseT<Decimater>::ILLEGAL_COLLAPSE;
  }

private:
  // hide this method
  void set_binary(bool) {};

};



template<class Decimater>
class BoundaryPreservationModule: public OpenMesh::Decimater::ModBaseT<Decimater> {

public:
  DECIMATING_MODULE(BoundaryPreservationModule, Decimater, BoundaryPrevervation);

  // prevents collapse of boundary edges at a vertex whose boundary edges are not (very) collinear

  // this is a binary module only

protected:
  double max_error;

public:
  BoundaryPreservationModule(Decimater& dec)
    : OpenMesh::Decimater::ModBaseT<Decimater>(dec, true), max_error(0) {}

  void set_max_error(double error) {
    max_error = error;
  }

  virtual float collapse_priority(const OpenMesh::Decimater::CollapseInfoT<Mesh> &ci) {

    typedef Vector<real,3> TV;

    Mesh const &mesh = OpenMesh::Decimater::ModBaseT<Decimater>::mesh();

    typename Mesh::HalfedgeHandle last, next;

    if (mesh.is_boundary(ci.v0v1)) {
      last = mesh.prev_halfedge_handle(ci.v0v1);
      next = ci.v0v1;
    } else if (mesh.is_boundary(ci.v1v0)) {
      last = ci.v1v0;
      next = mesh.next_halfedge_handle(ci.v1v0);
    } else {
      // no boundary, always allowed
      return OpenMesh::Decimater::ModBaseT<Decimater>::LEGAL_COLLAPSE;
    }

    // Measure distance from middle point to segment between outer points
    const real error = Segment<TV>(mesh.point(mesh.from_vertex_handle(last)),
                                   mesh.point(mesh.to_vertex_handle(next)))
                                  .distance(mesh.point(mesh.from_vertex_handle(next)));

    return float(error < max_error ? OpenMesh::Decimater::ModBaseT<Decimater>::LEGAL_COLLAPSE
                                   : OpenMesh::Decimater::ModBaseT<Decimater>::ILLEGAL_COLLAPSE);
  }

private:
  // hide this method
  void set_binary(bool) {};

};

// Decimate by collapsing edges, prioritized by quadric error, until no more
// collapsible edges are found. The angle error is the maximum allowed change
// normal of any of the faces, the max quadric error is a maximum allowed
// error (unit is length), roughly corresponding to maximum hausdorff distance,
// it will not create faces worse than min_face_quality, and it will not collapse
// edges on the boundary if a point would move more than the max quadric error.
// Returns the number of vertices collapsed.
OTHER_CORE_EXPORT int decimate(TriMesh &mesh,
                               int max_collapses,
                               double maxangleerror,
                               double maxquadricerror,
                               double min_face_quality);

}
#endif // USE_OPENMESH
