#ifdef USE_OPENMESH
#include <other/core/openmesh/decimate.h>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalFlippingT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <other/core/geometry/Triangle3d.h>
#include <other/core/python/module.h>
namespace other {

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
          //std::cerr << "not allowing collapse of " << ci.v0 << " -> " << ci.v1 << " because quality " << modified_face_quality(mesh, vf.handle(), ci.v0, ci.v1) << " for face " << get_vertex_handles(mesh, vf.handle()) << " would suck." << std::endl;
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
  double mindot;

public:
  BoundaryPreservationModule(Decimater& dec) : OpenMesh::Decimater::ModBaseT<Decimater>(dec, true), mindot(1e-5) {}

  double min_dot() { return mindot; }
  void min_dot(double dot) { mindot = dot; }

  virtual float collapse_priority(const OpenMesh::Decimater::CollapseInfoT<Mesh> &ci) {

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

    // compute dot between last and next
    typename Mesh::Normal vl, vn;
    mesh.calc_edge_vector(last, vl);
    mesh.calc_edge_vector(next, vn);
    vl.normalize();
    vn.normalize();

    if (dot(vl, vn) > mindot)
      return OpenMesh::Decimater::ModBaseT<Decimater>::LEGAL_COLLAPSE;
    else
      return OpenMesh::Decimater::ModBaseT<Decimater>::ILLEGAL_COLLAPSE;
  }

private:
  // hide this method
  void set_binary(bool) {};

};



void decimate(TriMesh &mesh, int max_collapses, double maxangleerror, double maxquadricerror, double min_face_quality, double min_boundary_dot) {

  // need normals for this
  mesh.request_face_normals();
  mesh.update_face_normals();

  // Decimater type
  typedef OpenMesh::Decimater::DecimaterT<TriMesh> DecimaterT;
  DecimaterT decimater(mesh);

  // module types
  typedef OpenMesh::Decimater::ModNormalFlippingT<DecimaterT>::Handle HModNormalFlipping;
  typedef OpenMesh::Decimater::ModQuadricT<DecimaterT>::Handle HModQuadric;

  // Quadric
  HModQuadric hModQuadric;
  decimater.add(hModQuadric);

  if (maxquadricerror == std::numeric_limits<double>::infinity())
    decimater.module(hModQuadric).unset_max_err(); // use only as priority
  else
    decimater.module(hModQuadric).set_max_err(sqrt(maxquadricerror));

  // prevent ruining the boundary
  if (min_boundary_dot > -1) {
    BoundaryPreservationModule<DecimaterT>::Handle hModBoundary;
    decimater.add(hModBoundary);
    decimater.module(hModBoundary).min_dot(min_boundary_dot);
  }

  // prevent creation of crappy triangles
  if (min_face_quality > 0.) {
    FaceQualityModule<DecimaterT>::Handle hModFaceQuality;
    decimater.add(hModFaceQuality);
    decimater.module(hModFaceQuality).min_quality(min_face_quality);
  }

  // normal change termination criterion
  if (maxangleerror > 0.) {
    HModNormalFlipping hModNormalFlipping;
    decimater.add(hModNormalFlipping);
    decimater.module(hModNormalFlipping).set_normal_deviation((float)maxangleerror);
  }

  if (!decimater.initialize()) {
    std::cerr << "ERROR: could not initialize decimation." << std::endl;
    return;
  }

  decimater.decimate(max_collapses);

  mesh.release_face_normals();

  mesh.garbage_collection();
}

}
using namespace other;

void wrap_decimate() {
  OTHER_FUNCTION(decimate)
}
#endif // USE_OPENMESH
