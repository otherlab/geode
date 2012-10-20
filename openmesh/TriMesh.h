#pragma once

#include <other/core/mesh/forward.h>

// OpenMesh includes
#include <cstddef>
// Since we're using hidden symbol visibility, dynamic_casts across shared library
// boundaries are problematic. Therefore, don't use them even in debug mode.
#define OM_FORCE_STATIC_CAST
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Utils/vector_traits.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include <tr1/unordered_set>
#include <tr1/unordered_map>

#include <other/core/utility/config.h>
#include <other/core/utility/Hasher.h>
#include <other/core/math/lerp.h>
#include <other/core/image/color_utils.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/range.h>

#include <other/core/python/from_python.h>
#include <other/core/python/to_python.h>
#include <other/core/python/stl.h>
#include <other/core/python/Object.h>
#include <other/core/python/Ptr.h>
#include <other/core/python/Ref.h>

#include <other/core/array/Array.h>
#include <other/core/array/RawField.h>
#include <other/core/vector/Vector.h>

#include <other/core/geometry/Box.h>
#include <other/core/geometry/Plane.h>
#include <other/core/geometry/Triangle3d.h>
#include <other/core/geometry/Segment3d.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Hashtable.h>

namespace other {

using std::vector;

// make a vector adapter that inherits from Vector<double,3>
template<class T, int d>
class OVec: public Vector<T,d> {
public:

  enum {size_ = d};

  OVec(): Vector<T,d>() {};
  OVec(Vector<T,d> const &v): Vector<T,d>(v) {};
  OVec(double x): Vector<T,d>(x) {};
  OVec(T const& x): Vector<T,d>(x) {};
  OVec(T const& x, T const& y): Vector<T,d>(x,y) {};
  OVec(T const& x, T const& y, T const& z): Vector<T,d>(x,y,z) {};
  OVec(T const& x, T const& y, T const& z, T const& w): Vector<T,d>(x,y,z,w) {};
  template<class SV> explicit OVec(const SV& v): Vector<T,d>(v) {}

  // functions needed by OpenMesh

  OVec &vectorize(T const &c) {
    for (int i = 0; i < d; ++i) {
      (*this)[i] = c;
    }
    return *this;
  }

  inline real length() const {
    return norm();
  }

  inline real norm() const {
    return sqrt(sqrnorm());
  }

  inline T sqrnorm() const {
    return dot(*this, *this);
  }

  inline T operator|(OVec const &v) {
    return dot(*this, v);
  }
};

template<class T,int d> struct IsScalarBlock<OVec<T,d> >:public IsScalarBlock<Vector<T,d> >{};

// gcc declares a generic template for isfinite, so we need this forwarding function to get to our definition
template<class T,int d> inline bool isfinite(const OVec<T,d>& v) {
  return isfinite(static_cast<const Vector<T,d>&>(v));
}

}

// we have to insert vector_traits into OpenMesh to make TriMesh accept our vector class
namespace OpenMesh {

template <class T, int d>
struct vector_traits<other::OVec<T,d> >
{
  /// Type of the vector class
  typedef other::OVec<T,d> vector_type;

  /// Type of the scalar value
  typedef T value_type;

  /// size/dimension of the vector
  static const size_t size_ = d;

  /// size/dimension of the vector
  static size_t size() { return d; }
};

}

namespace other {

template<class T,int d>
struct FromPython<OVec<T,d> > {
  OTHER_EXPORT static OVec<T,d> convert(PyObject* object) {
    return OVec<T,d>(FromPython<Vector<T,d> >::convert(object));
  }
};

// declare TriMesh base class
struct MeshTraits : public OpenMesh::DefaultTraits
{
  typedef OVec<real,3> Point;
  typedef OVec<real,3> Normal;
  typedef OVec<real,1> TexCoord1D;
  typedef OVec<real,2> TexCoord2D;
  typedef OVec<real,3> TexCoord3D;
  typedef OVec<unsigned char, 4> Color;

  VertexAttributes(OpenMesh::Attributes::Status);
  FaceAttributes(OpenMesh::Attributes::Status);
  EdgeAttributes(OpenMesh::Attributes::Status);
  HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge | OpenMesh::Attributes::Status);
};

typedef OpenMesh::TriMesh_ArrayKernelT<MeshTraits> OTriMesh;
typedef OTriMesh::VertexHandle VertexHandle;
typedef OTriMesh::EdgeHandle EdgeHandle;
typedef OTriMesh::HalfedgeHandle HalfedgeHandle;
typedef OTriMesh::FaceHandle FaceHandle;

// For hashing
template<> struct is_packed_pod<OpenMesh::BaseHandle>:public mpl::true_{};
template<> struct is_packed_pod<OpenMesh::VertexHandle>:public mpl::true_{};
template<> struct is_packed_pod<OpenMesh::EdgeHandle>:public mpl::true_{};
template<> struct is_packed_pod<OpenMesh::HalfedgeHandle>:public mpl::true_{};
template<> struct is_packed_pod<OpenMesh::FaceHandle>:public mpl::true_{};
template<class T,int d> struct is_packed_pod<OVec<T,d>>:public is_packed_pod<Vector<T,d>>{};

template<class PropHandle> struct PropToHandle;
template<class T> struct PropToHandle<OpenMesh::VPropHandleT<T>> { typedef VertexHandle type; };
template<class T> struct PropToHandle<OpenMesh::FPropHandleT<T>> { typedef FaceHandle type; };
template<class T> struct PropToHandle<OpenMesh::EPropHandleT<T>> { typedef EdgeHandle type; };

template<class T,class Handle> struct HandleToProp;
template<class T> struct HandleToProp<T,VertexHandle> { typedef OpenMesh::VPropHandleT<T> type; };
template<class T> struct HandleToProp<T,FaceHandle>   { typedef OpenMesh::FPropHandleT<T> type; };
template<class T> struct HandleToProp<T,EdgeHandle>   { typedef OpenMesh::EPropHandleT<T> type; };

}

namespace OpenMesh {

// overloaded functions need to be in the same namespace as their arguments to be found by
// the compiler (or declared before the declaration of whatever uses them)

// python interface for handles
static inline PyObject* to_python(BaseHandle h) {
  return ::other::to_python(h.idx());
}

}

namespace other {

template<> struct FromPython<VertexHandle> {
  static VertexHandle convert(PyObject* object) {
    return VertexHandle((unsigned int)from_python<int>(object));
  }
};

template<> struct FromPython<FaceHandle> {
  static FaceHandle convert(PyObject* object) {
    return FaceHandle((unsigned int)from_python<int>(object));
  }
};

template<> struct FromPython<EdgeHandle> {
  static EdgeHandle convert(PyObject* object) {
    return EdgeHandle((unsigned int)from_python<int>(object));
  }
};

template<> struct FromPython<HalfedgeHandle> {
  static HalfedgeHandle convert(PyObject* object) {
    return HalfedgeHandle((unsigned int)from_python<int>(object));
  }
};

template<class P> struct prop_handle_type;
template<class T> struct prop_handle_type<OpenMesh::FPropHandleT<T> >{typedef FaceHandle type;};
template<class T> struct prop_handle_type<OpenMesh::EPropHandleT<T> >{typedef EdgeHandle type;};
template<class T> struct prop_handle_type<OpenMesh::VPropHandleT<T> >{typedef VertexHandle type;};
template<class T> struct prop_handle_type<OpenMesh::HPropHandleT<T> >{typedef HalfedgeHandle type;};

template<class Iter> struct HandleIter : public Iter {
  HandleIter(Iter self) : Iter(self) {}
  auto operator*() const -> decltype(boost::declval<Iter>().handle()) { return this->handle(); }
};

template<class Iter> static inline Range<HandleIter<Iter> > handle_range(Iter begin, Iter end) {
  return Range<HandleIter<Iter> >(HandleIter<Iter>(begin),HandleIter<Iter>(end));
}

// TriMesh class
class OTHER_EXPORT TriMesh: public Object, public OTriMesh {
public:
  OTHER_DECLARE_TYPE
  typedef Object Base;
  typedef real T;
  typedef Vector<T,3> TV;

  // add constructors and access functions to make the OpenMesh class more usable
protected:
  TriMesh();
  TriMesh(const TriMesh& m);
  TriMesh(RawArray<const Vector<int,3> > tris, RawArray<const Vector<real,3> > X);
  TriMesh(Tuple<Ref<TriangleMesh>,Array<Vector<real,3>>> const &);

  void cut_and_mirror(Plane<real> const &p, bool mirror, T epsilon, T area_hack);
public:
  virtual ~TriMesh();

public:
  // assign from another TriMesh
  TriMesh & operator=(OTriMesh const &o);

  // full copy
  Ref<TriMesh> copy() const;

  // load from file/stream
  void read(string const &filename);
  void read(std::istream &is, string const &ext);

  // save to file/stream
  void write(string const &filename) const;
  void write_with_normals(string const &filename) const;
  void write(std::ostream &os, string const &ext) const;

  // add a property
  template<class T,class Handle> typename HandleToProp<T,Handle>::type add_prop(const char* name) {
    typename HandleToProp<T,Handle>::type prop;
    add_property(prop,name);
    return prop;
  }

  // add a bunch of vertices
  void add_vertices(RawArray<const TV> X);

  // add a bunch of faces
  void add_faces(RawArray<const Vector<int,3> > faces);

  // add a mesh to this (vertices/faces)
  void add_mesh(TriMesh const &mesh);

  // bounding box
  Box<Vector<real,3> > bounding_box() const;
  Box<Vector<real,3> > bounding_box(std::vector<FaceHandle> const &faces) const;

  //centroid
  Vector<real,3> centroid();

  real mean_edge_length() const;

  // get a triangle area
  real area(FaceHandle fh) const;

  // get the face as a Triangle
  Triangle<Vector<real, 3> > triangle(FaceHandle fh) const;

  // get an edge as a Segment
  Segment<Vector<real, 3> > segment(EdgeHandle eh) const;
  Segment<Vector<real, 3> > segment(HalfedgeHandle heh) const;

  // get the vertex handles incident to the given edge
  Vector<VertexHandle,2> vertex_handles(EdgeHandle fh) const;

  // get the vertex handles incident to the given face
  Vector<VertexHandle,3> vertex_handles(FaceHandle fh) const;

  // get the edge handles incident to the given face
  Vector<EdgeHandle,3> edge_handles(FaceHandle fh) const;

  // get the halfedge handles that make up the given edge
  Vector<HalfedgeHandle,2> halfedge_handles(EdgeHandle eh) const;

  // get the halfedge handles incident to the given face
  Vector<HalfedgeHandle,3> halfedge_handles(FaceHandle fh) const;

  // get the face handles incident to the given edge (one of which is invalid if at boundary)
  Vector<FaceHandle, 2> face_handles(EdgeHandle eh) const;

  // get a valid face handle incident to an edge (invalid only if there is none at all,
  // in that case, the edge should have been deleted)
  FaceHandle valid_face_handle(EdgeHandle eh) const;

  // get all vertices in the one-ring
  std::vector<VertexHandle> vertex_one_ring(VertexHandle vh) const;

  // check whether the quad around an edge is convex (and the edge can be flipped safely)
  bool quad_convex(EdgeHandle eh) const;

  // re-publish overloads from OTriMesh
  inline EdgeHandle edge_handle(HalfedgeHandle he) const {
    return OTriMesh::edge_handle(he);
  }
  inline HalfedgeHandle halfedge_handle(VertexHandle ve) const {
    return OTriMesh::halfedge_handle(ve);
  }
  inline HalfedgeHandle halfedge_handle(FaceHandle fe) const {
    return OTriMesh::halfedge_handle(fe);
  }
  inline HalfedgeHandle halfedge_handle(EdgeHandle e, int i) const {
    return OTriMesh::halfedge_handle(e, i);
  }

  // get the handle of the edge between two vertices (invalid if none)
  EdgeHandle edge_handle(VertexHandle vh1, VertexHandle vh2) const;

  // get the handle of the halfedge starting at vh1 going to vh2
  HalfedgeHandle halfedge_handle(VertexHandle vh1, VertexHandle vh2) const;

  // get the handle of the halfedge pointing at vh, in face fh
  HalfedgeHandle halfedge_handle(FaceHandle fh, VertexHandle vh) const;

  // get the incident faces to a vertex
  std::vector<FaceHandle> incident_faces(VertexHandle vh) const;

  /// return the common edge connecting two faces
  EdgeHandle common_edge(FaceHandle fh, FaceHandle fh2) const {
    for (ConstFaceHalfedgeIter e = cfh_iter(fh); e; ++e)
      if (opposite_face_handle(e) == fh2)
        return edge_handle(e);
    return TriMesh::InvalidEdgeHandle;
  }

  // get a normal for an edge (average of incident face normals)
  Normal normal(EdgeHandle eh) const;

  // re-publish overloads from OTriMesh
  inline Normal normal(VertexHandle vh) const {
    return OTriMesh::normal(vh);
  }

  inline Normal normal(FaceHandle fh) const {
    return OTriMesh::normal(fh);
  }

  // get an interpolated normal at any point on the mesh
  Normal smooth_normal(FaceHandle fh, Vector<real,3> const &bary) const;

  T dihedral_angle(EdgeHandle e) const;

  // make a triangle fan
  std::vector<FaceHandle> triangle_fan(std::vector<VertexHandle> const &boundary, VertexHandle center, bool closed);

  // extract a set of faces as a new mesh and store vertex correspondence: id2id[old] = new
  Ref<TriMesh> extract_faces(std::vector<FaceHandle> const &faces,
                             std::tr1::unordered_map<VertexHandle, VertexHandle, Hasher> &id2id) const;

  inline Ref<TriMesh> extract_faces(std::vector<FaceHandle> const &faces) const {
    std::tr1::unordered_map<VertexHandle, VertexHandle, Hasher> id2id;
    return extract_faces(faces, id2id);
  }

  // extract the inverse of a set of faces as a new mesh and store vertex correspondence
  Ref<TriMesh> inverse_extract_faces(std::vector<FaceHandle> const &faces,
                             std::tr1::unordered_map<VertexHandle, VertexHandle, Hasher> &id2id) const;

  inline Ref<TriMesh> inverse_extract_faces(std::vector<FaceHandle> const &faces) const {
    std::tr1::unordered_map<VertexHandle, VertexHandle, Hasher> id2id;
    return inverse_extract_faces(faces, id2id);
  }
  
  // compute the 2D silhouettes of the mesh as seem from the given direction
  std::vector<std::vector<Vector<real,2>>> silhouette(Normal const &n) const;
  
  // get the halfedges bounding the given set of faces (for all halfedges, face_handle(he) is in faces)
  std::tr1::unordered_set<HalfedgeHandle, Hasher> boundary_of(std::vector<FaceHandle> const &faces) const;

  // compute the approximate geodesic distance from one point to another, and store
  // all values computed on the way (can be used to re-trace the approximate shortest path)
  std::tr1::unordered_map<VertexHandle, double, Hasher> geodesic_distance(VertexHandle source,
                                                                          VertexHandle sinks) const;

  // compute the approximate geodesic distance from one point to a set of points, and store
  // all values computed on the way (can be used to re-trace the approximate shortest paths)
  std::tr1::unordered_map<VertexHandle, double, Hasher> geodesic_distance(VertexHandle source,
                                                                          std::vector<VertexHandle> const &sinks) const;

  // compute the approximate geodesic distance from one point to a set of points, and store
  // all values computed on the way (can be used to re-trace the approximate shortest paths)
  std::tr1::unordered_map<VertexHandle, double, Hasher> geodesic_distance(std::vector<VertexHandle> const &sources,
                                                                          std::vector<VertexHandle> const &sinks) const;

  // compute and return the approximate shortest path from one point to another
  std::vector<VertexHandle> shortest_path(VertexHandle source, VertexHandle sink) const;

  // compute the closest face to a point by breadth-first search starting at the given vertex/face
  FaceHandle local_closest_face(Point const &p, FaceHandle start) const;
  FaceHandle local_closest_face(Point const &p, VertexHandle start) const;

  // cut the mesh with a plane (negative side will be removed)
  void cut(Plane<real> const &p, double epsilon = 1e-4, double area_hack = 0);

  // mirror the mesh at a plane (positive side will be mirrored, negative replaced)
  void mirror(Plane<real> const &p, double epsilon = 1e-4);

  // find boundary loops
  std::vector<std::vector<HalfedgeHandle> > boundary_loops() const;

  // find the boundary loop starting at seed (empty if seed is not on the boundary)
  std::vector<HalfedgeHandle> boundary_loop(HalfedgeHandle const &seed) const;

  // fill the hole enclosed by the given halfedges, retain the new faces only if the surface area is smaller than max_area
  std::vector<FaceHandle> fill_hole(std::vector<HalfedgeHandle> const &loop, double max_area = std::numeric_limits<double>::max());

  // fill all holes with maximum area given
  void fill_holes(double max_area = std::numeric_limits<double>::max());

  void add_box(TV min, TV max);
  void add_sphere(TV c, real r, int divisions = 30);
  void add_cylinder(TV p1, TV p2, real r1, real r2, int divisions = 30, bool caps = true);

  void scale(real scale, const Vector<real, 3>& center);
  void scale(TV scale, const Vector<real, 3>& center);
  void translate(Vector<real,3> const &t);
  void rotate(Rotation<Vector<real, 3> > const &R, Vector<real,3> const &center);

  // flip all faces inside out
  void invert();

  real volume() const;
  real area() const;
  real area(RawArray<const FaceHandle> faces) const;

  // Warning: these construct new arrays or copy memory
  Array<Vector<int,3> > elements() const;
  Array<Vector<real,3> > X_python() const;
  void set_X_python(RawArray<const Vector<real,3>> new_X);
  void set_vertex_normals(RawArray<const Vector<real,3>> normals);
  void set_vertex_colors(RawArray<const Vector<real,3>> colors);

  // Warning: reference goes invalid if the mesh is changed
  RawArray<Vector<real,3> > X();
  RawArray<const Vector<real,3> > X() const;

  // Warning: reference goes invalid if the mesh is changed
  template<class PropHandle> RawField<typename PropHandle::Value,typename prop_handle_type<PropHandle>::type> prop(PropHandle p) {
    return RawField<typename PropHandle::Value,typename prop_handle_type<PropHandle>::type>(property(p).data_vector());
  }

  // Warning: reference goes invalid if the mesh is changed
  template<class PropHandle> RawField<const typename PropHandle::Value,typename prop_handle_type<PropHandle>::type> prop(PropHandle p) const {
    return RawField<const typename PropHandle::Value,typename prop_handle_type<PropHandle>::type>(const_cast_(property(p)).data_vector());
  }

  // Component analysis
  Tuple<int,Array<int> > component_vertex_map() const;
  Array<int> component_face_map() const;

  // Split a mesh into connected components
  vector<Ref<TriMesh> > component_meshes() const;
  vector<Ref<TriMesh> > nested_components() const;

  // Convenience functions for use in range-based for loops
  Range<HandleIter<VertexIter>> vertex_handles() { return handle_range(vertices_begin(),vertices_end()); }
  Range<HandleIter<ConstVertexIter>> vertex_handles() const { return handle_range(vertices_begin(),vertices_end()); }
  Range<HandleIter<EdgeIter>> edge_handles() { return handle_range(edges_begin(),edges_end()); }
  Range<HandleIter<ConstEdgeIter>> edge_handles() const { return handle_range(edges_begin(),edges_end()); }
  Range<HandleIter<HalfedgeIter>> halfedge_handles() { return handle_range(halfedges_begin(),halfedges_end()); }
  Range<HandleIter<ConstHalfedgeIter>> halfedge_handles() const { return handle_range(halfedges_begin(),halfedges_end()); }
  Range<HandleIter<FaceIter>> face_handles() { return handle_range(faces_begin(),faces_end()); }
  Range<HandleIter<ConstFaceIter>> face_handles() const { return handle_range(faces_begin(),faces_end()); }
};

// Silence OpenMesh output for the scope of this object
class OMSilencer {
  const bool log_enabled, err_enabled, out_enabled;
public:
  OMSilencer(bool log=true, bool err=true, bool out=true);
  ~OMSilencer();
};

Ref<TriMesh> merge(std::vector<Ref<const TriMesh>> meshes) OTHER_EXPORT;

}

// Reduce template bloat
namespace OpenMesh {
extern template class PropertyT<int>;
extern template class PropertyT<other::OVec<other::real,2>>;
extern template class PropertyT<other::OVec<other::real,3>>;
extern template class PropertyT<other::OVec<unsigned char,4>>;
extern template class PropertyT<VectorT<double,3>>;
extern template class PolyMeshT<AttribKernelT<FinalMeshItemsT<other::MeshTraits,true>,TriConnectivity>>;
}
