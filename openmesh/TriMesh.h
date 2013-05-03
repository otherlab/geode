#pragma once

#include <other/core/mesh/forward.h>

// OpenMesh includes
#include <cstddef>
// Since we're using hidden symbol visibility, dynamic_casts across shared library
// boundaries are problematic. Therefore, don't use them even in debug mode.
#define OM_FORCE_STATIC_CAST
#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#ifdef USE_OPENMESH
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Utils/vector_traits.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#endif

#ifdef _WIN32
#pragma warning( pop )
#endif

#include <other/core/utility/config.h>
#include <other/core/utility/Hasher.h>
#include <other/core/utility/tr1.h>
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

#include <boost/function.hpp>

#ifdef USE_OPENMESH

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
  static OVec<T,d> convert(PyObject* object) {
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

OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_CORE_EXPORT,2,VertexHandle)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_CORE_EXPORT,3,VertexHandle)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_CORE_EXPORT,2,FaceHandle)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_CORE_EXPORT,3,FaceHandle)

}

namespace OpenMesh {

// overloaded functions need to be in the same namespace as their arguments to be found by
// the compiler (or declared before the declaration of whatever uses them)

// python interface for handles
#ifdef OTHER_PYTHON
static inline PyObject* to_python(BaseHandle h) {
  return ::other::to_python(h.idx());
}
#endif

}

namespace other {

#ifdef OTHER_PYTHON

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

#endif

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
class OTHER_CORE_CLASS_EXPORT TriMesh: public Object, public OTriMesh {
public:
  OTHER_DECLARE_TYPE(OTHER_CORE_EXPORT)
  typedef Object Base;
  typedef real T;
  typedef Vector<T,3> TV;

  // add constructors and access functions to make the OpenMesh class more usable
protected:
  OTHER_CORE_EXPORT TriMesh();
  OTHER_CORE_EXPORT TriMesh(const TriMesh& m);
  OTHER_CORE_EXPORT TriMesh(RawArray<const Vector<int,3> > tris, RawArray<const Vector<real,3> > X);
  OTHER_CORE_EXPORT TriMesh(Tuple<Ref<TriangleMesh>,Array<Vector<real,3>>> const &);

  void cut_and_mirror(Plane<real> const &p, bool mirror, T epsilon, T area_hack);
public:
  OTHER_CORE_EXPORT virtual ~TriMesh();

public:
  // assign from another TriMesh
  OTHER_CORE_EXPORT TriMesh& operator=(OTriMesh const &o);

  // full copy
  OTHER_CORE_EXPORT Ref<TriMesh> copy() const;

  // load from file/stream
  OTHER_CORE_EXPORT void read(string const &filename);
  OTHER_CORE_EXPORT void read(std::istream &is, string const &ext);

  // save to file/stream
  OTHER_CORE_EXPORT void write(string const &filename) const;
  OTHER_CORE_EXPORT void write_with_normals(string const &filename) const;
  OTHER_CORE_EXPORT void write(std::ostream &os, string const &ext) const;

  // add a property
  template<class T,class Handle> typename HandleToProp<T,Handle>::type add_prop(const char* name) {
    typename HandleToProp<T,Handle>::type prop;
    add_property(prop,name);
    return prop;
  }

  // add a bunch of vertices
  OTHER_CORE_EXPORT void add_vertices(RawArray<const TV> X);

  // add a bunch of faces
  OTHER_CORE_EXPORT void add_faces(RawArray<const Vector<int,3> > faces);

  // add a mesh to this (vertices/faces)
  OTHER_CORE_EXPORT void add_mesh(TriMesh const &mesh);

  // bounding box
  OTHER_CORE_EXPORT Box<Vector<real,3> > bounding_box() const;
  OTHER_CORE_EXPORT Box<Vector<real,3> > bounding_box(vector<FaceHandle> const &faces) const;

  // area weighted centroid
  OTHER_CORE_EXPORT Vector<real,3> centroid() const;

  // centroid of a face (convenience function)
  OTHER_CORE_EXPORT Vector<real,3> centroid(FaceHandle fh) const;

  OTHER_CORE_EXPORT real mean_edge_length() const;

  // get a triangle area
  OTHER_CORE_EXPORT real area(FaceHandle fh) const;

  // get the face as a Triangle
  OTHER_CORE_EXPORT Triangle<Vector<real, 3> > triangle(FaceHandle fh) const;

  // get an edge as a Segment
  OTHER_CORE_EXPORT Segment<Vector<real, 3> > segment(EdgeHandle eh) const;
  OTHER_CORE_EXPORT Segment<Vector<real, 3> > segment(HalfedgeHandle heh) const;

  // compute the cotan weight for an edge
  OTHER_CORE_EXPORT real cotan_weight(EdgeHandle eh) const;

  // get the vertex handles incident to the given halfedge
  OTHER_CORE_EXPORT Vector<VertexHandle,2> vertex_handles(HalfedgeHandle heh) const;

  // get the vertex handles incident to the given edge
  OTHER_CORE_EXPORT Vector<VertexHandle,2> vertex_handles(EdgeHandle eh) const;

  // get the vertex handles incident to the given face
  OTHER_CORE_EXPORT Vector<VertexHandle,3> vertex_handles(FaceHandle fh) const;

  // get the edge handles incident to the given face
  OTHER_CORE_EXPORT Vector<EdgeHandle,3> edge_handles(FaceHandle fh) const;

  // get the halfedge handles that make up the given edge
  OTHER_CORE_EXPORT Vector<HalfedgeHandle,2> halfedge_handles(EdgeHandle eh) const;

  // get the halfedge handles incident to the given face
  OTHER_CORE_EXPORT Vector<HalfedgeHandle,3> halfedge_handles(FaceHandle fh) const;

  // get the face handles incident to the given edge (one of which is invalid if at boundary)
  OTHER_CORE_EXPORT Vector<FaceHandle, 2> face_handles(EdgeHandle eh) const;

  // get the edge-incident faces to a face (some of which may be invalid if at a boundary)
  OTHER_CORE_EXPORT Vector<FaceHandle,3> face_handles(FaceHandle fh) const;

  // get a valid face handle incident to an edge (invalid only if there is none at all,
  // in that case, the edge should have been deleted)
  OTHER_CORE_EXPORT FaceHandle valid_face_handle(EdgeHandle eh) const;

  // get all vertices in the one-ring
  OTHER_CORE_EXPORT vector<VertexHandle> vertex_one_ring(VertexHandle vh) const;

  // check whether the quad around an edge is convex (and the edge can be flipped safely)
  OTHER_CORE_EXPORT bool quad_convex(EdgeHandle eh) const;

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
  OTHER_CORE_EXPORT EdgeHandle edge_handle(VertexHandle vh1, VertexHandle vh2) const;

  // get the handle of the halfedge starting at vh1 going to vh2
  OTHER_CORE_EXPORT HalfedgeHandle halfedge_handle(VertexHandle vh1, VertexHandle vh2) const;

  // get the handle of the halfedge pointing at vh, in face fh
  OTHER_CORE_EXPORT HalfedgeHandle halfedge_handle(FaceHandle fh, VertexHandle vh) const;

  // get the incident faces to a vertex
  OTHER_CORE_EXPORT vector<FaceHandle> incident_faces(VertexHandle vh) const;

  /// return the common edge connecting two faces
  OTHER_CORE_EXPORT EdgeHandle common_edge(FaceHandle fh, FaceHandle fh2) const;

  // get a normal for an edge (average of incident face normals)
  OTHER_CORE_EXPORT Normal normal(EdgeHandle eh) const;

  // re-publish overloads from OTriMesh
  inline Normal normal(VertexHandle vh) const {
    return OTriMesh::normal(vh);
  }
  inline Normal normal(FaceHandle fh) const {
    return OTriMesh::normal(fh);
  }

  // get an interpolated normal at any point on the mesh
  OTHER_CORE_EXPORT Normal smooth_normal(FaceHandle fh, Vector<real,3> const &bary) const;

  // get rid of all infinite or nan vertices (they are simply deleted, along with incident faces)
  OTHER_CORE_EXPORT int remove_infinite_vertices();

  // dihedral angle between incident faces: positive for convex, negative for concave
  OTHER_CORE_EXPORT T dihedral_angle(EdgeHandle e) const;
  OTHER_CORE_EXPORT T dihedral_angle(HalfedgeHandle e) const;
  OTHER_CORE_EXPORT T cos_dihedral_angle(HalfedgeHandle e) const; // Quick version for when the sign doesn't matter

  // Fast version of calc_sector_angle for interior edges.  The angle is at v1 = to_vertex_handle(e)
  OTHER_CORE_EXPORT T cos_sector_angle(HalfedgeHandle e) const;

  // delete a set of faces
  OTHER_CORE_EXPORT void delete_faces(std::vector<FaceHandle> const &fh);

  // make a triangle fan
  OTHER_CORE_EXPORT vector<FaceHandle> triangle_fan(vector<VertexHandle> const &boundary, VertexHandle center, bool closed);

  // select a set of faces based on a predicate
  OTHER_CORE_EXPORT vector<FaceHandle> select_faces(boost::function<bool(FaceHandle)> pr) const;

  // extract a set of faces as a new mesh and store vertex correspondence: id2id[old] = new
  OTHER_CORE_EXPORT Ref<TriMesh> extract_faces(vector<FaceHandle> const &faces,
                             unordered_map<VertexHandle, VertexHandle, Hasher> &id2id) const;

  OTHER_CORE_EXPORT Ref<TriMesh> extract_faces(vector<FaceHandle> const &faces) const;

  // extract the inverse of a set of faces as a new mesh and store vertex correspondence
  OTHER_CORE_EXPORT Ref<TriMesh> inverse_extract_faces(vector<FaceHandle> const &faces,
                             unordered_map<VertexHandle, VertexHandle, Hasher> &id2id) const;

  OTHER_CORE_EXPORT Ref<TriMesh> inverse_extract_faces(vector<FaceHandle> const &faces) const;

  // compute the 2D silhouettes of the mesh as seem from the given rotation (with rotation*(0,0,1) as the normal)
  OTHER_CORE_EXPORT vector<vector<Vector<real,2>>> silhouette(const Rotation<TV>& rotation) const;

  // get the halfedges bounding the given set of faces (for all halfedges, face_handle(he) is in faces)
  OTHER_CORE_EXPORT unordered_set<HalfedgeHandle, Hasher> boundary_of(vector<FaceHandle> const &faces) const;

  // compute the approximate geodesic distance from one point to another, and store
  // all values computed on the way (can be used to re-trace the approximate shortest path)
  OTHER_CORE_EXPORT unordered_map<VertexHandle, double, Hasher> geodesic_distance(VertexHandle source,
                                                                VertexHandle sinks) const;

  // compute the approximate geodesic distance from one point to a set of points, and store
  // all values computed on the way (can be used to re-trace the approximate shortest paths)
  OTHER_CORE_EXPORT unordered_map<VertexHandle, double, Hasher> geodesic_distance(VertexHandle source,
                                                                vector<VertexHandle> const &sinks) const;

  // compute the approximate geodesic distance from one point to a set of points, and store
  // all values computed on the way (can be used to re-trace the approximate shortest paths)
  OTHER_CORE_EXPORT unordered_map<VertexHandle, double, Hasher> geodesic_distance(vector<VertexHandle> const &sources,
                                                                vector<VertexHandle> const &sinks) const;

  // compute and return the approximate shortest path from one point to another
  OTHER_CORE_EXPORT vector<VertexHandle> shortest_path(VertexHandle source, VertexHandle sink) const;

  // compute the closest face to a point by breadth-first search starting at the given vertex/face
  OTHER_CORE_EXPORT FaceHandle local_closest_face(Point const &p, FaceHandle start) const;
  OTHER_CORE_EXPORT FaceHandle local_closest_face(Point const &p, VertexHandle start) const;

  // compute edge-connected components around a boundary vertex
  OTHER_CORE_EXPORT vector<vector<FaceHandle>> surface_components(VertexHandle vh, unordered_set<EdgeHandle,Hasher> exclude_edges = (unordered_set<EdgeHandle,Hasher>())) const;

  // split a (boundary) vertex in as many vertices as there are edge-connected surface components
  // do not count exclude_edges as connections
  OTHER_CORE_EXPORT vector<VertexHandle> split_nonmanifold_vertex(VertexHandle vh, unordered_set<EdgeHandle,Hasher> exclude_edges = (unordered_set<EdgeHandle,Hasher>()));

  // split an edge in two if the incident faces are only connected through this edge
  // returns the newly created edges (including the old one). Both end points
  // have to be boundary vertices for this to happen.
  OTHER_CORE_EXPORT vector<EdgeHandle> separate_edge(EdgeHandle eh);

  // split the mesh along a string of edges. If the edges form loops, this
  // results in two holes per loop. All non-loop connected components create
  // a single hole. Returns all vertices that were split, and all vertices they
  // were split into.
  OTHER_CORE_EXPORT vector<VertexHandle> separate_edges(vector<EdgeHandle> ehs);

  // cut the mesh with a plane (negative side will be removed)
  OTHER_CORE_EXPORT void cut(Plane<real> const &p, double epsilon = 1e-4, double area_hack = 0);

  // mirror the mesh at a plane (positive side will be mirrored, negative replaced)
  OTHER_CORE_EXPORT void mirror(Plane<real> const &p, double epsilon = 1e-4);

  // check if mesh has a boundary (faster than !boundary_loops().empty())
  // this function will report a boundary for isolated vertices!
  OTHER_CORE_EXPORT bool has_boundary() const;

  // find boundary loops
  OTHER_CORE_EXPORT vector<vector<HalfedgeHandle> > boundary_loops() const;

  // find the boundary loop starting at seed (empty if seed is not on the boundary)
  OTHER_CORE_EXPORT vector<HalfedgeHandle> boundary_loop(HalfedgeHandle const &seed) const;

  // fill the hole enclosed by the given halfedges, retain the new faces only if the surface area is smaller than max_area
  OTHER_CORE_EXPORT vector<FaceHandle> fill_hole(vector<HalfedgeHandle> const &loop, double max_area = inf);

  // fill all holes with maximum area given, returns the number of holes filled
  OTHER_CORE_EXPORT int fill_holes(double max_area = inf);

  OTHER_CORE_EXPORT void add_box(TV min, TV max);
  OTHER_CORE_EXPORT void add_sphere(TV c, real r, int divisions = 30);
  OTHER_CORE_EXPORT void add_cylinder(TV p1, TV p2, real r1, real r2, int divisions = 30, bool caps = true);

  OTHER_CORE_EXPORT void scale(real scale, const TV& center=TV());
  OTHER_CORE_EXPORT void scale(TV scale, const TV& center=TV());
  OTHER_CORE_EXPORT void translate(const TV& c);
  OTHER_CORE_EXPORT void rotate(const Rotation<TV>& R, const TV& center=TV());
  OTHER_CORE_EXPORT void transform(const Frame<TV>& F);

  // flip all faces inside out
  OTHER_CORE_EXPORT void invert();

  // flip all faces in a connected component inside out (faces in component
  // must not have neighbors not in component)
  OTHER_CORE_EXPORT void invert_component(vector<FaceHandle> component);

  OTHER_CORE_EXPORT real volume() const;
  OTHER_CORE_EXPORT real volume(RawArray<const FaceHandle> faces) const;
  OTHER_CORE_EXPORT real area() const;
  OTHER_CORE_EXPORT real area(RawArray<const FaceHandle> faces) const;

  // Warning: these construct new arrays or copy memory
  OTHER_CORE_EXPORT Array<Vector<int,3> > elements() const;
  OTHER_CORE_EXPORT Array<Vector<int,2> > segments() const;
  OTHER_CORE_EXPORT Array<Vector<real,3> > X_python() const;
  OTHER_CORE_EXPORT void set_X_python(RawArray<const Vector<real,3>> new_X);
  OTHER_CORE_EXPORT void set_vertex_normals(RawArray<const Vector<real,3>> normals);
  OTHER_CORE_EXPORT void set_vertex_colors(RawArray<const Vector<real,3>> colors);

  // Warning: reference goes invalid if the mesh is changed
  OTHER_CORE_EXPORT RawArray<Vector<real,3> > X();
  OTHER_CORE_EXPORT RawArray<const Vector<real,3> > X() const;

  OTHER_CORE_EXPORT Ref<SimplexTree<Vector<real,3>,2>> face_tree() const;
  OTHER_CORE_EXPORT Ref<SimplexTree<Vector<real,3>,1>> edge_tree() const;
  OTHER_CORE_EXPORT Ref<ParticleTree<Vector<real,3>>> point_tree() const;

  // Warning: reference goes invalid if the mesh is changed
  template<class PropHandle> RawField<typename PropHandle::Value,typename prop_handle_type<PropHandle>::type> prop(PropHandle p) {
    return RawField<typename PropHandle::Value,typename prop_handle_type<PropHandle>::type>(property(p).data_vector());
  }

  // Warning: reference goes invalid if the mesh is changed
  template<class PropHandle> RawField<const typename PropHandle::Value,typename prop_handle_type<PropHandle>::type> prop(PropHandle p) const {
    return RawField<const typename PropHandle::Value,typename prop_handle_type<PropHandle>::type>(const_cast_(property(p)).data_vector());
  }

  // Component analysis
  OTHER_CORE_EXPORT Tuple<int,Array<int> > component_vertex_map() const;
  OTHER_CORE_EXPORT Array<int> component_face_map() const;

  // Split a mesh into connected components
  OTHER_CORE_EXPORT vector<Ref<TriMesh> > component_meshes() const;
  OTHER_CORE_EXPORT vector<Ref<TriMesh> > nested_components() const;

  // Convenience functions for use in range-based for loops
  inline Range<HandleIter<VertexIter>> vertex_handles() { return handle_range(vertices_sbegin(),vertices_end()); }
  inline Range<HandleIter<ConstVertexIter>> vertex_handles() const { return handle_range(vertices_sbegin(),vertices_end()); }
  inline Range<HandleIter<EdgeIter>> edge_handles() { return handle_range(edges_sbegin(),edges_end()); }
  inline Range<HandleIter<ConstEdgeIter>> edge_handles() const { return handle_range(edges_sbegin(),edges_end()); }
  inline Range<HandleIter<HalfedgeIter>> halfedge_handles() { return handle_range(halfedges_sbegin(),halfedges_end()); }
  inline Range<HandleIter<ConstHalfedgeIter>> halfedge_handles() const { return handle_range(halfedges_sbegin(),halfedges_end()); }
  inline Range<HandleIter<FaceIter>> face_handles() { return handle_range(faces_sbegin(),faces_end()); }
  inline Range<HandleIter<ConstFaceIter>> face_handles() const { return handle_range(faces_sbegin(),faces_end()); }
};

// Silence OpenMesh output for the scope of this object
class OMSilencer {
  const bool log_enabled, err_enabled, out_enabled;
public:
  OTHER_CORE_EXPORT OMSilencer(bool log=true, bool err=true, bool out=true);
  OTHER_CORE_EXPORT ~OMSilencer();
};

OTHER_CORE_EXPORT Ref<TriMesh> merge(vector<Ref<const TriMesh>> meshes);

}

#endif // USE_OPENMESH
