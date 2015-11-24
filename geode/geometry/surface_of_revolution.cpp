#include <geode/geometry/surface_of_revolution.h>
#include <geode/math/constants.h>
namespace geode {

namespace {
struct CylinderTopology {
  int nz; // Must be > 0
  int na; // Must be > 1
  bool top_closed;
  bool bottom_closed;

  bool size_valid() const { return (nz > 0) && (na > 1); }
  bool valid(const int z, const int a) const { assert(size_valid()); return (0 <= z && z < nz) && (0 <= a && a < na); }

  using VertexId = int;

  VertexId side_vertex(const int z, const int a) const {
    assert(valid(z,a));
    return VertexId(bottom_closed + z + a*nz);
  }

  int num_side_vertices() const { return nz*na; }
  int num_vertices() const { return num_side_vertices() + top_closed + bottom_closed; }
  int num_faces() const { assert(size_valid()); return na*((nz-1)*2 + top_closed + bottom_closed); }

  VertexId bottom_vertex() const { assert(bottom_closed); return VertexId(0); }
  VertexId top_vertex() const { assert(top_closed); return VertexId(num_side_vertices()+bottom_closed); }

  int next_a(const int a) const { return (a+1 == na) ? 0 : a+1; }

  Array<Vector<VertexId,3>> faces() const {
    Array<Vector<VertexId,3>> result;
    result.preallocate(num_faces());
    if(bottom_closed) {
      const VertexId v0 = bottom_vertex();
      for(int curr_a = 0; curr_a < na; ++curr_a) {
        const int next_a = this->next_a(curr_a);
        const VertexId v1 = side_vertex(0,next_a);
        const VertexId v2 = side_vertex(0,curr_a);
        result.append(vec(v0,v1,v2));
      }
    }
    for(int z = 0; z < nz-1; ++z) {
      for(int curr_a = 0; curr_a < na; ++curr_a) {
        const int next_a = this->next_a(curr_a);
        const VertexId ll = side_vertex(z+0,curr_a);
        const VertexId lr = side_vertex(z+0,next_a);
        const VertexId ur = side_vertex(z+1,next_a);
        const VertexId ul = side_vertex(z+1,curr_a);
        result.append(vec(ll,lr,ul));
        result.append(vec(lr,ur,ul));
      }
    }
    if(top_closed) {
      const VertexId v2 = top_vertex();
      for(int curr_a = 0; curr_a < na; ++curr_a) {
        const int next_a = this->next_a(curr_a);
        const VertexId v0 = side_vertex(nz-1,curr_a);
        const VertexId v1 = side_vertex(nz-1,next_a);
        result.append(vec(v0,v1,v2));
      }
    }
    return result;
  }
};
} // anonymous namespace

template<class T> static T r(const Vector<T,2>& rz) { return rz.x; }
template<class T> static T z(const Vector<T,2>& rz) { return rz.y; }

// Templated version used to provide overloads for float and double
template<class T> static Tuple<Array<Vec3i>,Array<Vector<T,3>>> surface_of_revolution_helper(RawArray<const Vector<T,2>> profile_rz, const int sides) {
  using TV2 = Vector<T,2>;
  using TV3 = Vector<T,3>;
  const auto bottom_z = z(profile_rz.front());
  const auto top_z = z(profile_rz.back());

  // Trim endpoints of profile if it already tapers to a point
  int profile_revolve_start = 0;
  int profile_revolve_end = profile_rz.size();
  constexpr T tolerance = 0;
  if(profile_revolve_start < profile_rz.size() 
        && r(profile_rz[profile_revolve_start]) <= tolerance) {
    ++profile_revolve_start;
  }
  if(profile_revolve_end > 0
        && r(profile_rz[profile_revolve_end-1]) <= tolerance) {
    --profile_revolve_end;
  }
  const RawArray<const TV2> body_profile = profile_rz.slice(profile_revolve_start, profile_revolve_end);

  const CylinderTopology topology = {body_profile.size(), sides, true, true};

  Array<TV3> X;
  X.preallocate(topology.num_vertices());
  const T segment_angle = tau/sides;
  X.append(TV3(0, 0, bottom_z));
  for(const int i : range(sides)) {
    const TV2 r_axis = polar(segment_angle*i);
    for(const auto& p : body_profile)
      X.append_assuming_enough_space(TV3(r(p)*r_axis,z(p)));
  }
  X.append(TV3(0,0,top_z));
  assert(X.size() == topology.num_vertices());
  return tuple(topology.faces(), X);
}

Tuple<Array<Vector<int,3>>,Array<Vector<float,3>>> surface_of_revolution(RawArray<const Vector<float,2>> profile_rz, const int sides) {
  return surface_of_revolution_helper(profile_rz, sides);
}

Tuple<Array<Vector<int,3>>,Array<Vector<double,3>>> surface_of_revolution(RawArray<const Vector<double,2>> profile_rz, const int sides) {
  return surface_of_revolution_helper(profile_rz, sides);
}

} // geode namespace
