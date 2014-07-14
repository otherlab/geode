// Mesh offsets

#include <geode/geometry/offset_mesh.h>
#include <geode/exact/mesh_csg.h>
#include <geode/python/wrap.h>
namespace geode {

typedef real T;
typedef Vector<real,3> TV;
using std::make_heap;
using std::push_heap;
using std::pop_heap;
using std::cout;
using std::endl;

typedef Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>> Result;

/*
 * We first describe how to perform simple offsets of meshes viewed as shells, including meshes
 * with boundary.  If a simple offset as described below would be unsafe, we split the mesh along
 * edges until everything is good; this is guaranteed to terminate since a single triangle always
 * has a simple offset.  Volume offsets are then computed as the union of the original mesh with
 * the possibly split shell offset.
 *
 * Given a shell mesh, a simple shell offset consists of
 *
 * 1. One vertex offset above and below each original vertex, offset along the vertex pseudonormal.
 * 2. For boundary vertices, one or two additional vertices offset outwards along boundary normals.
 *    If the boundary vertex has incident outward edge normals n0,n1, the one vertex case uses
 *    offset normal nv = normalized(n0+n1), and the two vertex case uses n0+nv, n1+nv.
 * 3. Above each original face, one offset face above and one below, using the vertical offset vertices.
 * 4. Above each original boundary edge, two quads connecting the upwards pair of vertices to the
 *    outwards pair to the downwards pair.
 * 5. Above each original boundary vertex with two boundary offset vertices, two triangles connecting
 *    one vertex above to 2 vertices outwards to one vertex below.
 *
 * There are two requirements for a simple offset to be safe.  First, the simplices above a given
 * feature must be appropriately far from the mesh.  Since all offset vertices are computed as
 * x + offset*n where |n| = 1, they are never too far but might be too close if the normals vary
 * quickly around the feature.  For vertical offset vertices, it is sufficient for the offset normal
 * to have good dot products with all incident face normals.  In the code below, good means bounded
 * below by alpha = 1/4.  The same test occurs above boundary vertices, and above boundary edges we
 * need a related check to verify that the above-mid-below swing around the edge doesn't go too close
 * to the vertex (search for gamma below).
 *
 * Second, we must guarantee that the shell offset has everywhere nonnegative depth (winding number).
 * We use the following trivial lemma to guarantee safety:
 *
 * Lemma: Let M, M' be two closed meshes, and T a set of positively oriented tetrahedra with
 * boundary(T) = outer_boundary(M) + inner_boundary(M').  Then the winding number of M' is higher
 * than M everywhere.  In particular, if M has nonnegative winding number, so does M'.
 *
 * Thus, whenever we place a piece of offset surface M' above part of the original surface M, we
 * must show that it is possible to tetrahedralize the intervening space with positive tetrahedra.
 * We do not actually need to make these tetrahedra in the code, only show that they exist.  There
 * are three kinds of tetrahedra in consideration: those above faces, boundary edges, and boundary vertices.
 * Above faces and boundary edges, we have one and two triangular prisms, respectively, which can be
 * tetrahedralized in 6 different ways.  As we need our emphemeral tetrahedra to be consistent across
 * the mesh, the prism_safe routine below requires all 6 tetrahedralizations to be safe in order to sign
 * off on an offset.  Above a boundary vertex, we have 0 or 2 additional tetrahedra in a fixed pattern,
 * which are easier to check for validity.
 */

// Can a vertex be split?
static bool can_split(const TriangleTopology& mesh, const VertexId v) {
  const auto e = mesh.halfedge(v);
  return e != mesh.left(mesh.left(e));
}

template<class F> static inline HalfedgeId
min_interior_edge(const TriangleTopology& mesh, RawField<const TV,VertexId> X, const VertexId v, F&& f) {
  const auto end = mesh.halfedge(v);
  auto e = end;
  if (mesh.is_boundary(e)) {
    e = mesh.left(mesh.left(e));
    GEODE_ASSERT(e != end,"Need at least one interior edge");
  }
  HalfedgeId min_e;
  T min_f = inf;
  do {
    const auto fe = f(e);
    if (min_f > fe) {
      min_f = fe;
      min_e = e;
    }
    e = mesh.left(e);
  } while (e != end);
  return min_e;
}

// Find the edge with the worst dihedral angle
static HalfedgeId worst_dihedral(const TriangleTopology& mesh, RawField<const TV,VertexId> X, const VertexId v) {
  return min_interior_edge(mesh,X,v,[&](const HalfedgeId e) {
    return mesh.cos_dihedral(X,e);
  });
}

// Find the interior edge furthest from a given direction
static HalfedgeId furthest_edge(MutableTriangleTopology& mesh, RawField<const TV,VertexId> X,
                                const VertexId v, const TV direction) {
  const auto xv = X[v];
  return min_interior_edge(mesh,X,v,[&](const HalfedgeId e) {
    const auto ex = X[mesh.dst(e)]-xv;
    return dot(direction,ex)/magnitude(ex);
  });
}

// Triangulate a quad given as four counterclockwise vertices, picking the minimum dihedral diagonal.
static Vector<Vector<int,3>,2> triangulate_quad(const int v0, const TV x0,
                                                const int v1, const TV x1,
                                                const int v2, const TV x2,
                                                const int v3, const TV x3) {
  const TV n0 = cross(x1-x0,x3-x0),
           n1 = cross(x1-x0,x2-x0),
           n2 = cross(x2-x1,x3-x1),
           n3 = cross(x2-x0,x3-x0);
  const T cos_dihedral_13 = dot(n0,n2)/sqrt(sqr_magnitude(n0)*sqr_magnitude(n2)),
          cos_dihedral_02 = dot(n1,n3)/sqrt(sqr_magnitude(n1)*sqr_magnitude(n3));
  return cos_dihedral_13 > cos_dihedral_02 ? vec(vec(v0,v1,v3),vec(v1,v2,v3))
                                           : vec(vec(v0,v1,v2),vec(v0,v2,v3));
}

// Check that *all* ways of tetrahedralizing a prism are safe.  It is not enough that one way works,
// since we need compatibility with neighboring prisms, and do not know how those prisms will be
// tetrahedralized.  The prism is triangle x0,x1,x2 below triangle x3,x4,x5, both triangles pointing up.
static bool prism_safe(const TV x0, const TV x1, const TV x2, const TV x3, const TV x4, const TV x5) {
  // Code generated by prism-helper
  #define S(a,b,c,d) (det(x##b-x##a,x##c-x##a,x##d-x##a) > 0)
  return S(0,1,2,3) && S(0,1,2,4) && S(0,1,2,5) && S(0,1,5,3) && S(0,1,5,4) && S(0,2,3,4)
      && S(0,2,5,4) && S(0,3,4,5) && S(1,2,3,4) && S(1,2,3,5) && S(1,3,4,5) && S(2,3,4,5);
  #undef S
}

// Decompose a shell into pieces, each of which can be offset simply without introducing
// foldovers, then union the simple offsets of the pieces.
static Tuple<Array<Vector<int,3>>,Array<TV>> rough_offset_shell_helper(const TriangleTopology& mesh_in,
                                                                       RawField<const TV,VertexId> X_in,
                                                                       const T offset) {
  // Negative shell offsets aren't very interesting
  if (offset <= 0)
    return Tuple<Array<Vector<int,3>>,Array<TV>>();
  GEODE_ASSERT(mesh_in.is_manifold_with_boundary());

  const T alpha = 1./4, // Dot products have to be at least this good
          gamma = 3./4; // Dot products near the boundary must be at least this small

  // Copy the mesh so that we can split vertices apart
  const auto mesh = mesh_in.mutate();
  const auto X_id = mesh->add_field(X_in.copy());
  // Each vertex has one offset vertex above and below, with normals n and -n.  TV() means unprocessed.
  const auto vn_id = mesh->add_field(Field<TV,VertexId>(mesh->n_vertices()));
  const auto& X = mesh->field(X_id);
  const auto& vertex_normals = mesh->field(vn_id);

  // We'll be using the fact that new boundary edges appear with more negative indices,
  // and later that vertex ids are contiguous.
  mesh->collect_garbage();

  // Outwards pointing boundary normals
  const auto boundary_normal = [&](const HalfedgeId e) {
    assert(mesh->is_boundary(e));
    return normalized(cross(X[mesh->src(e)]-X[mesh->dst(e)],
                            mesh->normal(X,mesh->face(mesh->reverse(e)))));
  };

  // Each boundary has one or two additional normals pointing outwards: nr,nl from right to left.
  Array<Vector<TV,2>> boundary_normals(mesh->n_boundary_edges(),uninit);

  // Our control flow is a bit complicated: we need to process vertices until all safety conditions
  // hold above vertices, boundary edges, and faces.  Checking conditions may cause vertices to be
  // split, in which case more processing must take place.  Thus, after processing a vertex, we
  // check all incident faces and boundary edges that are complete (have all vertices processed).
  Array<VertexId> work;
  work.preallocate(mesh->n_vertices());
  for (const auto v : mesh->vertices())
    work.append(v);
  while (work.size()) {
    const auto v = work.pop();
    auto split_v = v;

    // Check that our vertex normal is sufficiently well aligned with all incident faces
    const auto nv = mesh->normal(X,v);
    for (const auto e : mesh->outgoing(v)) {
      const auto f = mesh->face(e);
      if (f.valid() && dot(nv,mesh->normal(X,f)) < alpha)
        goto split;
    }
    vertex_normals[v] = nv;

    // If we're a boundary vertex, check safety above vertex and boundary edges
    if (mesh->is_boundary(v)) {
      const auto er = mesh->halfedge(v), // v to right
                 el = mesh->left(er);    // v to left

      // Compute normals
      const auto ner = boundary_normal(er),
                 nel = boundary_normal(mesh->reverse(el)),
                 ne = normalized(ner+nel);

      // Is our vertical orthogonal orthogonal enough to the boundary normals?
      if (   abs(dot(nv,ner)) > gamma
          || abs(dot(nv,nel)) > gamma
          || abs(dot(nv,ne )) > gamma)
        goto split;

      // Prepare to store boundary normals
      boundary_normals.resize(max(boundary_normals.size(),-er.id),uninit);
      auto& B = boundary_normals[-1-er.id];

      // Is it safe to use a single mid vertex?
      if (   dot(ne,ner) > alpha
          && dot(ne,nel) > alpha)
        B = vec(ne,ne);
      else {
        // Is it safe to use two mid vertices?
        const auto nr = normalized(ne+ner),
                   nl = normalized(ne+nel);
        if (det(nv,nr,nl) <= 0)
          goto split;
        B = vec(nr,nl);
      }

      // We're good above the vertex, so check safety above both boundary edges if they're complete.
      const auto vr = mesh->dst(er),
                 vl = mesh->dst(el);
      const auto xv = X[v],
                 nr = vertex_normals[vr],
                 nl = vertex_normals[vl];
      #define CHECK_BOUNDARY(ne,v0,n0,B0,v1,n1,B1) { \
        const auto x0 = X[v0], \
                   x1 = X[v1]; \
        if (   !prism_safe(x0,x0+offset*B0.x,x0+offset*n0, \
                           x1,x1+offset*B1.y,x1+offset*n1) \
            || !prism_safe(x0,x0-offset*n0,x0+offset*B0.x, \
                           x1,x1-offset*n1,x1+offset*B1.y)) { \
          /* The prisms don't work, so we need to split one of the vertices.  Split the one whose */ \
          /* mid normal is furthest from the edge normal.  Heuristic is fine here. */ \
          const auto cs0 = can_split(mesh,v0), \
                     cs1 = can_split(mesh,v1); \
          GEODE_ASSERT(cs0 || cs1); \
          split_v = !cs1 || (cs0 && dot(ne,B0.x) < dot(ne,B1.y)) ? v0 : v1; \
          goto split; \
        } \
      }
      if (nr != TV()) {
        const auto& Br = boundary_normals[-1-mesh->next(er).id];
        CHECK_BOUNDARY(ner,v,nv,B,vr,nr,Br)
      }
      if (nl != TV()) {
        const auto& Bl = boundary_normals[-1-mesh->prev(er).id];
        CHECK_BOUNDARY(nel,vl,nl,Bl,v,nv,B)
      }
      #undef CHECK_BOUNDARY
    }

    // We're good above the vertex, and near the boundary if we're a boundary vertex.  Check if all
    // surrounding faces are clean.
    for (const auto e : mesh->outgoing(v)) {
      const auto f = mesh->face(e);
      if (f.valid()) {
        const auto vs = mesh->vertices(f);
        const auto n0 = vertex_normals[vs.x],
                   n1 = vertex_normals[vs.y],
                   n2 = vertex_normals[vs.z];
        if (n0!=TV() && n1!=TV() && n2!=TV()) {
          const auto x0 = X[vs.x],
                     x1 = X[vs.y],
                     x2 = X[vs.z];
          if (   !prism_safe(x0,x1,x2,x0+offset*n0,x1+offset*n1,x2+offset*n2)
              || !prism_safe(         x0-offset*n0,x1-offset*n1,x2-offset*n2,x0,x1,x2)) {
            // Bad face.  Split the vertex whose normal is furthest from the face normal
            const auto nf = mesh->normal(X,f);
            const T d0 = dot(nf,n0),
                    d1 = dot(nf,n1),
                    d2 = dot(nf,n2);
            const auto cs0 = can_split(mesh,vs.x),
                       cs1 = can_split(mesh,vs.y),
                       cs2 = can_split(mesh,vs.z);
            GEODE_ASSERT(cs0 || cs1 || cs2);
            split_v = (!cs1 && !cs2) || (cs0 && d0<d1 && d0<d2) ? vs.x
                    :          !cs2  || (cs1 && d1<d2         ) ? vs.y : vs.z;
            goto split;
          }
        }
      }
    }

    // If we get here, we're all good!
    continue;

    // If we jump here from above, we've failed a safety condition and need to split vertex split_v.
    split:
    {
      const auto v = split_v;
      const auto eb = mesh->halfedge(v);
      Vector<VertexId,3> vs;
      if (mesh->is_boundary(eb)) {
        const auto dir = boundary_normal(eb)+boundary_normal(mesh->prev(eb));
        const auto e = furthest_edge(mesh,X,v,dir);
        mesh->split_along_edge(e);
        vs = Vector<VertexId,3>(mesh->vertices(e));
      } else {
        const auto e0 = worst_dihedral(mesh,X,v);
        const auto v0 = mesh->dst(e0);
        mesh->split_along_edge(e0);
        const auto e1 = furthest_edge(mesh,X,v,X[v0]-X[v]);
        mesh->split_along_edge(e1);
        vs = vec(v,v0,mesh->dst(e1));
      }
      // Split any newly nonmanifold vertices, mark them uninitialized and add them back to the work set
      for (const auto v : vs)
        if (v.valid())
          for (const auto u : mesh->split_nonmanifold_vertex(v)) {
            work.append(u);
            vertex_normals[u] = TV();
          }
      GEODE_ASSERT(mesh->is_manifold_with_boundary());
    }
  }

  // Create vertices.  Vertices above and below original vertex v are 2*v and 2*v+1.
  Array<TV> new_X;
  new_X.preallocate(2*mesh->n_vertices()+mesh->n_boundary_edges());
  for (const auto v : mesh->vertices()) {
    const auto x = X[v],
               dx = offset*vertex_normals[v];
    new_X.extend(vec(x+dx,x-dx));
  }
  Array<Vector<int,2>> boundary_v(mesh->n_boundary_edges(),uninit);
  for (const auto e : mesh->boundary_edges()) {
    const auto x = X[mesh->src(e)];
    const auto& B = boundary_normals[-1-e.id];
    const int v0 =                 new_X.append(x+offset*B.x),
              v1 = B.x==B.y ? v0 : new_X.append(x+offset*B.y);
    boundary_v[-1-e.id] = vec(v0,v1);
  }

  // Triangulate everything
  Array<Vector<int,3>> new_soup;
  for (const auto f : mesh->faces()) {
    const auto v = mesh->vertices(f);
    new_soup.extend(vec(vec(2*v.x.id  ,2*v.y.id  ,2*v.z.id  ),
                        vec(2*v.x.id+1,2*v.z.id+1,2*v.y.id+1)));
  }
  for (const auto e : mesh->boundary_edges()) {
    const auto v0 = mesh->src(e),
               v1 = mesh->dst(e);
    const auto vv0 = boundary_v[-1-e.id],
               vv1 = boundary_v[-1-mesh->next(e).id];
    // Triangulate above vertex v0
    if (vv0.x != vv0.y)
      new_soup.extend(vec(vec(2*v0.id  ,vv0.x,vv0.y),
                          vec(2*v0.id+1,vv0.y,vv0.x)));
    // Triangulate above edge e
    #define ADD_QUAD(a,b,c,d) new_soup.extend(triangulate_quad(a,new_X[a],b,new_X[b],c,new_X[c],d,new_X[d]));
    ADD_QUAD(2*v0.id  ,2*v1.id,  vv1.y,vv0.x)
    ADD_QUAD(2*v1.id+1,2*v0.id+1,vv0.x,vv1.y)
    #undef ADD_QUAD
  }

  // All done!
  return tuple(new_soup,new_X);
}

// Offset but don't run CSG
Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>>
rough_preoffset_shell(const TriangleTopology& mesh, RawField<const TV,VertexId> X, const T offset) {
  const auto S = rough_offset_shell_helper(mesh,X,offset);
  return tuple(new_<const TriangleTopology>(S.x),Field<const TV,VertexId>(S.y));
}

Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>>
rough_offset_shell(const TriangleTopology& mesh, RawField<const TV,VertexId> X, const T offset) {
  const auto H = rough_offset_shell_helper(mesh,X,offset);
  const auto S = split_soup(new_<TriangleSoup>(H.x),H.y,0);
  return tuple(new_<const TriangleTopology>(S.x),Field<const TV,VertexId>(S.y));
}

Tuple<Ref<const TriangleTopology>,Field<const TV,VertexId>>
rough_offset_mesh(const TriangleTopology& mesh, RawField<const TV,VertexId> X, const T offset) {
  const auto H = rough_offset_shell_helper(mesh,X,offset);
  const auto S = split_soup(new_<TriangleSoup>(concatenate(mesh.elements(),X.size()+H.x)),
                            concatenate(X.flat,H.y),0);
  return tuple(new_<const TriangleTopology>(S.x),Field<const TV,VertexId>(S.y));
}

}
using namespace geode;

void wrap_offset_mesh() {
  GEODE_FUNCTION(rough_preoffset_shell) // For testing purposes
  GEODE_FUNCTION(rough_offset_shell)
  GEODE_FUNCTION(rough_offset_mesh)
}
