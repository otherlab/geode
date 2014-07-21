#!/usr/bin/env python

from __future__ import division
from geode import *
from geode.geometry.platonic import *

def test_basic():
  a = TriangleTopology([(0,1,2)])
  b = TriangleTopology(TriangleSoup([(0,1,2)]))
  assert all(a.elements()==[(0,1,2)])
  assert all(a.elements()==b.elements())
  try:
    TriangleTopology([(-1,1,2)])
    assert False
  except ValueError:
    pass
  c = TriangleTopology()
  assert all(c.elements().shape==(0,3))

def test_collect_boundary_garbage():
  random.seed(13131)
  soup = torus_topology(4,5)
  tris = torus_topology(4,5).elements.copy()
  random.shuffle(tris)
  mesh = MutableTriangleTopology()
  assert mesh.is_garbage_collected()
  mesh.add_vertices(soup.nodes())
  mesh.add_faces(tris)
  assert not mesh.is_garbage_collected()
  mesh.collect_boundary_garbage()
  mesh.assert_consistent(True)
  assert mesh.is_garbage_collected()

def construction_test(Mesh,random_edge_flips=random_edge_flips,random_face_splits=random_face_splits,mesh_destruction_test=mesh_destruction_test):
  random.seed(813177)
  nanosphere = TriangleSoup([(0,1,2),(0,2,1)])
  print
  print Mesh.__name__
  for soup in nanosphere,icosahedron_mesh()[0],torus_topology(4,5),double_torus_mesh(),cylinder_topology(6,5):
    # Learn about triangle soup
    n_vertices = soup.nodes()
    n_edges = len(soup.segment_soup().elements)
    n_faces = len(soup.elements)
    chi = n_vertices-n_edges+n_faces
    open = len(soup.boundary_mesh().elements)!=0
    def sort_tris(tris):
      return tris[lexsort(tris.T[::-1])]
    print 'chi = v%d - e%d + f%d = %d'%(n_vertices,n_edges,n_faces,chi)

    def check_counts(mesh):
      assert mesh.n_vertices==n_vertices
      assert mesh.n_edges==n_edges
      assert mesh.n_faces==n_faces
      assert mesh.chi==chi

    def check_add_faces(mesh,tris):
      mesh.assert_consistent(True)
      try:
        mesh.add_faces(tris)
        mesh.assert_consistent(True)
      except:
        mesh.dump_internals()
        raise

      if open:
        assert mesh.is_manifold_with_boundary()
        loops = mesh.boundary_loops()
        assert len(loops)==2 and all(loops.sizes()==5)
      else:
        assert mesh.is_manifold()
        assert not mesh.has_boundary()
      check_counts(mesh)
      assert all(sort_tris(tris)==sort_tris(mesh.elements()))

    def check_add_face(mesh,tris):
      mesh.assert_consistent(True)
      for t in tris:
        try:
          assert mesh.has_boundary() or mesh.has_isolated_vertices()
          mesh.add_face(t)
          mesh.assert_consistent(True)
        except:
          mesh.dump_internals()
          raise

      if open:
        assert mesh.is_manifold_with_boundary()
        loops = mesh.boundary_loops()
        assert len(loops)==2 and all(loops.sizes()==5)
      else:
        assert mesh.is_manifold()
        assert not mesh.has_boundary()
      check_counts(mesh)
      assert all(sort_tris(tris)==sort_tris(mesh.elements()))

    # Turn the soup into a halfedge mesh
    base = Mesh()
    base.add_vertices(soup.nodes())
    # check batch insertion and throw result away
    check_add_faces(base.copy(),soup.elements)
    # check face insertion one by one
    check_add_face(base,soup.elements)

    for key in xrange(5):
      # Mangle the clean mesh using a bunch of edge flips
      mesh = base.copy()
      flips = random_edge_flips(mesh,1000,key)
      print 'flips = %d'%flips
      if soup is not nanosphere:
        assert flips>484
      mesh.assert_consistent(True)
      tris = mesh.elements()
      random.shuffle(tris)
      # Reconstruct the mangled mesh one triangle at a time
      partial = Mesh()
      partial.add_vertices(soup.nodes())
      check_add_faces(partial,tris)
      # Check that face (and maybe edge) splits are safe
      random_face_splits(mesh,20,key+10)
      mesh.assert_consistent(True)
      # Tear the mesh apart in random order
      mesh_destruction_test(mesh,key+20)
      assert mesh.n_vertices==mesh.n_edges==mesh.n_faces==0

def test_halfedge_construction():
  construction_test(HalfedgeMesh)

def test_corner_construction():
  construction_test(MutableTriangleTopology,corner_random_edge_flips,corner_random_face_splits,corner_mesh_destruction_test)

def test_flip():
  soup = icosahedron_mesh()[0]
  mesh = MutableTriangleTopology()
  mesh.add_vertices(soup.nodes())
  mesh.add_faces(soup.elements)
  mesh.assert_consistent(True)

  # make sure we can flip a closed mesh
  mesh.flip()
  mesh.assert_consistent(True)

  # make sure we can flip a mesh with boundaries
  mesh.erase_face(0, False)
  mesh.flip()
  mesh.assert_consistent(True)

def test_nonmanifold_vertices():
  soup = icosahedron_mesh()[0]
  mesh = MutableTriangleTopology()
  mesh.add_vertices(soup.nodes())
  mesh.add_faces(soup.elements)

  mesh.assert_consistent(True)

  # mess with the neighborhood of vertex 0, such much that it needs to be split
  for i,he in enumerate(mesh.outgoing(0)):
    if i == 1 or i == 3:
      mesh.split_along_edge(he)
      mesh.assert_consistent(False)

  r = mesh.split_nonmanifold_vertices()
  mesh.assert_consistent(True)

  assert len(r.sizes()) == 1
  assert len(r[0]) == 2
  assert r[0][0] == 0


def test_fields():
  random.seed(813177)
  nanosphere = TriangleSoup([(0,1,2),(0,2,1)])
  for soup in nanosphere,icosahedron_mesh()[0],torus_topology(4,5),double_torus_mesh(),cylinder_topology(6,5):
    # Make a MutableTriangleTopology
    mesh = MutableTriangleTopology()
    mesh.add_vertices(soup.nodes())
    mesh.add_faces(soup.elements)

    # Make a field
    Vi = mesh.add_vertex_field('int32',invalid_id)
    assert mesh.has_field(Vi)

    # Check size
    V = mesh.field(Vi)
    assert V.shape==(mesh.n_vertices,)

    # Remove a field
    mesh.remove_field(Vi)
    assert not mesh.has_field(Vi)

    # Permute vertices and check invariants
    Vi = mesh.add_vertex_field('i',invalid_id)
    assert mesh.has_field(Vi)
    V = mesh.field(Vi)
    for v in mesh.all_vertices():
      V[v] = mesh.halfedge(v)
    perm = random.permutation(mesh.n_vertices).astype(int32)
    mesh.permute_vertices(perm,False)

    # Update field data (vertex fields are invalidated by permute_vertices)
    V = mesh.field(Vi)
    for v in mesh.all_vertices():
      assert V[v] == mesh.halfedge(v)
    mesh.remove_field(Vi)

    # Randomly remove some faces (but no vertices) and collect garbage and check invariants
    print 'garbage collection test for faces...'

    # Make another (vector) field, which is stored as face colors
    Fi = mesh.add_face_field('3i',face_color_id)
    assert mesh.has_field(Fi)
    F = mesh.field(Fi)
    assert F.shape == (mesh.n_faces,3)

    # Write vertex ids into it
    nf = mesh.n_faces
    for f in mesh.all_faces():
      F[f] = mesh.face_vertices(f)
      # Delete half of the faces (but not vertices)
      if random_permute(nf,131371,f)<nf//2:
        mesh.erase_face(f,False)

    mesh.collect_garbage()
    F = mesh.field(Fi)

    # Since we didn't delete any vertices, their ordering hasn't changed, and we
    # should be connected to the same vertices as before
    for f in mesh.all_faces():
      assert all(F[f] == mesh.face_vertices(f))
    mesh.remove_field(Fi)

    # Split some faces, check that the content of new faces is as expected
    # (we're already checking consistency)

    # Flip some edges, check that the content of affected faces is as expected
    # (we're already checking consistency)

if __name__=='__main__':
  test_fields()
  test_corner_construction()
  test_halfedge_construction()
