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
      mesh.assert_consistent()
      try:
        mesh.add_faces(tris)
        mesh.assert_consistent()
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
      mesh.assert_consistent()
      for t in tris:
        try:
          assert mesh.has_boundary() or mesh.has_isolated_vertices()
          mesh.add_face(t)
          mesh.assert_consistent()
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
      mesh.assert_consistent()
      tris = mesh.elements()
      random.shuffle(tris)
      # Reconstruct the mangled mesh one triangle at a time
      partial = Mesh()
      partial.add_vertices(soup.nodes())
      check_add_faces(partial,tris)
      # Check that face (and maybe edge) splits are safe
      random_face_splits(mesh,20,key+10)
      mesh.assert_consistent()
      # Tear the mesh apart in random order
      mesh_destruction_test(mesh,key+20)
      assert mesh.n_vertices==mesh.n_edges==mesh.n_faces==0

def test_halfedge_construction():
  construction_test(HalfedgeMesh)

def test_corner_construction():
  construction_test(MutableTriangleTopology,corner_random_edge_flips,corner_random_face_splits,corner_mesh_destruction_test)

def test_properties():
  random.seed(813177)
  nanosphere = TriangleSoup([(0,1,2),(0,2,1)])
  for soup in nanosphere,icosahedron_mesh()[0],torus_topology(4,5),double_torus_mesh(),cylinder_topology(6,5):
    # Make a MutableTriangleTopology
    mesh = MutableTriangleTopology()
    mesh.add_vertices(soup.nodes())
    mesh.add_faces(soup.elements)

    # make a prop
    propid1 = mesh.add_vertex_property('int32', invalid_id)
    assert mesh.has_property(propid1)

    # check size
    prop1 = mesh.property(propid1)
    assert prop1.shape == (mesh.n_vertices,)

    # remove a prop
    mesh.remove_property(propid1)
    assert not mesh.has_property(propid1)

    # permute vertices and check invariants
    print 'permuting vertices...'
    vprop = mesh.add_vertex_property('i', invalid_id)
    assert mesh.has_property(vprop)
    vdata = mesh.property(vprop)

    for vertex in mesh.all_vertices():
      vdata[vertex] = mesh.halfedge(vertex)

    perm = array(random.permutation(mesh.n_vertices),dtype=int32)
    mesh.permute_vertices(perm, False)

    # update property data (vertex properties are invalidated by permute_vertices)
    vdata = mesh.property(vprop)

    for vertex in mesh.all_vertices():
      assert vdata[vertex] == mesh.halfedge(vertex)

    mesh.remove_property(vprop)

    # randomly remove some faces (but no vertices) and collect garbage and check invariants
    print 'garbage collection test for faces...'

    # make another (vector) prop, which is stored as face colors
    fprop = mesh.add_face_property('3i', face_color_id)
    assert mesh.has_property(fprop)

    # get the property memory
    fdata = mesh.property(fprop)
    assert fdata.shape == (mesh.n_faces,3)

    # write vertex ids into it
    for face in mesh.all_faces():
      fdata[face] = mesh.face_vertices(face)
      # delete around 50% of faces (but not vertices)
      if random.uniform() > .5:
        #print 'erasing face %s, vertices %s' % (face, mesh.face_vertices(face))
        mesh.erase_face(face, False)

    mesh.collect_garbage()
    fdata = mesh.property(fprop)

    # since we didn't delete any vertices, their ordering hasn't changed, and we
    # should be connected to the same vertices as before
    for face in mesh.all_faces():
      #print "face: %s, vertices %s, stored %s" % (face, fdata[face], mesh.face_vertices(face))
      assert all(fdata[face] == mesh.face_vertices(face))

    mesh.remove_property(fprop)

    # split some faces, check that the content of new faces is as expected
    # (we're already checking consistency)

    # flip some edges, check that the content of affected faces is as expected
    # (we're already checking consistency)

if __name__=='__main__':
  test_properties()
  test_corner_construction()
  test_halfedge_construction()

