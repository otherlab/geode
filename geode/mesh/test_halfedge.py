#!/usr/bin/env python

from __future__ import division
from geode import *
from geode.geometry.platonic import *

def construction_test(Mesh,random_edge_flips=random_edge_flips,random_face_splits=random_face_splits,mesh_destruction_test=mesh_destruction_test):
  random.seed(813177)
  nanosphere = TriangleSoup([(0,1,2),(0,2,1)])
  print
  print Mesh.__name__
  for soup in nanosphere,icosahedron_mesh()[0],torus_topology(4,5),double_torus_mesh(),cylinder_topology(6,5):
    # Learn about triangle soup
    n_vertices = soup.nodes()
    n_edges = len(soup.segment_mesh().elements)
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
      # Check that face splits are safe
      random_face_splits(mesh,20,key+10)
      mesh.assert_consistent()
      # Tear the mesh apart in random order
      mesh_destruction_test(mesh,key+20)
      assert mesh.n_vertices==mesh.n_edges==mesh.n_faces==0

def test_halfedge_construction():
  construction_test(HalfedgeMesh)

def test_corner_construction():
  construction_test(TriangleTopology,corner_random_edge_flips,corner_random_face_splits,corner_mesh_destruction_test)

if __name__=='__main__':
  test_corner_construction()
  test_halfedge_construction()

