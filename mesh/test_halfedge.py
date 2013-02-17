#!/usr/bin/env python

from __future__ import division
from other.core import *
from other.core.geometry.platonic import *

def test_construction():
  random.seed(813177)
  nanosphere = TriangleMesh([(0,1,2),(0,2,1)])
  print
  for soup in nanosphere,icosahedron_mesh()[0],torus_topology(4,5),double_torus_mesh(),cylinder_topology(5,6):
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
      assert mesh.n_halfedges==2*n_edges
      assert mesh.n_faces==n_faces
      assert mesh.chi==chi

    def check_add_faces(mesh,tris):
      base.assert_consistent()
      for t in tris:
        try:
          assert mesh.has_boundary()
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
    base = HalfedgeMesh()
    base.add_vertices(soup.nodes())
    check_add_faces(base,soup.elements)

    for key in xrange(5):
      # Mangle the clean mesh using a bunch of edge flips
      mesh = base.copy()
      random_edge_flips(mesh,1000,key)
      mesh.assert_consistent()
      tris = mesh.elements()
      random.shuffle(tris)
      # Reconstruct the mangled mesh one triangle at a time
      partial = HalfedgeMesh()
      partial.add_vertices(soup.nodes())
      check_add_faces(partial,tris)

if __name__=='__main__':
  test_construction()
