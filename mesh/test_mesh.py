#!/usr/bin/env python

from __future__ import division

from numpy import *
from other.core.mesh import PolygonMesh, SegmentMesh, TriangleMesh
from other.core.geometry.platonic import icosahedron_mesh, sphere_mesh
from other.core.vector import relative_error

def test_misc():
    counts = array([3,4],dtype=int32)
    vertices = array([0,1,2,2,1,3,4],dtype=int32)
    polygon_mesh = PolygonMesh(counts,vertices)
    assert all(polygon_mesh.counts==counts)
    assert all(polygon_mesh.vertices==vertices)
    triangle_mesh = polygon_mesh.triangle_mesh()
    triangles = array([0,1,2,2,1,3,2,3,4]).reshape([-1,3])
    assert all(triangle_mesh.elements==triangles)

    def make_set(s):
        return set(tuple(sorted(a)) for a in s)
    segments=make_set(array([0,1,1,2,2,0,2,1,1,3,3,4,4,2],dtype=int32).reshape(-1,2))
    assert make_set(polygon_mesh.segment_mesh().elements)==segments
    assert make_set(triangle_mesh.segment_mesh().elements)==segments|set([(2,3)])
    
def test_bad():
    try:
        PolygonMesh(array([-2],dtype=int32),array([1],dtype=int32))
        assert False
    except AssertionError:
        pass

def test_incident_segments():
    mesh = SegmentMesh([(0,1),(0,2)])
    incident = mesh.incident_elements()
    print incident.offsets
    print incident.flat
    incident = map(list,mesh.incident_elements())
    assert incident==[[0,1],[0],[1]]

def test_incident_triangles():
    mesh = TriangleMesh([(0,1,2),(0,3,4)])
    incident =  mesh.incident_elements()
    incident = map(list,incident)
    assert incident==[[0,1],[0],[0],[1],[1]]

def test_boundary_mesh():
    mesh = TriangleMesh([(0,1,2),(2,1,3)])
    boundary = mesh.boundary_mesh()
    assert all(boundary.elements==[[0,1],[2,0],[1,3],[3,2]])

def test_bending_quadruples():
    mesh = TriangleMesh([(0,1,2),(2,1,3)])
    assert all(mesh.bending_quadruples()==[(0,1,2,3)])

def test_adjacent_segments():
    mesh = SegmentMesh([(0,1),(1,2)])
    assert all(mesh.adjacent_elements()==[(-1,1),(0,-1)])

def test_adjacent_triangles():
    mesh = TriangleMesh([(0,1,2),(2,1,3)])
    assert all(mesh.adjacent_elements()==[(-1,1,-1),(0,-1,-1)])

def test_nodes_touched():
    mesh = TriangleMesh([(4,7,5)])
    assert all(mesh.nodes_touched()==[4,5,7])

def test_volume():
    mesh,X = icosahedron_mesh()
    a = 2 # see http://en.wikipedia.org/wiki/Icosahedron
    assert relative_error(mesh.surface_area(X),5*sqrt(3)*a**2) < 1e-5
    assert relative_error(mesh.volume(X),5/12*(3+sqrt(5))*a**3) < 1e-5
    mesh,X = sphere_mesh(4)
    assert relative_error(mesh.surface_area(X),4*pi) < .01
    assert relative_error(mesh.volume(X),4/3*pi) < .01

def test_neighbors():
    mesh = SegmentMesh([(0,1),(0,2),(0,2)])
    assert all(mesh.neighbors()==[[1,2],[0],[0]])

def injection(n):
  map = random.randint(5,5+n*n,n).astype(int32)
  assert len(unique(map))==len(map)
  return map

def test_nonmanifold_segments():
  random.seed(71318)
  map = injection(12)
  mesh = SegmentMesh(map[asarray([(0,0), # degenerate segment
                                  (1,2),(3,4), # open curve
                                  (4,5),(4,6), # two outgoing segments
                                  (8,7),(9,7), # two incoming segments
                                  (10,11),(11,10)],int32)]) # closed loop
  assert all(mesh.nonmanifold_nodes(False)==sort(map[asarray(xrange(10))]))
  assert all(mesh.nonmanifold_nodes(True)==sort(map[asarray([0,4,7])]))

def test_nonmanifold_triangles():
  random.seed(71318)
  triangles = array([(0,1,1),(2,3,2),(4,4,5), # degenerate triangles
                     (6,7,8), # lone triangle
                     (9,10,11),(11,10,12), # correctly oriented pair
                     (13,14,15),(14,15,16), # incorrectly oriented pair
                     (17,18,19),(17,19,18),(17,20,21),(17,21,20), # touching spheres
                     (22,23,24),(22,25,26), # touching triangles
                     (27,28,29),(27,29,28)],int32) # manifold sphere
  closed = asarray(range(18)+range(22,27))
  open = asarray(range(6)+[14,15,17,22])
  for _ in xrange(5):
    # Results should be covariant under sparsity and permutation
    map = injection(30)
    tris = map[triangles]
    # Results should be invariant under cyclic permutation of triangles
    s = random.randint(3,size=len(tris)).reshape(-1,1)
    tris = hstack([tris,tris])[arange(len(tris)).reshape(-1,1),hstack([s,s+1,s+2])]
    # Are we good?
    mesh = TriangleMesh(ascontiguousarray(tris))
    assert all(mesh.nonmanifold_nodes(False)==sort(map[closed]))
    assert all(mesh.nonmanifold_nodes(True)==sort(map[open]))
