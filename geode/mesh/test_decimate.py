#!/usr/bin/env python

from __future__ import division,print_function
from geode import *
from geode.geometry.platonic import *
import unittest

def hausdorff(A,B,boundary=False):
  m = A[0],B[0] 
  X = A[1],B[1]
  def f(i,j):
    if boundary:
      Y = X[i][map(m[i].src,m[i].boundary_loops().flat)]
      surface = SimplexTree(SegmentSoup(map(m[j].halfedge_vertices,m[j].boundary_loops().flat)),X[j])
    else:
      Y = X[i]
      surface = m[j].face_tree(X[j])[0]
    return surface_levelset(ParticleTree(Y),surface)[0].max()
  return max(f(0,1),f(1,0))

@unittest.skip("assert 1 == 2")
def test_decimate():
  # fail this test -- it is hanging
  assert 1 == 2
  for steps in 2,3:
    mesh = TriangleSoup([(0,1,2),(0,2,3),(0,3,1)])
    _,X = tetrahedron_mesh()
    mesh,X = loop_subdivide(mesh,X,steps=steps)
    mesh = TriangleTopology(mesh)
    def test(distance,boundary_distance=0):
      md,Xd = decimate(mesh,X,distance=distance,boundary_distance=boundary_distance)
      H = hausdorff((mesh,X),(md,Xd))
      Hb = hausdorff((mesh,X),(md,Xd),boundary=1)
      print('distance %g, boundary %g, H %g, Hb %g'%(distance,boundary_distance,H,Hb))
      assert H<=distance
      assert Hb<=boundary_distance
    test(distance=.01)
    test(distance=.05,boundary_distance=.02)
    test(distance=3,boundary_distance=.1)
    test(distance=inf,boundary_distance=inf)

if __name__ == '__main__':
  test_decimate()
