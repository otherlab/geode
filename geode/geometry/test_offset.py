#!/usr/bin/env python

from __future__ import division
from geode.geometry.platonic import *
from geode.vector import *

def test_offset():
  random.seed(8123171)
  offset = .1
  alpha = 1/4
  small = 1e-8
  def single():
    return TriangleSoup([(0,1,2)]),[(0,0,0),(1,0,0),(0,1,0)]
  for name,(m,X) in ('single',single()),('tet',tetrahedron_mesh()),('cube',cube_mesh()),('ico',icosahedron_mesh()):
    for i in xrange(10):
      # Generate a random affine transform, and make it rather singular
      if i:
        A = random.randn(3,3)
        for _ in xrange(2):
          A = dot(A,A)
        A *= linalg.det(A)**(-1/3)
        AX = Matrix(A)*X
      else:
        AX = asarray(X,dtype=float)

      for shell in 0,1:
        top = TriangleTopology(m)
        if not shell and top.has_boundary():
          continue
        print('%s : %s'%(name,('volume','shell')[shell]))
        # Perform offset
        om,oX = (rough_offset_shell if shell else rough_offset_mesh)(top,AX,offset)
        assert om.is_manifold()

        # Verify that random points on the surface have nice distances
        ns = 100
        w = random.rand(ns,3)
        w /= w.sum(axis=-1)[:,None]
        sX = (w[:,:,None]*oX[om.elements()[random.randint(0,om.n_faces,size=ns)]]).sum(axis=1)
        phi,_,_,_ = evaluate_surface_levelset(ParticleTree(sX),SimplexTree(m,AX),inf,not shell)
        assert all(alpha*offset <= phi+small)
        assert all(phi <= offset+small)

if __name__=='__main__':
  test_offset()
