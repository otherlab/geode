#!/usr/bin/env python

from __future__ import division
from other.core.vector import *
from other.core.openmesh import *
from other.core.geometry.platonic import *

if openmesh_enabled():
  def test_trimesh():
    # Make sure we can create one and then check its exposed functions
    trimesh = TriMesh()
    v0 = trimesh.add_vertex((0,0,0))
    v1 = trimesh.add_vertex((1,0,0))
    v2 = trimesh.add_vertex((0,1,0))
    assert (v0,v1,v2)==(0,1,2)
    assert trimesh.n_vertices()==3
    f = trimesh.add_face(v0,v1,v2)
    assert f==0
    assert trimesh.n_faces()==1

  def test_curvature():
    r = e
    known = [('sphere',       sphere_mesh(4,radius=r),1/r,1/(r*r))
            ,('cylinder',     open_cylinder_mesh(x0=0,x1=(1,0,0),radius= r,na=1000,nz=10), .5/r,0)]
    for name,(mesh,X),H,K in known:
      for scale in 1,-1:
        print '\n%s %g'%(name,scale)
        H *= scale
        tm = TriMesh()
        tm.add_vertices(scale*X)
        tm.add_faces(mesh.elements)
        Ha = mean_curvatures(tm)
        He = relative_error(H,Ha)
        print 'H error = %g (range %g %g, correct %g)'%(He,Ha.min(),Ha.max(),H)
        assert He<1e-5
        Ka = gaussian_curvatures(tm)
        Ke = relative_error(K,Ka,absolute=1e-6)
        print 'K error = %g (range %g %g, correct %g)'%(Ke,Ka.min(),Ka.max(),K)
        assert Ke<2e-3

if __name__=='__main__':
  test_trimesh()
  test_curvature()
