#!/usr/bin/env python

from other.core.vector import *
from other.core.geometry import *
from other.core.random import Sobol
from math import pi

shapes = [
  Sphere((1,2),2),
  Sphere((1,2,3),2),
  Box((-1,-2),(1,2)),
  Box((-1,-2,-3),(1,2,3)),
  Capsule((-.5,-.5),(1,2),1),
  Capsule((-.5,-.5,-.5),(1,2,3),1),
  Cylinder(0,(0,0,1),1),
  Cylinder((-1,-2,-3),(4,2,1),1.5)]

def test_consistency():
  n=1000
  def sv(x):
    return '[%s]'%','.join(map(str,x))
  for shape in shapes:
    print shape
    box = shape.bounding_box()
    scale = max(box.sizes())
    sobol = Sobol(box.thickened(scale/pi))
    inner_box = empty_box(len(box.min))
    small = 1e-4*scale
    for _ in range(n):
      X = sobol.get_vector()
      if shape.lazy_inside(X):
        inner_box.enlarge(X)
      phi = shape.phi(X)
      normal = shape.normal(X)
      project = X-phi*normal
      phi2 = shape.phi(project)
      #print 'shape %r, X %s, phi %g'%(shape,sv(X),phi)
      assert abs(phi2)<small,'%s = %s - %g * %s, %g'%(sv(project),sv(X),phi,sv(normal),phi2)
      assert shape.lazy_inside(X)==(phi<=0)
      surface = shape.surface(X)
      assert abs(shape.phi(surface))<small, 'X %s, phi %g, surface %s, surface phi %g'%(sv(X),phi,sv(surface),shape.phi(surface))
      surface_phi = magnitude(surface-X)
      assert abs(surface_phi-abs(phi))<small,'%s != %s'%(surface_phi,phi)
      # TODO: test Boundary and Principal_Curvatures
    assert box.lazy_inside(inner_box.min)
    assert box.lazy_inside(inner_box.max)
    assert max(box.sizes()-inner_box.sizes())<scale/4,'%s != %s'%(box,inner_box)

"""
def test_generate_triangles():
  tolerance=1e-5
  for shape in shapes:
    print shape
    surface=shape.generate_triangles()
    if not isinstance(shape,PLANE_f):
      assert not surface.mesh.non_manifold_nodes()
    assert surface.mesh.orientations_consistent()
    particles=surface.particles
    for X in particles.X:
      assert abs(shape.phi(X))<tolerance
    for t in range(1,len(surface.mesh.elements)+1):
      i,j,k=surface.mesh.elements[t]
      X=(particles.X[i]+particles.X[j]+particles.X[k])/3
      shape_normal=shape.normal(X)
      surface_normal=surface.normal(X,t)
      dot=dot(shape_normal,surface_normal)
      assert dot>.7,'(%s) . (%s) = %s'%(shape_normal,surface_normal,dot)
"""
