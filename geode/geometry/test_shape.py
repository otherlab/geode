#!/usr/bin/env python

from __future__ import division
from geode import *
from geode.random import Sobol

def test_segments():
  n = 10000
  segment_tests_2d(n)
  segment_tests_3d(n)

def test_bounding_box():
  def check(X,min,max):
    box = bounding_box(X)
    try:
      assert all(box.min==min)
      assert all(box.max==max)
    except:
      print 'check failed: X %s, box %s, expected %s %s'%(X,box,min,max)
      raise
  check([[4,-2],[-1,5]],[-1,-2],[4,5])
  check(asarray([[4.,-2],[-1,5]]),[-1,-2],[4,5])
  check([[[1,2,3],[2,4,2]],[7,-1,-2]],[1,-1,-2],[7,4,3])

def test_consistency():
  random.seed(98183)
  shapes = [
    Sphere((1,2),2),
    Sphere((1,2,3),2),
    Box((-1,-2),(1,2)),
    Box((-1,-2,-3),(1,2,3)),
    Capsule((-.5,-.5),(1,2),1),
    Capsule((-.5,-.5,-.5),(1,2,3),1),
    Cylinder(0,(0,0,1),1),
    Cylinder((-1,-2,-3),(4,2,1),1.5),
    ThickShell(TriangleSoup([(0,1,2)]),[(0,0,0),(1,0,0),(0,1,0)],(.1,.2,.3)),
    ThickShell(TriangleSoup([(0,1,2)]),random.randn(3,3),.2*abs(random.randn(3)))]
  n = 1000
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
      X = sobol.vector()
      if shape.lazy_inside(X):
        inner_box.enlarge(X)
      phi = shape.phi(X)
      normal = shape.normal(X)
      project = X-phi*normal
      phi2 = shape.phi(project)
      #print 'shape %r, X %s, phi %g, normal %s'%(shape,sv(X),phi,sv(normal))
      if abs(phi2)>=small:
        print 'project = %s, X = %s, phi = %g, normal = %s, phi2 = %g, small = %g'%(sv(project),sv(X),phi,sv(normal),phi2,small)
        assert abs(phi2)<small,'%s = %s - %g * %s, abs(%g) !< %g'%(sv(project),sv(X),phi,sv(normal),phi2,small)
      assert shape.lazy_inside(X)==(phi<=0)
      surface = shape.surface(X)
      inner_box.enlarge(surface)
      assert abs(shape.phi(surface))<small, 'X %s, phi %g, surface %s, surface phi %g'%(sv(X),phi,sv(surface),shape.phi(surface))
      surface_phi = magnitude(surface-X)
      assert abs(surface_phi-abs(phi))<small,'%s != %s'%(surface_phi,phi)
      # TODO: test boundary and principal_curvatures
    assert box.lazy_inside(inner_box.min)
    assert box.lazy_inside(inner_box.max)
    box_error = max(box.sizes()-inner_box.sizes())/scale
    if box_error>5e-3:
      print 'box error %g'%box_error
      print 'box %s, sizes %s, volume %g\ninner box %s, sizes %s, volume %g'%(box,box.sizes(),box.volume(),inner_box,inner_box.sizes(),inner_box.volume())
      assert False

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

if __name__=='__main__':
  test_segments()
  test_bounding_box()
  test_consistency()
