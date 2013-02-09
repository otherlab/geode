'''Convenience functions for generating platonic solids
   See also platonic.h'''

from __future__ import (division,absolute_import)
from other.core import *

def sphere_mesh(refinements,center=0,radius=1):
  return sphere_mesh_py(refinements,center,radius)

def tetrahedron_mesh():
  X = array([[ 1, 1, 1],[-1,-1, 1],[-1,+1,-1],[+1,-1,-1]],dtype=real)
  mesh = TriangleMesh([[1,2,3],[3,2,0],[3,0,1],[0,2,1]])
  return mesh,X

def cube_mesh():
  X = array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],dtype=real)
  mesh = TriangleMesh([[0,1,2], [2,1,3], 
                       [1,0,5], [5,0,4], 
                       [3,1,7], [7,1,5], 
                       [0,2,4], [4,2,6], 
                       [2,3,6], [6,3,7], 
                       [5,6,7], [6,5,4]])
  return mesh,X

def circle_mesh(n,center=0,radius=1):
  i = arange(n,dtype=int32)
  segments = empty((n,2),dtype=int32)
  segments[:,0] = i
  segments[:-1,1] = i[1:]
  segments[-1,1] = 0
  mesh = SegmentMesh(segments)
  if center is None:
    return mesh
  theta = 2*pi/n*i
  return mesh,(radius*vstack([cos(theta),sin(theta)])).T.copy()

def open_cylinder_topology(na,nz):
  '''Construct a open cylinder TriangleMesh with na triangles around and nz along'''
  i = arange(na)
  j = arange(nz).reshape(-1,1)
  tris = empty((nz,na,2,3),dtype=int32)
  ip = (i+1)%na
  tris[:,:,0,0] = tris[:,:,1,0] = na*j+ip
  tris[:,:,0,1] = na*j+i
  tris[:,:,0,2] = tris[:,:,1,1] = na*(j+1)+i
  tris[:,:,1,2] = na*(j+1)+ip
  return TriangleMesh(tris.reshape(-1,3))

def surface_of_revolution(base,axis,radius,height,resolution):
  '''Construct a surface of revolution with radius and height curves'''
  shape = broadcast(radius,height).shape
  assert len(shape)==1
  x = unit_orthogonal_vector(axis)
  y = normalized(cross(axis,x))
  a = 2*pi/resolution*arange(resolution)
  circle = x*cos(a).reshape(-1,1)-y*sin(a).reshape(-1,1)
  X = base+radius[...,None,None]*circle+height[...,None,None]*axis
  return open_cylinder_topology(resolution,shape[0]-1),X.reshape(-1,3)

def open_cylinder_mesh(x0,x1,radius,na,nz=None):
  '''radius may be a scalar or a 1d array'''
  radius = asarray(radius)
  if nz is None:
    assert radius.ndim<2
    nz = 1 if radius.ndim==0 else len(radius)-1
  else:
    assert radius.shape in ((),(nz+1,))
  x0 = asarray(x0)
  x1 = asarray(x1)
  z = normalized(x1-x0)
  x = unit_orthogonal_vector(z)
  y = cross(z,x)
  i = arange(na)
  a = 2*pi/na*i
  circle = x*cos(a).reshape(-1,1)-y*sin(a).reshape(-1,1)
  height = arange(nz+1)/(nz+1)
  X = x0+radius[...,None,None]*circle+arange(nz+1).reshape(-1,1,1)/nz*(x1-x0)
  return open_cylinder_topology(na,nz),X.reshape(-1,3)
