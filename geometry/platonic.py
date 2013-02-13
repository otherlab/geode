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

def cylinder_topology(na,nz,closed=False):
  '''Construct a open cylinder TriangleMesh with na triangles around and nz along.
  closed can be either a single bool or an array of two bools (one for each end).'''
  closed = asarray(closed)
  c0,c1 = closed if closed.ndim else (closed,closed)
  i = arange(na)
  j = arange(nz).reshape(-1,1)
  tris = empty((nz,na,2,3),dtype=int32)
  ip = (i+1)%na
  tris[:,:,0,0] = tris[:,:,1,0] = na*j+ip
  tris[:,:,0,1] = na*j+i
  tris[:,:,0,2] = tris[:,:,1,1] = na*(j+1)+i
  tris[:,:,1,2] = na*(j+1)+ip
  if c0 and c1: tris = concatenate([tris[0,:,1],tris[1:-1].reshape(-1,3),tris[-1,:,0]])
  elif c0:      tris = concatenate([tris[0,:,1],tris[1:  ].reshape(-1,3)])
  elif c1:      tris = concatenate([            tris[ :-1].reshape(-1,3),tris[-1,:,0]])
  if c1: tris = minimum(tris.ravel(),na*nz)
  if c0: tris = maximum(0,tris.ravel()-(na-1))
  return TriangleMesh(tris.reshape(-1,3))

def surface_of_revolution(base,axis,radius,height,resolution,closed=False):
  '''Construct a surface of revolution with given radius and height curves.
  closed can be either a single bool or an array of two bools (one for each end).
  For each closed end, height should have one more point than radius.'''
  closed = asarray(closed,dtype=int32)
  c0,c1 = closed if closed.ndim else (closed,closed)
  assert radius.ndim<=1 and height.ndim<=1
  assert height.size>=1+c0+c1
  height = height.reshape(-1)
  axis = asarray(axis)
  x = unit_orthogonal_vector(axis)
  y = normalized(cross(axis,x))
  a = 2*pi/resolution*arange(resolution)
  circle = x*cos(a).reshape(-1,1)-y*sin(a).reshape(-1,1)
  X = base+radius[...,None,None]*circle+height[c0:len(height)-c1,None,None]*axis
  X = concatenate(([[base+height[0]*axis]] if c0 else []) + [X.reshape(-1,3)] + ([[base+height[-1]*axis]] if c1 else []))
  return cylinder_topology(resolution,len(height)-1,closed=closed),X

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
  return cylinder_topology(na,nz),X.reshape(-1,3)

def capsule_mesh(x0,x1,radius,n=30):
  x0 = asarray(x0)
  length,axis = magnitudes_and_normalized(x1-x0)
  theta = linspace(0,pi/2,(n+1)//2)
  r = radius*cos(theta[:-1])
  h = radius*sin(theta)
  r = hstack([r[::-1],r])
  h = hstack([-h[::-1],h+length])
  return surface_of_revolution(x0,axis,r,h,n,closed=True)
