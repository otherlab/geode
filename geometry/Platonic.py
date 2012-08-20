"""geometry/Platonic.py"""

from __future__ import (division,absolute_import)

from ..vector import magnitude,Matrix,relative_error,normalized
from ..mesh import *
from .. import real

def close_find(l,x,tol=1e-5):
    for i,y in enumerate(l):
        if relative_error(x,y)<tol:
            return i
    return -1

def close_group(G,tol=1e-5):
    G = list(G)
    grew = 1
    while grew: 
        grew = 0
        for a in G:
            for b in G:
                ab = a*b
                if close_find(G,ab)>=0:
                    break
                else:
                    G.append(ab)
                    grew = 1
    return G

def icosahedral_group():
    P = Matrix([[0,1,0],[0,0,1],[1,0,0]],dtype=float)
    p = (+1+sqrt(5))/2
    q = (-1+sqrt(5))/2
    A = Matrix([[q,-p,1],[p,1,q],[-1,q,p]])/2
    assert allclose(dot(A.T,A),eye(3))
    G = close_group([P,A])
    assert len(G)==60
    return G

p = (1+sqrt(5))/2
_icosahedron_mesh = (TriangleMesh([(0,1,2),(0,2,4),(1,3,6),(0,3,1),(0,4,7),(2,5,8),(1,5,2),(1,6,5),(3,7,9),(0,7,3),(4,8,10),
                                   (2,8,4),(5,6,11),(6,9,11),(3,9,6),(4,10,7),(5,11,8),(7,10,9),(8,11,10),(9,10,11)]),
                     array([(1,p,0),(0,1,p),(p,0,1),(-1,p,0),(p,0,-1),(0,-1,p),(-p,-0,1),(0,1,-p),(1,-p,0),(-p,0,-1),(0,-1,-p),(-1,-p,0)]))
_icosahedron_mesh[1].flags.writeable = False
def icosahedron_mesh():
    return _icosahedron_mesh
    '''
    # The following "first principles" computation of an icosahedron mesh is slow enough that we've inlined the result into the code instead
    p = (1+sqrt(5))/2
    x0 = array([0,1,p])
    G = icosahedral_group()
    X = []
    for g in G:
        y = g*x0
        if close_find(X,y)<0:
            X.append(y)
    X = array(X)
    assert len(X)==12
    length = min(magnitude(y-x0) for y in X if not allclose(y,x0))
    assert allclose(length,2)
    for y in X:
        if allclose(magnitude(y-x0),length):
            x1 = y
            break
    for y in X:
        if allclose(magnitude(y-x0),length) and allclose(magnitude(y-x1),length):
            x2 = y
            break
    if dot(x0,cross(x1-x0,x2-x0))<0:
        x1,x2 = x2,x1
    triangles = []
    for g in G:
        y0,y1,y2 = [g*x for x in x0,x1,x2]
        i,j,k = [close_find(X,y) for y in y0,y1,y2]
        if i<j and i<k:
            triangles.append((i,j,k))
    return TriangleMesh(triangles),X
    '''

def sphere_mesh(refinements,center=0,radius=1):
    mesh,X = icosahedron_mesh()
    X = normalized(X)
    for _ in xrange(refinements):
        mesh,X = linear_subdivide(mesh,X)
        X = normalized(X)
    return mesh,center+radius*X

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
