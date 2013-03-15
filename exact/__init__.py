'''Exact geometric computation'''

from __future__ import absolute_import
import platform

if platform.system()=='Windows':
  import other_all as other_core
else:
  import other_core

def delaunay_points(X,Mesh=other_core.CornerMesh,validate=False):
  if Mesh is other_core.HalfedgeMesh:
    return other_core.delaunay_points_halfedge(X,validate)
  elif Mesh is other_core.CornerMesh:
    return other_core.delaunay_points_corner(X,validate)
  raise TypeError('expected Mesh = HalfedgeMesh or CornerMesh, got %s'%Mesh)
