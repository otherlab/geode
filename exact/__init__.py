'''Exact geometric computation'''

from __future__ import absolute_import
import platform

from other.core.array import *

def delaunay_points(X,Mesh=CornerMesh,validate=False):
  if Mesh is HalfedgeMesh:
    return delaunay_points_halfedge(X,validate)
  elif Mesh is CornerMesh:
    return delaunay_points_corner(X,validate)
  raise TypeError('expected Mesh = HalfedgeMesh or CornerMesh, got %s'%Mesh)

def polygon_union(*polys):
  '''The union of possibly intersecting polygons, assuming consistent ordering'''
  return split_polygons(Nested.concatenate(*polys),0)

def polygon_intersection(*polys):
  '''The intersection of possibly intersecting polygons, assuming consistent ordering'''
  return split_polygons(Nested.concatenate(*polys),len(polys)-1)
