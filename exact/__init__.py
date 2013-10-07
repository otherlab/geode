'''Exact geometric computation'''

from __future__ import absolute_import
import platform
from other.core.array import *
if platform.system()=='Windows':
  from other_all import *
  import other_all as other_core
else:
  from other_core import *
  import other_core

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

# Must be kept in sync with the C++ types in circle_csg.cpp.
CircleArc = dtype([('x','2f%d'%real.itemsize),
                   ('q','f%d'%real.itemsize)])
ExactCircleArc = dtype([('center','2i8'),
                        ('radius','i8'),
                        ('index','i4'),
                        ('positive','b'),
                        ('left','b'),
                        ('_pad','2V')]) # Work around https://github.com/numpy/numpy/issues/2383

other_core._set_circle_arc_dtypes(CircleArc,ExactCircleArc)

def circle_arc_union(*arcs):
  '''The union of possibly intersecting circular arc polygons, assuming consistent ordering'''
  all_arcs = Nested.concatenate(*arcs)
  split = split_circle_arcs if all_arcs.flat.dtype==CircleArc else exact_split_circle_arcs
  return split(all_arcs,0)

def circle_arc_intersection(*arcs):
  '''The intersection of possibly intersecting circular arc polygons, assuming consistent ordering'''
  all_arcs = Nested.concatenate(*arcs)
  split = split_circle_arcs if all_arcs.flat.dtype==CircleArc else exact_split_circle_arcs
  return split(all_arcs,len(arcs)-1)
