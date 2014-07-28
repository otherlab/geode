'''Exact geometric computation'''

from __future__ import absolute_import
import platform
from geode.array import *
from geode.mesh import merge_meshes, meshify
if platform.system()=='Windows':
  from other_all import *
  import other_all as geode_wrap
else:
  from .. import geode_wrap
  from ..geode_wrap import *

if not has_exact():
  raise ImportError('geode/exact is unavailable since geode was compiled without gmp support')

def delaunay_points(X,edges=zeros((0,2),dtype=int32),validate=False):
  return delaunay_points_py(X,edges,validate)

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

geode_wrap._set_circle_arc_dtypes(CircleArc,ExactCircleArc)

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

def split_soup(mesh,X,depth=0):
  '''If depth is None, extract nonmanifold mesh with triangles at all depths'''
  if depth is None:
    depth = -1<<31
  return geode_wrap.split_soup(mesh,X,depth)

def split_soup_with_weight(mesh,X,weight,depth=0):
  if depth is None:
    depth = -1<<31
  return geode_wrap.split_soup_with_weight(mesh,X,weight,depth)

def split_soups(meshes,depth=0):
  return split_soup(*merge_meshes(meshes),depth=depth)

def soup_union(*meshes):
  return split_soups(meshes,depth=0)

def soup_intersection(*meshes):
  return split_soups(meshes,depth=len(meshes)-1)

def split_mesh_with_weight(mesh, weights, depth=0):
  return meshify(*split_soup_with_weight(mesh.face_soup()[0], mesh.vertex_field(vertex_position_id), weights, depth))

def split_mesh(mesh, depth=0):
  return meshify(*split_soup(mesh.face_soup()[0], mesh.vertex_field(vertex_position_id), depth))

def split_meshes(meshes, depth=0):
  soups = [(m.face_soup()[0], m.vertex_field(vertex_position_id)) for m in meshes]
  return meshify(*split_soups(soups,depth))

# meshes is an array of tuples (mesh,weight) -- one uniform weight for each input mesh
def split_meshes_with_weight(meshes_and_weights, depth=0):
  soups = [(m.face_soup()[0], m.vertex_field(vertex_position_id)) for m,weight in meshes_and_weights]
  soup,X = merge_meshes(soups)
  weights = []
  for m,weight in meshes_and_weights:
    weights += [weight] * m.n_faces
  return meshify(*split_soup_with_weight(soup, X, weights, depth))
