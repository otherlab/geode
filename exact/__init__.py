'''Exact geometric computation'''

from __future__ import absolute_import
import platform

if platform.system()=='Windows':
  import other_all as other_core
else:
  import other_core

def delaunay_points(X,validate=False):
  return other_core.delaunay_points_py(X,validate)
