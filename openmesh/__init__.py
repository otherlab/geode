from __future__ import absolute_import

import libother_core
from libother_core import *
from numpy import inf

def decimate(mesh, max_collapses=(1<<31)-1, max_angle_error=90, max_quadric_error=inf, min_quality=1e-5,min_boundary_dot=.99):
  return libother_core.decimate(mesh,max_collapses,max_angle_error,max_quadric_error,min_quality,min_boundary_dot)
