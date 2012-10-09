from __future__ import absolute_import

import libother_openmesh
from libother_openmesh import *
from numpy import inf

def decimate(mesh, max_collapses=(1<<31)-1, max_angle_error=90, max_quadric_error=inf):
  return libother_openmesh.decimate(mesh,max_collapses,max_angle_error,max_quadric_error)
