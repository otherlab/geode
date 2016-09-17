from __future__ import absolute_import

import platform
if platform.system()=='Windows':
  import other_all as geode_wrap
else:
  from .. import geode_wrap
from numpy import inf

def decimate_openmesh(mesh, max_collapses=(1<<31)-1, max_angle_error=90, max_quadric_error=inf, min_quality=1e-5):
  return geode_wrap.decimate_openmesh(mesh,max_collapses,max_angle_error,max_quadric_error,min_quality)
