"""utility module"""

from __future__ import absolute_import
from . import Log
import platform
import os

if platform.system()=='Windows':
  import other_all as other_core
else:
  import other_core

def curry(f,*a,**k):
  def g(*a2,**k2):
    k3 = k.copy()
    k3.update(k2)
    return f(*(a+a2),**k3)
  return g

def resource(*paths):
  return other_core.resource(os.path.join(*paths)) 
