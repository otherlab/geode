"""utility module"""

from __future__ import absolute_import
from . import Log
import os

import libother_core
from libother_core import *

def curry(f,*a,**k):
  def g(*a2,**k2):
    k3 = k.copy()
    k3.update(k2)
    return f(*(a+a2),**k3)
  return g

def resource(*paths):
  return libother_core.resource(os.path.join(*paths)) 
