"""utility module"""

from __future__ import absolute_import
from . import Log

from libother_core import *

def curry(f,*a,**k):
  def g(*a2,**k2):
    k3 = k.copy()
    k3.update(k2)
    return f(*(a+a2),**k3)
  return g
