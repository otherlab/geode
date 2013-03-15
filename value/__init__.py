from __future__ import absolute_import

from other.core import *
from . import parser

def Prop(name,default,shape=None):
  if shape is None:
    return make_prop(name,default)
  return make_prop_shape(name,default,shape)
