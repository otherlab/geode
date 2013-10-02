from __future__ import absolute_import

from other.core import *
from . import parser

def Prop(name,default,shape=None):
  if shape is None:
    return make_prop(name,default)
  return make_prop_shape(name,default,shape)

class cache_method(object):
  '''Decorator to cache a class method per instance.  The equivalent of 'cache' in the function case.'''
  def __init__(self,f):
    self._name = '__'+f.__name__
    self.f = f
  def __get__(self,instance,owner):
    try:
      return getattr(instance,self._name)
    except AttributeError:
      value = cache(types.MethodType(self.f,instance,owner))
      object.__setattr__(instance,self._name,value)
      return value
