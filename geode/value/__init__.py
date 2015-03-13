from __future__ import absolute_import

from geode import *
from . import parser
import types

def is_value(value):
  return isinstance(value, Value)

def is_prop(prop):
  return is_value(prop) and prop.is_prop()

def const_value(value, name=""):
  return const_value_py(value, name)

def Prop(name,default,shape=None):
  if shape is None:
    o = make_prop(name,None)
    o.set(default)
    return o
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
      if type(instance)==types.InstanceType:
        raise TypeError('cache_method can only be used on new-style classes (must inherit from object)')
      value = cache(types.MethodType(self.f,instance,owner))
      object.__setattr__(instance,self._name,value)
      return value

def cache_named(name):
  def inner(f):
    return cache_named_inner(f, name)
  return inner
