"""utility module"""

from __future__ import absolute_import
from . import Log
import platform
import tempfile
import types
import os

if platform.system()=='Windows':
  from other_all import resource_py,cache
else:
  from other_core import resource_py,cache

def curry(f,*a,**k):
  def g(*a2,**k2):
    k3 = k.copy()
    k3.update(k2)
    return f(*(a+a2),**k3)
  return g

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

def resource(*paths):
  return resource_py(os.path.join(*paths)) 

class _NamedTmpFile(object):
  def __init__(self,name,delete):
    self.name = name
    self.delete = delete
  def __del__(self):
    if self.delete:
      os.remove(self.name)

def named_tmpfile(suffix='', prefix='tmp', dir=None, delete=True):
  '''The Windows version of tempfile.NamedTemporaryFile doesn't not work for the most
  common case, since Windows does not allow one to open the same file twice.  Therefore,
  we implement our own, following Andre Pang's code from
    http://stackoverflow.com/questions/2549384/how-do-i-create-a-named-temporary-file-on-windows-in-python'''
  (file,name) = tempfile.mkstemp(prefix=prefix,suffix=suffix,dir=dir)
  os.close(file)
  return _NamedTmpFile(name,delete)
