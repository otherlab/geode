'''Base class to freeze the fields of Python objects'''

import contextlib

class Frozen(object):
  '''Inheriting from Frozen disallows new field creation except inside a thaw block'''

  def __setattr__(self,key,value):
    if hasattr(self,key) or hasattr(self,'_thawed'):
      object.__setattr__(self,key,value)
    else:
      raise TypeError("%s is a frozen class; can't create field '%s' outside a thaw block"%(type(self).__name__,key))

  @contextlib.contextmanager
  def thaw(self):
    '''Temporarily allow creation of new fields'''
    object.__setattr__(self,'_thawed',True)
    try:
      yield
    finally:
      object.__delattr__(self,'_thawed')
