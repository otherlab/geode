from __future__ import absolute_import

import platform
from numpy import *
if platform.system()=='Windows':
  from other_all import *
  from other_all import _set_nested_array
else:
  from other_core import *
  from other_core import _set_nested_array

class NestedArray(object):
  """Represents a nested array of arrays using flat storage for efficiency.
  This turns into the template class NestedArray<T> in C++.
  """

  __slots__=['offsets','flat']
  single_zero = zeros(1,dtype=int32)

  def __init__(self,x,dtype=None):
    if isinstance(x,NestedArray):
      object.__setattr__(self,"offsets",x.offsets)
      flat = x.flat
    else:
      object.__setattr__(self,"offsets",hstack([self.single_zero,cumsum([len(y) for y in x],dtype=int32)])) 
      flat = concatenate(x)
    if dtype is not None:
      flat = flat.astype(dtype)
    object.__setattr__(self,"flat",ascontiguousarray(flat))

  @staticmethod
  def zeros(lengths,dtype=int32):
    lengths = asarray(lengths)
    assert all(lengths>=0)
    self = object.__new__(NestedArray)
    object.__setattr__(self,'offsets',hstack([self.single_zero,cumsum(lengths,dtype=int32)]))
    self.offsets.setflags(write=False)
    object.__setattr__(self,'flat',zeros(self.offsets[-1],dtype=dtype))
    return self

  @staticmethod
  def empty(lengths,dtype=int32):
    lengths = asarray(lengths)
    assert all(lengths>=0)
    self = object.__new__(NestedArray)
    object.__setattr__(self,'offsets',hstack([self.single_zero,cumsum(lengths,dtype=int32)]))
    self.offsets.setflags(write=False)
    object.__setattr__(self,'flat',empty(self.offsets[-1],dtype=dtype))
    return self

  def __setattr__(*args):
    raise TypeError('NestedArray attributes cannot be set')

  def __len__(self):
    return len(self.offsets)-1

  def __getitem__(self,i):
    if isinstance(i,int) or isinstance(i,integer):
      return self.flat[self.offsets[i]:self.offsets[i+1]]
    else:
      i,j = i
      ij = self.offsets[:-1][i]+j
      assert all(0<=j)
      assert all(ij<self.offsets[1:][i])
      return self.flat[self.offsets[:-1][i]+j]

  def __eq__(self,other):
    try:
      other = NestedArray(other)
    except:
      raise NotImplementedError
    return all(self.offsets==other.offsets) and all(self.flat==other.flat)

  def __str__(self):
    return str([list(self[i]) for i in xrange(len(self))]) 

  def __repr__(self):
    return 'NestedArray(%s)'%repr([list(self[i]) for i in xrange(len(self))]) 

  def sizes(self):
    return self.offsets[1:]-self.offsets[:-1]

_set_nested_array(NestedArray)
