from __future__ import absolute_import

import platform
from numpy import *
if platform.system()=='Windows':
  from other_all import *
  import other_all as other_core
else:
  import other_core

class Nested(object):
  """Represents a nested array of arrays using flat storage for efficiency.
  This turns into the template class Nested<T> in C++.
  """

  __slots__ = ['offsets','flat']
  single_zero = zeros(1,dtype=int32)

  def __init__(self,x,dtype=None):
    if isinstance(x,Nested):
      offsets = x.offsets
      flat = x.flat
    elif isinstance(x,ndarray):
      m,n = x.shape
      offsets = n*arange(m+1,dtype=int32)
      flat = x.ravel()
    else:
      offsets = hstack([self.single_zero,cumsum([len(y) for y in x],dtype=int32)])
      flat = concatenate(x)
    object.__setattr__(self,"offsets",offsets)
    object.__setattr__(self,"flat",ascontiguousarray(flat,dtype=dtype))

  @staticmethod
  def zeros(lengths,dtype=int32):
    lengths = asarray(lengths)
    assert all(lengths>=0)
    self = object.__new__(Nested)
    object.__setattr__(self,'offsets',hstack([self.single_zero,cumsum(lengths,dtype=int32)]))
    self.offsets.setflags(write=False)
    object.__setattr__(self,'flat',zeros(self.offsets[-1],dtype=dtype))
    return self

  @staticmethod
  def empty(lengths,dtype=int32):
    lengths = asarray(lengths)
    assert all(lengths>=0)
    self = object.__new__(Nested)
    object.__setattr__(self,'offsets',hstack([self.single_zero,cumsum(lengths,dtype=int32)]))
    self.offsets.setflags(write=False)
    object.__setattr__(self,'flat',empty(self.offsets[-1],dtype=dtype))
    return self

  def __setattr__(*args):
    raise TypeError('Nested attributes cannot be set')

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
      other = Nested(other)
    except:
      raise NotImplementedError
    return all(self.offsets==other.offsets) and all(self.flat==other.flat)

  def __str__(self):
    return str([list(self[i]) for i in xrange(len(self))])

  def __repr__(self):
    return 'Nested(%s)'%repr([list(self[i]) for i in xrange(len(self))])

  def sizes(self):
    return self.offsets[1:]-self.offsets[:-1]

  @staticmethod
  def concatenate(*args):
    args = map(Nested,args)
    if len(args)<=1:
      return args[0]
    self = object.__new__(Nested)
    offsets = [args[0].offsets]
    flats = [args[0].flat]
    for a in args[1:]:
      offsets.append(offsets[-1][-1]+a.offsets[1:])
      flats.append(a.flat)
    object.__setattr__(self,'offsets',concatenate(offsets))
    object.__setattr__(self,'flat',concatenate(flats))
    return self

  # Support pickling
  def __getstate__(self):
    return self.offsets,self.flat
  def __setstate__(self,(offsets,flat)):
    assert offsets[0]==0
    assert offsets[-1]==len(flat)
    assert all(offsets[:-1]<=offsets[1:])
    object.__setattr__(self,'offsets',offsets)
    object.__setattr__(self,'flat',flat)

other_core._set_nested_array(Nested)
other_core._set_recarray_type(recarray)
