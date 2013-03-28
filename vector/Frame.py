"""Class Frames

Frames store n-dimensional arrays of 2D/3D frames (rigid transforms),
represented as translation+rotation.  They can be converted
to and from Frame<TV> and Array<Frame<TV> > in O(1) time.
"""

from __future__ import (division,absolute_import)

import numpy
from numpy import *
from . import *
from . import Rotation
from .Rotation import Rotations
real = other_core.real

class Frames(ndarray):
    dtypes = dict((d,dtype([('t','%df%d'%(d,real.itemsize)),('r',Rotations[d].dtype)])) for d in (2,3))
    inv_dtypes = dict((v,k) for k,v in dtypes.items())
    __array_priority__ = -1.

    def __new__(cls,tr,r=None,d=None):
        if r is None:
          return asarray(tr,dtype=(None if d is None else cls.dtypes[d])).view(cls)
        t = asarray(tr)
        d = r.d
        assert t.shape[-1]==d
        f = empty(broadcast(t[...,0],r).shape,dtype=cls.dtypes[d]).view(cls)
        f.t = t
        f.r = r
        return f

    def __array_finalize__(self,*args):
        self.d = self.inv_dtypes[self.dtype]

    def __getitem__(self,k):
        x = ndarray.__getitem__(self,k)
        if x.ndim: return x
        return array(x).view(Frames)

    def t(self):
        return self.view(ndarray)['t']
    def set_t(self,t):
        self.view(ndarray)['t'] = t
    t = property(t,set_t)

    def r(self):
        return self.view(ndarray)['r'].view(Rotations[self.d])
    def set_r(self,r):
        self.view(Rotations[self.d])['r'] = r
    r = property(r,set_r)

    def __mul__(self,x):
        if isinstance(x,Frames):
            return Frames(self.t+self.r*x.t,self.r*x.r)
        return self.t+self.r*x

    d_to_size = {2:4,3:7}
    def reals(self):
        return (frombuffer(self.data,real) if self.data else empty(0,real)).reshape(self.shape+(self.d_to_size[self.d],))

    def inverse(self):
      ri = self.r.inverse()
      return Frames(ri*-self.t,ri)

    def __eq__(self,x):
        assert isinstance(x,Frames)
        return all(self.reals()==x.reals(),axis=-1)

    def __ne__(self,x):
        assert isinstance(x,Frames)
        return any(self.reals()!=x.reals(),axis=-1)

    def matrix(self):
        d = self.d
        m = zeros(self.shape+(d+1,d+1)).view(Matrix)
        mv = m.reshape(-1,d+1,d+1)
        mv[:,-1,-1] = 1
        mv[:,:d,-1] = self.t
        mv[:,:d,:d] = self.r.matrix()
        return m

def identity(d):
  return Frames(zeros(d),Rotation.identity(d))

def interpolation(f1,f2,s):
  interp = frame_interpolation_2d if f1.d==2 else frame_interpolation_3d
  if f1.shape!=f2.shape:
    b = broadcast(f1,f2)
    new_f = [empty(b.shape,dtype=f1.dtype).view(Frames) for _ in xrange(2)]
    new_f[0][:] = f1
    new_f[1][:] = f2
    f1,f2 = new_f
  return interp(f1.ravel(),f2.ravel(),s).reshape(f1.shape)

size_to_d = {4:2,7:3}
def from_reals(x):
  x = asarray(x)
  d = size_to_d[x.shape[-1]]
  return x.ravel().view(Frames.dtypes[d]).view(Frames).reshape(x.shape[:-1])

other_core._set_frame_type(Frames)
