"""Class Rotations

Rotations store n-dimensional arrays of 2D/3D rotations,
represented as complex numbers or quaternions.  They can be converted
to and from Rotation<TV> and Array<Rotation<TV> > in O(1) time.
"""

from __future__ import (division,absolute_import)

import numpy
from numpy import *
from . import *
real = geode_wrap.real

class Rotations2d(ndarray):
    d = 2
    dtype = dtype('c%d'%(2*real.itemsize))
    __array_priority__ = -1.

    def __array_finalize__(self,*args):
        assert self.dtype==type(self).dtype

    def __getitem__(self,k):
        x = ndarray.__getitem__(self,k)
        if x.ndim: return x
        return array(x).view(Rotations2d)

    def __mul__(self,x):
        if isinstance(x,Rotations2d):
            return ndarray.__mul__(self,x)
        x = asarray(x)
        assert x.shape[-1]==2
        return ascontiguousarray(x.astype(real)).view(self.dtype).__mul__(self[...,None]).view(type(x)).view(real)

    def inverse(self):
        return conjugate(self)

    def __eq__(self,x):
        assert isinstance(x,Rotations2d)
        return self.view(ndarray)==x.view(ndarray)

    def __ne__(self,x):
        assert isinstance(x,Rotations2d)
        return self.view(ndarray)!=x.view(ndarray)

    def matrix(self):
        m = empty(self.shape+(2,2)).view(Matrix)
        mv = m.reshape(-1,4)
        mv[:,0] = self.real
        mv[:,1] = -self.imag
        mv[:,2] = self.imag
        mv[:,3] = self.real
        return m

class Rotations3d(ndarray):
    d = 3
    dtype = dtype([('s','f%d'%real.itemsize),('v','3f%d'%real.itemsize)])
    __array_priority__ = -1.

    def __array_finalize__(self,*args):
        assert self.dtype==type(self).dtype

    def __getitem__(self,k):
        x = ndarray.__getitem__(self,k)
        if x.ndim: return x
        return array(x).view(Rotations3d)

    def s(self):
        return self.view(ndarray)['s']
    def set_s(self,s):
        self.view(ndarray)['s'] = s
    s = property(s,set_s)

    def v(self):
        return self.view(ndarray)['v']
    def set_v(self,v):
        self.view(ndarray)['v'] = v
    v = property(v,set_v)

    @property
    def sv(self):
        return self.view(ndarray).ravel().view(real).reshape(self.shape+(4,))

    def __mul__(self,x):
        s = self.s
        v = self.v
        if isinstance(x,Rotations3d):
            xs = x.s
            xv = x.v
            nv = s[...,None]*xv+xs[...,None]*v+cross(v,xv)
            q = empty(nv.shape[:-1],dtype=Rotations3d.dtype).view(Rotations3d)
            q.s = s*xs-dots(v,xv)
            q.v = nv
            return q
        x = asarray(x)
        assert x.shape[-1]==3
        return 2*s[...,None]*cross(v,x)+(sqr(s)-sqr_magnitudes(v))[...,None]*x+(2*dots(v,x))[...,None]*v

    def inverse(self):
        q = empty(self.shape,dtype=self.dtype).view(Rotations3d)
        q.s = self.s
        q.v = -self.v
        return q

    def reals(self):
        return frombuffer(self.data,real).reshape(self.shape+(-1,))

    def matrix(self):
        m = empty(self.shape+(3,3)).view(Matrix)
        mv = m.reshape(-1,3,3)
        s = self.s.reshape(-1,1)
        v = self.v.reshape(-1,3)
        v2 = 2*v
        vv = v*v2
        off = v[:,1]*v2[:,2],v[:,0]*v2[:,2],v[:,0]*v2[:,1]
        sv = s*v2
        mv[:,0,0] = 1-vv[:,1]-vv[:,2]
        mv[:,1,1] = 1-vv[:,2]-vv[:,0]
        mv[:,2,2] = 1-vv[:,0]-vv[:,1]
        mv[:,0,1] = off[2]-sv[:,2]
        mv[:,1,0] = off[2]+sv[:,2]
        mv[:,1,2] = off[0]-sv[:,0]
        mv[:,2,1] = off[0]+sv[:,0]
        mv[:,2,0] = off[1]-sv[:,1]
        mv[:,0,2] = off[1]+sv[:,1]
        return m

    def angle_axis(self):
        m,v = magnitudes_and_normalized(self.v)
        v.reshape(-1,3)[asarray(self.s).reshape(-1)<0] *= -1
        return 2*atan2(m,abs(self.s)),v

    def rotation_vector(self):
        m,v = magnitudes_and_normalized(self.v)
        return atan2(m,self.s)[...,None]*v

Rotations = {2:Rotations2d,3:Rotations3d}

def from_angle(angle):
    angle = asarray(angle)
    r = empty(angle.shape,dtype=Rotations2d.dtype)
    r.real = cos(angle)
    r.imag = sin(angle)
    return r.view(Rotations2d)

def from_angle_axis(angle,axis):
    angle = .5*asarray(angle)
    axis = normalized(axis)
    assert axis.shape[-1]==3
    v = sin(angle)[...,None]*axis
    q = empty(v.shape[:-1],dtype=Rotations3d.dtype).view(recarray)
    q.s = cos(angle)
    q.v = v
    return q.view(Rotations3d)

def from_sv(*args):
    assert len(args)<=2
    if len(args)==2:
      s,v = map(asarray,args)
    else:
      sv = asarray(args[0])
      s = sv[...,0]
      v = sv[...,1:]
    assert v.shape[-1]==3
    q = empty(broadcast(s[...,None],v).shape[:-1],dtype=Rotations3d.dtype).view(recarray)
    q.s = s
    q.v = v
    return q.view(Rotations3d)

def from_rotated_vector(initial,final):
  initial = normalized(initial)
  final = normalized(final)
  d = initial.shape[-1]
  if d==3:
    cos_theta = clamp(dots(initial,final),-1,1)
    v = cross(initial,final)
    v_norms = magnitudes(v)
    zeros = v_norms==0
    if any(zeros): # Some initial and final vectors are collinear
      zeros = asarray(zeros).reshape(-1)
      v.reshape(-1,d)[zeros] = orthogonal_vector(initial.reshape(-1,d)[zeros])
      v_norms = magnitudes(v)
    sqr_s = .5*(1+cos_theta) # half angle formula
    return from_sv(sqrt(sqr_s),(sqrt(1-sqr_s)/v_norms)[...,None]*v)
  else:
    raise NotImplemented()

from_matrix = rotation_from_matrix

def identity(d):
  if d==2:
    return from_angle(0)
  elif d==3:
    return from_sv(1,(0,0,0))

def random(*shape):
  return from_sv(normalized(numpy.random.randn(*(shape+(4,)))))

geode_wrap._set_rotation_types(Rotations2d,Rotations3d)
