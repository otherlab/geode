#######################################################################
## Module vector
#######################################################################
##
## Instead of exposing C++ vector functionality directly, we reimplement it in numpy where possible.  This avoids the need for explicit conversions
## and allows the various functions to be used in vectorized form.  For example, cross can be used on two arrays of 3D vectors at a time.
##
#######################################################################
from __future__ import (division,absolute_import)

import platform
import numpy
from numpy import *
if platform.system()=='Windows':
  import geode_all as geode_wrap
  from geode_all import *
else: 
  from .. import geode_wrap
  from ..geode_wrap import *
from .Matrix import Matrix
geode_wrap._set_matrix_type(Matrix)

# Rename some numpy functions to be more like C++
from numpy import (square as sqr, arctan2 as atan2, arcsin as asin, arccos as acos, arctan as atan,
                   arcsinh as asinh, arccosh as acosh, arctanh as atanh)
from numpy.linalg import norm as magnitude

SolidMatrix = {2:SolidMatrix2d,3:SolidMatrix3d}

class ConvergenceError(RuntimeError):
  def __init__(self,s,x):
    RuntimeError.__init__(self,s)
    self.x = x

def V(*args):
  "convenience constructor for vectors: V(1,2,3) is shorter than array([1,2,3])"
  return array(args)

def cube(x):
  return x*x*x

def dots(u,v):
  return multiply(u,v).sum(-1)

def clamp(x,xmin,xmax):
  return minimum(xmax,maximum(xmin,x))

def sqr_magnitude(v):
  return vdot(v,v)

def magnitudes(v,axis=-1):
  "same as magnitude for 1D vectors, but returns array of magnitudes for arrays of vectors"
  return sqrt(sqr(v).sum(axis))

def sqr_magnitudes(v,axis=-1):
  "same as sqr_magnitude for 1D vectors, but returns array of magnitudes for arrays of vectors"
  return sqr(v).sum(axis)

def axis_vector(axis,d=3,dtype=real):
  v = zeros(d,dtype)
  v[axis] = 1
  return v

def magnitudes_and_normalized(v):
  "returns magnitudes(v),normalized(v), but doesn't compute magnitudes twice"
  mags = magnitudes(v)
  zeros = mags==0
  result = v/(mags+zeros)[...,None]
  if any(zeros):
    fallback = axis_vector(0,v.shape[-1])
    if isinstance(zeros,ndarray):
      result[zeros] = fallback
    else:
      result = fallback
  return mags,result

def normalized(v):
  "normalizes a vector or array of vectors"
  return magnitudes_and_normalized(v)[1]

def projected_orthogonal_to_unit_direction(self,direction):
  direction = asarray(direction)
  return self-dots(self,direction)[...,None]*direction

def projected_on_unit_direction(self,direction):
  direction = asarray(direction)
  return dots(self,direction)[...,None]*direction

def projected(self,direction):
  direction = asarray(direction)
  return (dots(self,direction)/sqr_magnitudes(direction))[...,None]*direction

def orthogonal_vector(v):
  v = asarray(v)
  i = absolute(v).argmax(axis=-1)
  n = v.shape[-1]
  j = (i+1)%n
  o = zeros_like(v)
  o_ = o.reshape(-1,n)
  v_ = v.reshape(-1,n)
  k = arange(len(o_))
  o_[k,i] =  v_[k,j]
  o_[k,j] = -v_[k,i]
  return o

def unit_orthogonal_vector(v):
  return normalized(orthogonal_vector(v))

def det(*args):
  return linalg.det(vstack(args))

def cross(u,v):
  # The numpy version doesn't always broadcast correctly, so we roll our own cross product routine.
  # Unfortunately, it's impossible to make 1D/2D cross products work correctly together with
  # broadcasting, so we require either both 2D or both 3D.
  u,v = asarray(u),asarray(v)
  d = u.shape[-1]
  assert d==v.shape[-1]
  if d==3:
    uv = empty(broadcast(u,v).shape,u.dtype)
    u0,u1,u2 = u[...,0],u[...,1],u[...,2]
    v0,v1,v2 = v[...,0],v[...,1],v[...,2]
    uv[...,0] = u1*v2-u2*v1
    uv[...,1] = u2*v0-u0*v2
    uv[...,2] = u0*v1-u1*v0
    return uv
  elif d==2:
    return u[...,0]*v[...,1]-u[...,1]*v[...,0]
  raise ValueError('expected 2D or 3D vectors')

def angle_between(u,v):
  u = asarray(u)
  v = asarray(v)
  c = cross(u,v)
  if u.shape[-1]!=2:
    c = magnitudes(c)
  return atan2(c,dots(u,v))

def signed_angle_between(u,v,n):
  c = cross(u,v)
  s = sign(dots(c,n))
  return s*angle_between(u,v)

def angle(v):
  v = asarray(v)
  if iscomplexobj(v):
    return numpy.angle(v)
  assert v.shape[-1]==2
  return atan2(v[...,1],v[...,0])

def polar(a):
  a = asarray(a)
  v = empty(a.shape+(2,))
  v[...,0] = cos(a)
  v[...,1] = sin(a)
  return v

def maxabs(x):
  x = asarray(x)
  return absolute(x).max() if x.size else 0

def minmag(array,axis=-1):
  s = array.shape
  a = rollaxis(array,axis,len(s)).reshape(-1,s[axis])
  a = a[arange(a.shape[0]),abs(a).argmin(axis=-1)]
  return a.reshape(s[:axis]+s[axis:][1:])

def maxmag(array,axis=-1):
  s = array.shape
  a = rollaxis(array,axis,len(s)).reshape(-1,s[axis])
  a = a[arange(a.shape[0]),abs(a).argmax(axis=-1)]
  return a.reshape(s[:axis]+s[axis:][1:])

def relative_error(a,b,absolute=1e-30):
  return maxabs(a-b)/max(maxabs(a),maxabs(b),absolute)

def rotate_right_90(v):
  v = asarray(v)
  assert v.shape[-1]==2
  rv = empty_like(v)
  rv[...,0] = v[...,1]
  rv[...,1] = -v[...,0]
  return rv

def rotate_left_90(v):
  v = asarray(v)
  assert v.shape[-1]==2
  rv = empty_like(v)
  rv[...,0] = -v[...,1]
  rv[...,1] = v[...,0]
  return rv

def ahash(*args):
  '''Hash the raw binary data in a numpy array'''
  return hash(''.join(asarray(a).tostring() for a in args))

def homogeneous_times(A,b):
  y = Matrix(A[...,:-1])*b+A[...,-1]
  return asarray(y[...,:-1]/y[...,-1,None])

def compact_str(v):
  '''Stringify a numpy array with no whitespace'''
  def s(a):
    try:
      return '[%s]'%','.join(map(s,a))
    except TypeError:
      return str(a)
  return s(v)

from . import (Rotation,Frame)
from .Frame import Frames
