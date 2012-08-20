#!/usr/bin/env python

from numpy import *
from other.core.vector import *

def test_basic():
  v=vector_test((1,2,3))
  assert v.base is None
  assert all(v==(1,2,3))
  vector_stream_test()

def test_misc():
  v=V(1,3,7)
  assert list(ones(4))==[1,1,1,1]
  assert list(axis_vector(2,4))==[0,0,1,0]
  assert all(projected_orthogonal_to_unit_direction(v,(0,1,0))==(1,0,7))
  assert all(projected_on_unit_direction(v,(0,1,0))==(0,3,0))
  assert all(projected(v,(0,2,0))==(0,3,0))
  assert det((2,0,0),(0,0,3),(0,-1,0))==6
  assert abs(angle_between((1,0,0),(1,1,0))-pi/4)<1e-6
  t = arange(0,pi,.1)
  a = (2*cos(t),2*sin(t),0)
  b = (-sin(t),cos(t),0)
  n = (0,0,-1)*ones_like(a)
  print a,b,n
  assert (abs(signed_angle_between(a,b,n)+pi/2)<1e-6).all()

def test_magnitude():
  tolerance=1e-6
  v=V(1,2,3)
  mag_squared=sum(a*a for a in v)
  mag=sqrt(mag_squared)
  assert abs(sqr_magnitude(v)-mag_squared)<tolerance
  assert abs(magnitude(v)-mag)<tolerance
  assert magnitude(v/mag-normalized(v))<tolerance

def test_multiple():
  a=array([[1,2,3],[4,5,6],[0,0,0]])
  for i in range(len(a)):
    assert all(dots(a,a-1)[i]==dot(a[i],a[i]-1))
    assert all(magnitudes(a)[i]==magnitude(a[i]))
    assert all(magnitudes_and_normalized(a)[0][i]==magnitude(a[i]))
    assert all(sqr_magnitudes(a)[i]==sqr_magnitude(a[i]))
    assert all(normalized(a)[i]==normalized(a[i]))
    assert all(magnitudes_and_normalized(a)[1][i]==normalized(a[i]))

def test_max():
  random.seed(18)
  x = random.randn(10)
  assert allclose(max_magnitude(x),maxabs(x))
  for d in 1,2,3:
    x = random.randn(10,d)
    assert allclose(max_magnitude(x),magnitudes(x).max())
  try:
    max_magnitude(random.randn(10,4))
    assert False
  except ValueError:
    pass

def test_minmag():
  random.seed(731231)
  x = random.randn(3,3,3,3,3)
  mx = minmag(x,-3)
  assert all(mx==minmag(x,2))
  assert all(abs(mx)==abs(x).min(-3))
