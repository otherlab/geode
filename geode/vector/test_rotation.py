#!/usr/bin/env python

from numpy import *
from geode.vector import *

def test_2d():
  x = array([1,1])
  r1 = Rotation.from_angle(pi/4)
  r2 = Rotation.from_angle([pi/4,3*pi/4])
  x1 = (r1*r2)*x
  x2 = r1*(r2*x)
  assert allclose(x1,x2)
  assert allclose(x1,[[-1,1],[-1,-1]])
  r1s = rotation_test_2d(r1)
  assert type(r1) is type(r1s)
  assert allclose(r1*r1,r1s)
  assert allclose(r1*x,r1.matrix()*x)
  assert allclose(r1*x,Rotation.from_matrix(r1.matrix())*x)
  assert allclose((r2*r2)*x,rotation_array_test_2d(r2)*x)
  assert type(r2[0])==type(r2)
  assert all((r1==r2)==[1,0])
  assert allclose(r1.inverse()*r1*x,x)

def test_3d():
  x = array([1,0,0])
  r1 = Rotation.from_angle_axis(pi/3,(0,1,0))
  r2 = Rotation.from_angle_axis([pi/4,pi/7],(0,0,1))
  x1 = (r1*r2)*x
  x2 = r1*(r2*x)
  assert allclose(x1,x2)
  assert allclose(-x,(r1*r1*r1)*x)
  r1s = rotation_test_3d(r1)
  assert type(r1) is type(r1s)
  assert allclose(r1*x,r1.matrix()*x)
  assert allclose(r1*x,Rotation.from_matrix(r1.matrix())*x)
  assert allclose((r2*r2)*x,rotation_array_test_3d(r2)*x)
  assert type(r2[0])==type(r2)
  r3 = r2.copy()
  r3[0] = r1
  assert all((r1==r3)==[1,0])
  assert allclose(r1.inverse()*r1*x,x)
  # Test fields
  assert type(r1.s) is ndarray 
  assert type(r2.v) is ndarray 
  assert all(normalized(r2.v)==(0,0,1))
  r2.v = (0,1,0)
  assert all(r2.v==(0,1,0))
  r2.v[:] = (1,0,0)
  assert all(r2.v==(1,0,0))
  theta = .2*random.randn(3,3)
  assert allclose(Rotation.from_euler_angles(theta).euler_angles(),theta)

def test_from_3d():
  random.seed(12313)
  x = random.randn(3)
  y = random.randn(3)
  r = Rotation.from_rotated_vector(x,y)
  assert allclose(normalized(r*x),normalized(y))
