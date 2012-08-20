#!/usr/bin/env python

from numpy import *
from other.core.vector import *

def test_frame_2d():
    random.seed(18312)
    r1 = Rotation.from_angle(pi/3)
    r2 = Rotation.from_angle(pi/7)
    f1 = Frames(random.randn(2),r1)
    f2 = Frames(random.randn(5,2),r2)
    x = random.randn(2)
    y = f1*(f2*x)
    assert allclose(y,(f1*f2)*x)
    yc = frame_array_test_2d(f1,f2,x)
    assert allclose(yc[0]*x,frame_test_2d(f1,f2[0],x)*x)
    assert allclose((f1*f2)*(x+x),yc*x)
    assert allclose((f2.matrix()*hstack([x,1]))[:,:-1],f2*x)
    f3 = f2.copy()
    f3[:2] = f1
    assert all((f1==f3)==[1,1,0,0,0])
    assert allclose(f1.inverse()*f1*x,x)

def test_frame_3d():
    random.seed(18312)
    r1 = Rotation.from_angle_axis(pi/3,(1,2,3))
    r2 = Rotation.from_angle_axis(pi/7,(4,2,3))
    f1 = Frames(random.randn(3),r1)
    f2 = Frames(random.randn(5,3),r2)
    x = random.randn(3)
    y = f1*(f2*x)
    assert allclose(y,(f1*f2)*x)
    yc = frame_array_test_3d(f1,f2,x)
    assert allclose(yc[0]*x,frame_test_3d(f1,f2[0],x)*x)
    assert allclose((f1*f2)*(x+x),yc*x)
    assert allclose((f2.matrix()*hstack([x,1]))[:,:-1],f2*x)
    f3 = f2.copy()
    f3[:2] = f1
    assert all((f1==f3)==[1,1,0,0,0])
    assert allclose(f1.inverse()*f1*x,x)

if __name__=='__main__':
  test_frame_2d()
  test_frame_3d()
