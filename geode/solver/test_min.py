#!/usr/bin/env python

from __future__ import division
from geode import *
from geode.solver import nelder_mead
from numpy import *

def test_brent():
  def f(x):
    return x**3*(x-4)
  tol = 1e-7
  x,fx,it = brent(f,(-2,1,4),tol,100)
  assert abs(x-3)<tol
  assert abs(fx+27)<tol

def test_bracket():
  random.seed(8523815)
  for _ in xrange(20):
    co = 5*random.randn(4)
    def f(x):
      return co[0]+x*(co[1]+x*(co[2]+x*(co[3]+x)))
    (a,b,c),(fa,fb,fc) = bracket(f,0,.1*random.randn())
    assert a<b<c
    assert fb<min(fa,fc)
    assert allclose(fa,f(a))
    assert allclose(fb,f(b))
    assert allclose(fc,f(c))

def test_powell():
  evals = [0]
  for tweak in 0,1/2:
    def f(x):
      evals[0] += 1
      x,y = x
      f = x*x+2*y*y+x-3*y+3 + tweak*sin(5*x)*sin(5*y)
      #print 'f(%g,%g) = %g'%(x,y,f)
      return f
    x = zeros(2)
    tol = 1e-4
    fx,i = powell(f,x,.1,tol,tol,100)
    print 'x = %s, fx = %g, iters = %d, evals = %d'%(x,fx,i,evals[0])
    xc = (-0.87892353,0.89360935) if tweak else (-.5,.75)
    fc = 1.34897 if tweak else 13/8
    assert maxabs(x-xc)<2*tol
    assert abs(fx-fc)<tol

def test_nelder_mead():
  def f((x,y)):
    return abs((3-2*x)*x-2*y+1)**(7/3) + abs((3-2*y)*y-x+1)**(7/3)
  x = nelder_mead.optimize(f,(-.9,-1),.3,1e-5,verbose=1)
  assert f(x) < 1e-9

if __name__=='__main__':
  test_powell()
  test_bracket()
  test_brent()
