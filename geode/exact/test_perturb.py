#!/usr/bin/env python

from __future__ import division,absolute_import
from geode import *
import random
import numpy

def test_monomials():
  for n in xrange(4):
    assert all(perturb_monomials(0,n)==zeros((1,n)))
    assert all(perturb_monomials(1,n)==concatenate([zeros((1,n)),eye(n)[::-1]]))
  assert all(perturb_monomials(2,2)==[(0,0),(0,1),(1,0),(0,2),(1,1),(2,0)])
  assert all(perturb_monomials(2,3)==[(0,0,0),(0,0,1),(0,1,0),(1,0,0),(0,0,2),(0,1,1),(0,2,0),(1,0,1),(1,1,0),(2,0,0)])

def test_interpolation():
  numpy.random.seed(183181)
  bound = 10000
  for d in xrange(1,5):
    for n in xrange(1,6):
      beta = perturb_monomials(d,n)
      coefs = numpy.random.randint(2*bound,size=len(beta))-bound
      in_place_interpolating_polynomial_test(d,beta,coefs,False)

def test_snap_divs():
  random.seed(83198131)
  bits = 8*dtype(int).itemsize
  bound = 2**53-1
  ratio = dtype('double').itemsize//dtype('uint').itemsize
  def limbs(n,count):
    assert abs(n)<2**(bits*count)
    mask = 2**bits-1
    return [n>>bits*i&mask for i in xrange(count)]
  hi = bound*3//2
  overflow = 0
  imaginary = 0
  for p in 1,2:
    lo = 0 if p==2 else -hi
    for c in 1,2,3,4:
      for i in xrange(10):
        x = random.randrange(lo,hi)
        d = random.getrandbits(bits*c-1)*random.choice([-1,1])
        n = x**p*d + random.randrange(abs(x**(p-1)*d//(3-p)))*random.choice([-1,1])
        if p==2 and i<2:
          n = -n
        try:
          count = c+ratio*p+random.randrange(3)
          values = asarray([limbs(n,count),limbs(d,count)],dtype='uint')
          y = snap_divs_test(values,p==2)
          assert abs(x)<bound
          assert x==y
        except OverflowError:
          assert abs(x)>bound
          overflow += 1
        except RuntimeError:
          assert p==2 and n//d<0
          imaginary += 1
  assert overflow>20
  assert imaginary==8

def test_perturbed_sign():
  perturbed_sign_test_1()
  perturbed_sign_test_2()
  perturbed_sign_test_3()

def test_perturbed_ratio():
  perturbed_ratio_test()

if __name__=='__main__':
  test_monomials()
  test_interpolation()
  test_perturbed_sign()
  test_snap_divs()
  test_perturbed_ratio()
