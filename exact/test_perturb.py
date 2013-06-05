#!/usr/bin/env python

from __future__ import division
from other.core import *

def test_monomials():
  for n in xrange(4):
    assert all(perturb_monomials(0,n)==zeros((1,n)))
    assert all(perturb_monomials(1,n)==concatenate([zeros((1,n)),eye(n)[::-1]]))
  assert all(perturb_monomials(2,2)==[(0,0),(0,1),(1,0),(0,2),(1,1),(2,0)])
  assert all(perturb_monomials(2,3)==[(0,0,0),(0,0,1),(0,1,0),(1,0,0),(0,0,2),(0,1,1),(0,2,0),(1,0,1),(1,1,0),(2,0,0)])

def test_interpolation():
  random.seed(183181)
  bound = 10000
  for d in xrange(1,5):
    for n in xrange(1,6):
      beta = perturb_monomials(d,n)
      coefs = random.randint(2*bound,size=len(beta))-bound
      in_place_interpolating_polynomial_test(d,beta,coefs,False)

def test_perturbed_sign():
  perturbed_sign_test_1()
  perturbed_sign_test_2()
  perturbed_sign_test_3()

if __name__=='__main__':
  test_monomials()
  test_interpolation()
  test_perturbed_sign()
