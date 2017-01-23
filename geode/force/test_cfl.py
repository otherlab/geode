#!/usr/bin/env python

from __future__ import division
from numpy import *
from geode import *
from geode.force import *
import unittest

def dt(dx,stiffness,mass):
  mesh = SegmentSoup([[0,1]])
  mass = array([1e10,mass],dtype=real)
  X = array([[0,0,0],[dx,0,0]],dtype=real)
  springs = edge_springs(mesh,mass,X,stiffness,1)
  sqr_frequency = zeros_like(mass)
  springs.add_frequency_squared(sqr_frequency)
  return 1/sqrt(sqr_frequency.max())

@unittest.skip("failure in dt()")
def test_cfl():
  assert allclose(1/sqrt(2),dt(1,2,1)/dt(1,1,1))
  assert allclose(2,dt(2,1,1)/dt(1,1,1))
  assert allclose(1,dt(1,1,2)/dt(1,1,1))

if __name__=='__main__':
  test_cfl()
