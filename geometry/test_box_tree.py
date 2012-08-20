#!/usr/bin/env python

from __future__ import division

from numpy import *
from other.core.python import real
from other.core.vector import *
from other.core.geometry import *

def test_box_tree():
  random.seed(10098331)
  for n in 0,1,35,99,100,101,199,200,201:
    print
    x = random.randn(n,3).astype(real)
    tree = BoxTree(x,10)
    tree.check(x)

def test_particle_tree():
  random.seed(10098331)
  for n in 0,1,35,99,100,101,199,200,201:
    print
    X = random.randn(n,3).astype(real)
    tree = ParticleTree(X,10)
    tree.check(X)
    X[:] = random.randn(n,3).astype(real)
    tree.update()
    tree.check(X)
