#!/usr/bin/env python

from __future__ import division
from other.core import *
from other.core.geometry.platonic import *

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

def test_simplex_tree():
  mesh,X = sphere_mesh(4)
  tree = SimplexTree(mesh,X,4)
  rays = 1000
  hits = ray_traversal_test(tree,rays,1e-6)
  print 'rays = %d, hits = %d'%(rays,hits)
  assert hits==642

if __name__=='__main__':
  test_simplex_tree()
