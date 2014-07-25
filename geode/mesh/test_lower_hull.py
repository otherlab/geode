#!/usr/bin/env python

from __future__ import division

from numpy import *
from geode import TriangleSoup, lower_hull, write_mesh
from geode.geometry.platonic import cube_mesh, icosahedron_mesh, sphere_mesh
from geode.vector import relative_error

def test_lower_hull(filename = None):
  mesh,X = icosahedron_mesh()
  mlh,Xlh = lower_hull(mesh, X, [0.3, 0.3, 1.0], -4., 5./180.*pi, 30./180.*pi)

  if filename is not None:
    write_mesh(filename+'-input.obj', mesh, X);
    write_mesh(filename+'-output.obj', mlh, Xlh);

if __name__ == '__main__':
  test_lower_hull("lower_hull_test")
