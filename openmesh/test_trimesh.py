#!/usr/bin/env python

from other.core.vector import *
from other.core.openmesh import *

def test_trimesh():
  # make sure we can create one and then check its exposed functions
  
  trimesh = TriMesh();
  
  vh0 = trimesh.add_vertex((0., 0., 0.))
  assert vh0 == 0
  vh1 = trimesh.add_vertex((1., 0., 0.))
  assert vh1 == 1
  vh2 = trimesh.add_vertex((0., 1., 0.))
  assert vh2 == 2

  assert trimesh.n_vertices() == 3

  fh = trimesh.add_face(vh0, vh1, vh2)
  assert fh == 0

  assert trimesh.n_faces() == 1
