#!/usr/bin/env python

from __future__ import division
from other.core.geometry.Platonic import *
from other.core.vector import *

def test_icosahedron():
    mesh,X = icosahedron_mesh()
    a = 2 # see http://en.wikipedia.org/wiki/Icosahedron
    assert relative_error(mesh.surface_area(X),5*sqrt(3)*a**2) < 1e-5
    assert relative_error(mesh.volume(X),5/12*(3+sqrt(5))*a**3) < 1e-5
    sphere = sphere_mesh(3)

def test_tetrahedron():
    mesh,X = tetrahedron_mesh()
    a = sqrt(8) # see http://en.wikipedia.org/wiki/Tetrahedron
    assert relative_error(mesh.surface_area(X),sqrt(3)*a**2) < 1e-5
    print mesh.volume(X),sqrt(2)/12*a**3
    assert relative_error(mesh.volume(X),sqrt(2)/12*a**3) < 1e-5

def test_sphere():
    mesh,X = sphere_mesh(4)
    assert relative_error(magnitudes(X),1)<1e-7
    centers = normalized(X[mesh.elements].mean(axis=1))
    assert relative_error(dots(centers,mesh.element_normals(X)),1)<2e-5

if __name__=='__main__':
    test_icosahedron()
    test_tetrahedron()
    test_sphere()
