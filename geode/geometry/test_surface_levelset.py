#!/usr/bin/env python

from __future__ import division
from geode import *
from geode.geometry.platonic import *
from numpy import random

def test_surface_levelset(): 
  random.seed(127130)
  mesh,X = sphere_mesh(4)
  surface = SimplexTree(mesh,X,10)
  particles = ParticleTree(random.randn(1000,3),10)
  print 'fast'
  phi,normal,triangles,weights = evaluate_surface_levelset(particles,surface,10,True)
  mags,Xn = magnitudes_and_normalized(particles.X)
  print 'phi range %g %g'%(phi.min(),phi.max())
  # Compare with sphere distances
  phi3 = mags-1
  normal3 = Xn
  assert absolute(phi-phi3).max() < .002
  assert maxabs(absolute(dots(normal,normal3))-1) < .001
  normal_error = (phi>1e-3)*magnitudes(normal-normal3)
  if 1:
    i = argmax(normal_error)
    print 'i %d, X %s, phi %g, phi3 %g, normal %s, normal3 %s'%(i,particles.X[i],phi[i],phi3[i],normal[i],normal3[i])
  e = max(normal_error)
  print 'normal error =',e
  assert e < .04
  # Check weights
  closest = particles.X-phi.reshape(-1,1)*normal
  assert maxabs(magnitudes(closest)-1) < .002
  closest2 = (weights.reshape(-1,3,1)*X[mesh.elements[triangles]]).sum(axis=1)
  assert relative_error(closest,closest2) < 1e-7
  # Compare with slow mesh distances
  print 'slow'
  phi2,normal2,_,_ = slow_evaluate_surface_levelset(particles,surface)
  if 0:
    i = argmax(abs(abs(phi)-phi2))
    print 'i %d, phi %g, phi2 %g'%(i,phi[i],phi2[i])
  assert relative_error(abs(phi),phi2) < 1e-7
  assert all(magnitudes(cross(normal,normal2))<1e-7)
