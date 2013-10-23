#!/usr/bin/env python

from __future__ import absolute_import

from numpy import *
from other.core import real
from other.core.force import *
from other.core.force.force_test import *
from other.core.geometry.platonic import *

def test_gravity():
  random.seed(12871)
  X = random.randn(1,3)
  gravity = Gravity([1])
  force_test(gravity,X,verbose=1)

def test_ether_drag():
  random.seed(12871)
  X = random.randn(1,3)
  drag = EtherDrag([1.2],7)
  force_test(drag,X,verbose=1)

def test_springs():
  random.seed(12871)
  X0 = random.randn(2,3)
  X = random.randn(2,3)
  springs = Springs([[0,1]],[1.1,1.2],X0,5,7)
  force_test(springs,X,verbose=1)
  springs = Springs([[0,1]],[1.1,1.2],X0,[5],[7])
  force_test(springs,X,verbose=1)

def test_fvm_2d():
  random.seed(12872)
  model = neo_hookean()
  X = random.randn(3,2)
  dX = .1*random.randn(3,2)
  fvm = finite_volume([(0,1,2)],1000,X,model)
  force_test(fvm,X+dX,verbose=1)

def test_fvm_s3d():
  random.seed(12872)
  model = neo_hookean()
  X = random.randn(3,3)
  dX = .1*random.randn(3,3)
  fvm = finite_volume([(0,1,2)],1000,X,model)
  force_test(fvm,X+dX,verbose=1)

def test_fvm_3d():
  random.seed(12873)
  model = neo_hookean()
  X = random.randn(4,3)
  dX = .1*random.randn(4,3)
  fvm = finite_volume([(0,1,2,3)],1000,X,model)
  force_test(fvm,X+dX,verbose=1)

def test_simple_shell():
  for i in 0,1,3,4,7:
    print '\ni = %d'%i
    random.seed(12872+i)
    X = random.randn(3,2)
    X2 = .1*random.randn(3,3)
    X2[:,:2] += X
    shell = simple_shell([(0,1,2)],1000,X=X,stretch=(7,6),shear=3)
    shell.F_threshold = 1e-7
    force_test(shell,X2,verbose=1)

def test_bending():
  random.seed(7218414)
  stiffness = 7
  damping = 3
  for d in 2,3:
    mesh = SegmentMesh([[0,1],[1,2]]) if d==2 else TriangleSoup([[0,2,1],[1,2,3]])
    X = random.randn(d+1,d)
    dX = .1*random.randn(d+1,d)
    for bend in linear_bending_elements(mesh,X,stiffness,damping),cubic_hinges(mesh,X,stiffness,damping):
      print '\n',type(bend).__name__
      if 'CubicHinges' in type(bend).__name__: # Compare energy with slow_energy
        angles = bend.angles(mesh.bending_tuples(),X)
        for theta in random.randn(20):
          iX = concatenate([X[:-1],[X[1]+(Rotation.from_angle(theta) if d==2 else Rotation.from_angle_axis(theta,X[2]-X[1]))*(X[-1]-X[1])]])
          bend.update_position(iX,False)
          energy = bend.elastic_energy()
          slow_energy = bend.slow_elastic_energy(angles,X,iX)
          error = relative_error(energy,slow_energy)
          print 'slow energy error = %g (slow %g vs. %g, theta = %g)'%(error,slow_energy,energy,theta)
          assert error<1e-8
      force_test(bend,X+dX,verbose=1)
  # Test against exact sphere energies.  We don't actually compute the correct answers in 3D, since hinge based energies are fundamentally mesh dependent.
  radius = 78
  analytics = [('S_1',circle_mesh(1000,radius=radius),pi/radius,1,1e-5),
               ('S_2',sphere_mesh(3,radius=radius),2*pi,4,.015),
               ('S_1 x [0,e]',open_cylinder_mesh((0,0,0),(0,0,e),radius=radius,na=100),pi*e/radius,3,2e-4)]
  for name,(mesh,X),known,fudge,tolerance in analytics:
    flat = zeros(len(mesh.bending_tuples()))
    bend = cubic_hinges(mesh,X,stiffness,damping,angles=flat)
    energy = bend.slow_elastic_energy(flat,X,X)/stiffness/fudge
    error = relative_error(energy,known)
    print '%s: known %g, energy %g, ratio %r, error %g'%(name,known,energy,energy/known,error)
    assert error<tolerance

def test_linear_fvm_2d():
  random.seed(12872)
  X = random.randn(3,2)
  dX = .1*random.randn(3,2)
  fvm = linear_finite_volume([(0,1,2)],X,1000)
  force_test(fvm,X+dX,verbose=1)

def test_linear_fvm_s3d():
  random.seed(12872)
  X = random.randn(3,3)
  dX = .1*random.randn(3,3)
  fvm = linear_finite_volume([(0,1,2)],X,1000)
  force_test(fvm,X+dX,verbose=1)

def test_linear_fvm_3d():
  random.seed(12873)
  X = random.randn(4,3)
  dX = .1*random.randn(4,3)
  fvm = linear_finite_volume([(0,1,2,3)],X,1000)
  force_test(fvm,X+dX,verbose=1)

def test_linear_fvm_hex():
  random.seed(12873)
  X = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]+.1*random.randn(8,3)
  dX = .1*random.randn(8,3)
  fvm = linear_finite_volume([arange(8)],X,1000)
  force_test(fvm,X+dX,verbose=1)

def test_air_pressure():
  random.seed(2813)
  mesh,X = icosahedron_mesh()
  if 1:
    X = vstack([random.randn(3),X,random.randn(3)])
    mesh = TriangleSoup(mesh.elements+1)
  X2 = X + random.randn(*X.shape)/10
  for closed in 0,1:
    for side in 1,-1:
      print '\nclosed %d, side %d'%(closed,side)
      air = AirPressure(mesh,X,closed,side)
      force_test(air,X2,verbose=1)

def test_pins():
  random.seed(17310)
  X = random.randn(5,3)
  nodes = [1,3]
  mass = random.randn(len(X))**2
  targets = random.randn(len(nodes),3)
  pins = Pins(nodes,mass,targets,3.5,1.1)
  force_test(pins,X,verbose=1)

def test_axis_pins():
  random.seed(17310)
  X = random.randn(5,3)
  nodes = [1,3]
  mass = random.randn(len(X))**2
  targets = random.randn(len(nodes),3)
  pins = AxisPins(nodes,mass,targets,3.5,1.1)
  pins.axis = (2,3,4)
  force_test(pins,X,verbose=1)

def test_surface_pins():
  random.seed(127130)
  # Test a sphere mesh ignoring errors in the Hessian
  target_mesh,target_X = sphere_mesh(0)
  X = random.randn(100,3)
  mass = absolute(random.randn(len(X)))
  nodes = unique(random.randint(len(X),size=50).astype(int32))
  pins = SurfacePins(nodes,mass,target_mesh,target_X,7,1.1)
  force_test(pins,X,verbose=1,ignore_hessian=1)
  # Hessian should be correct if the closest points are all on faces
  target_mesh = TriangleSoup([[0,1,2]])
  target_X = 100*eye(3)
  pins = SurfacePins(nodes,mass,target_mesh,target_X,7,1.1)
  force_test(pins,X,verbose=1)

def test_binding_springs():
  random.seed(73210)
  for k in 2,3:
    nodes = [1,2]
    X = random.randn(20,3)
    parents = {2:[[5,6],[8,9]],
               3:[[5,6,7],[10,11,12]]}[k]
    weights = random.uniform(1,2,(2,k))
    weights /= weights.sum(axis=1).reshape(-1,1)
    mass = absolute(random.randn(len(X)))
    springs = binding_springs(nodes,parents,weights,mass,7,1.2)
    force_test(springs,X,verbose=1)

def test_particle_binding_springs():
  random.seed(73210)
  X = random.randn(4,3)
  mass = absolute(random.randn(len(X)))
  springs = particle_binding_springs([[1,2]],mass,7,1.2)
  force_test(springs,X,verbose=1)

if __name__=='__main__':
  test_simple_shell()
