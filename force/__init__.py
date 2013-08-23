from __future__ import division,absolute_import
from other.core import *

def edge_springs(mesh,mass,X,stiffness,damping_ratio):
  return Springs(mesh.segment_mesh().elements,mass,X,stiffness,damping_ratio)

def bending_springs(mesh,mass,X,stiffness,damping_ratio):
  springs = ascontiguousarray(mesh.bending_quadruples()[:,(0,3)])
  return Springs(springs,mass,X,stiffness,damping_ratio)

StrainMeasure = {2:StrainMeasure2d,3:StrainMeasure3d}
FiniteVolume = {(2,2):FiniteVolume2d,(3,2):FiniteVolumeS3d,(3,3):FiniteVolume3d}
LinearFiniteVolume = {(2,2):LinearFiniteVolume2d,(3,2):LinearFiniteVolumeS3d,(3,3):LinearFiniteVolume3d}

def finite_volume(mesh,density,X,model,m=None,plasticity=None,verbose=True):
  elements = mesh.elements if isinstance(mesh,Object) else asarray(mesh,dtype=int32)
  mx,d = asarray(X).shape[1],elements.shape[1]-1
  if m is None:
    m = mx
  strain = StrainMeasure[d](elements,X)
  if verbose:
    strain.print_altitude_statistics()
  if isinstance(model,dict):
    model = model[d]
  return FiniteVolume[m,d](strain,density,model,plasticity)

def linear_finite_volume(mesh,X,density,youngs_modulus=3e6,poissons_ratio=.4,rayleigh_coefficient=.05):
  elements = mesh.elements if isinstance(mesh,Object) else asarray(mesh,dtype=int32)
  m,d = asarray(X).shape[1],elements.shape[1]-1
  if d==7:
    return LinearFiniteVolumeHex(StrainMeasureHex(elements,X),density,youngs_modulus,poissons_ratio,rayleigh_coefficient)
  else:
    return LinearFiniteVolume[m,d](elements,X,density,youngs_modulus,poissons_ratio,rayleigh_coefficient)

def neo_hookean(youngs_modulus=3e6,poissons_ratio=.475,rayleigh_coefficient=.05,failure_threshold=.25):
  return {2:NeoHookean2d(youngs_modulus,poissons_ratio,rayleigh_coefficient,failure_threshold),
          3:NeoHookean3d(youngs_modulus,poissons_ratio,rayleigh_coefficient,failure_threshold)}

def simple_shell(mesh,density,Dm=None,X=None,stretch=(0,0),shear=0):
  mesh = mesh if isinstance(mesh,Object) else TriangleMesh(asarray(mesh,dtype=int32))
  if Dm is None:
    X = asarray(X)
    assert X.ndim==2 and X.shape[1]==2, 'Expected 2D rest state'
    tris = mesh.elements
    Dm = X[tris[:,1:]].swapaxes(1,2)-X[tris[:,0]].reshape(-1,2,1)
  else:
    assert X is None
  shell = SimpleShell(mesh,ascontiguousarray(Dm),density)
  shell.stretch_stiffness = stretch
  shell.shear_stiffness = shear
  return shell

LinearBendingElements = {2:LinearBendingElements2d,3:LinearBendingElements3d}
def linear_bending_elements(mesh,X,stiffness,damping):
  X = asarray(X)
  bend = LinearBendingElements[X.shape[1]](mesh,X)
  bend.stiffness = stiffness
  bend.damping = damping
  return bend

CubicHinges = {2:CubicHinges2d,3:CubicHinges3d}
def cubic_hinges(mesh,X,stiffness,damping,angles=None):
  bends = mesh.bending_tuples()
  X = asarray(X)
  Hinges = CubicHinges[X.shape[1]]
  if angles is None:
    angles = Hinges.angles(bends,X)
  hinges = CubicHinges[X.shape[1]](bends,angles,X)
  hinges.stiffness = stiffness
  hinges.damping = damping
  return hinges

BindingSprings = {2:BindingSprings2d,3:BindingSprings3d}
def binding_springs(nodes,parents,weights,mass,stiffness,damping_ratio):
  parents = asarray(parents,dtype=int32)
  return BindingSprings[parents.shape[1]](nodes,parents,weights,mass,stiffness,damping_ratio)

particle_binding_springs = ParticleBindingSprings
edge_binding_springs = BindingSprings2d
face_binding_springs = BindingSprings3d
