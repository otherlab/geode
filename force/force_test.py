from __future__ import absolute_import,division
from other.core import *

def force_test(force,X,dx_scale=1e-5,tolerance=1e-5,iterations=10,verbose=False,ignore_hessian=False,single_input=None):
  '''Test a force's methods against numerical differentiation'''
  try:
    structure = SolidMatrixStructure(len(X))
    force.structure(structure)
    matrix = SolidMatrix[X.shape[1]](structure)
  except NotImplementedError:
    matrix = None
  def rand(scale):
    U = scale*random.randn(*X.shape).astype(real)
    if single_input is not None:
      U[:single_input] = U[single_input:][1:] = 0
    return U
  def elastic_energy(X):
    force.update_position(X,False)
    return force.elastic_energy()
  def elastic_force(X):
    F = zeros_like(X)
    force.update_position(X,False)
    force.add_elastic_force(F)
    return F
  def elastic_differential(X,dX):
    dF = zeros_like(X)
    force.update_position(X,False)
    force.add_elastic_differential(dF,dX)
    return dF
  def elastic_gradient_block_diagonal_times(X,dX):
    force.update_position(X,False)
    return force.elastic_gradient_block_diagonal_times(dX)
  def damping_energy(X,V):
    force.update_position(X,False)
    return force.damping_energy(V)
  def damping_force(X,V):
    F = zeros_like(X)
    force.update_position(X,False)
    force.add_damping_force(F,V)
    return F
  def elastic_gradient_times(X,dX):
    matrix.zero()
    force.update_position(X,False)
    force.add_elastic_gradient(matrix)
    F = empty_like(X)
    matrix.multiply(dX,F)
    return F
  def damping_gradient_times(X,dV):
    matrix.zero()
    force.update_position(X,False)
    force.add_damping_gradient(matrix)
    F = empty_like(X)
    matrix.multiply(dV,F)
    return F
  V = rand(1)
  U0 = elastic_energy(X)
  Fe0 = elastic_force(X)
  K0 = damping_energy(X,V)
  Fd0 = damping_force(X,V)
  for _ in xrange(iterations):
    # Test elastic force
    dX = rand(dx_scale)
    U2 = elastic_energy(X+dX)
    Fe1 = elastic_force(X+dX/2)
    e = relative_error(tensordot(Fe1,dX,axes=2),U0-U2)
    if verbose:
      print '|Fe1| %g, U0 %g, U2 %g'%(maxabs(Fe1),U0,U2)
      print 'elastic force error =',e
    assert e < tolerance

    # Test elastic differentials
    if not ignore_hessian:
      Fe2 = elastic_force(X+dX)
      dFe = elastic_differential(X+dX/2,dX)
      e = relative_error(dFe,Fe2-Fe0)
      if verbose:
        print '|dFe| %g, |Fe2| %g, |Fe0| %g, |Fe2-Fe0| %g'%(maxabs(dFe),maxabs(Fe2),maxabs(Fe0),maxabs(Fe2-Fe0))
        print 'elastic differential error =',e
      assert e < tolerance

    # Test elastic gradient
    if matrix:
      dF2 = elastic_gradient_times(X,dX)
      dF = elastic_differential(X,dX)
      e = relative_error(dF,dF2)
      if verbose:
        print 'elastic gradient force errors = %s',magnitudes(dF-dF2)
        print 'elastic gradient error =',e
      assert e < tolerance

    # Test elastic gradient block diagonal
    i = random.randint(len(X))
    dXi = zeros_like(dX)
    dXi[i] = dX[i]
    dFi = elastic_differential(X,dXi)
    try:
      dFi2 = elastic_gradient_block_diagonal_times(X,dXi)
      e = relative_error(dFi[i],dFi2[i])
      if verbose:
        print 'elastic gradient block diagonal error =',e
      assert e < tolerance
    except NotImplementedError:
      pass

    # Test damping force
    dV = rand(tolerance)
    K2 = damping_energy(X,V+dV)
    Fd1 = damping_force(X,V+dV/2)
    e = relative_error(tensordot(Fd1,dV,axes=2),K0-K2)
    if verbose:
      print 'damping force error =',e
    assert e < tolerance

    # Test damping linearity
    V2 = rand(1)
    Fd2 = damping_force(X,V2)
    Fd3 = damping_force(X,V+V2)
    e = relative_error(Fd0+Fd2,Fd3)
    if verbose:
      print 'damping linearity error =',e
    assert e < tolerance

    # Test damping gradient
    if matrix:
      Fd2 = damping_gradient_times(X,V)
      e = relative_error(Fd0,Fd2)
      if verbose:
        print 'damping gradient error =',e
      assert e < tolerance
