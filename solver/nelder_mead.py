"""Nelder Mead simplex optimization method

Implemented using coroutines (generators) to allow integration with a false time integration loop.
We assume function evaluations are extremely slow, so we can afford to be very lazy otherwise.
See http://en.wikipedia.org/wiki/Nelder-Mead_method for details."""

from __future__ import division
from numpy import *
from other.core.vector import *
from other.core.utility import Log

def optimize_generator(x0,step0,tolerance,verbose=False):
  # Initialize simplex 
  d = len(x0)
  x = empty((d+1,d))
  x[:] = asarray(x0).reshape(1,d)
  for i in xrange(d):
    x[i+1,i] += step0
  f = empty(d+1)
  for i in xrange(d+1):
    f[i] = yield x[i],False
    if verbose:
      Log.write('nelder-mead: initialized f(%s) = %g'%(x[i],f[i]))

  # Control parameters
  alpha = 1.
  gamma = 2.
  rho = .5
  sigma = .5

  # Loop until convergence
  while 1:
    # Sort vertices in increasing order of f
    p = sorted(xrange(d+1),key=lambda i:f[i])
    f = f[p]
    x = x[p]

    if verbose:
      Log.write('nelder-mead: best  x = %s, f(x) = %g'%(x[0],f[0]))
      Log.write('nelder-mead: worst x = %s, f(x) = %g'%(x[-1],f[-1]))

    # Check if we're converged 
    diameter = max(magnitude(x[i]-x[j]) for i in xrange(d+1) for j in xrange(i+1,d+1))
    if verbose:
      Log.write('nelder-mead: diameter = %g'%diameter)
    if diameter <= tolerance:
      yield x[0],True
      return

    def replace(fn,xn):
      f[-1] = fn
      x[-1] = xn

    # Perform reflection
    xm = x[:d].mean(axis=0)
    xr = xm + alpha*(xm-x[-1])
    fr = yield xr,False
    if f[0] <= fr <= f[-2]: # Accept reflection
      if verbose:
        Log.write('nelder-mead: reflected')
      replace(fr,xr)
    elif fr <= f[0]: # other.core expansion
      xe = xm + gamma*(xm-x[-1])
      fe = yield xe,False
      if fe < fr:
        if verbose:
          Log.write('nelder-mead: expansion succeeded')
        replace(fe,xe)
      else:
        if verbose:
          Log.write('nelder-mead: expansion failed')
        replace(fr,xr)
    else: # other.core contraction
      xc = x[-1] + rho*(xm-x[-1])
      fc = yield xc,False
      if fc < f[-1]:
        if verbose:
          Log.write('nelder-mead: contracted')
        replace(fc,xc)
      else: # All else failed; perform reduction
        if verbose:
          Log.write('nelder-mead: reduced')
        for i in xrange(1,d+1):
          x[i] = x[0] + sigma*(x[i]-x[0])
          f[i] = yield x[i],False

def optimize(f,x0,step0,tolerance,verbose=False):
  gen = optimize_generator(x0,step0,tolerance,verbose)
  fx = None
  while 1:
    x,done = gen.send(fx)
    if done:
      if verbose:
        print 'converged: x = %s'%x
      return x
    else:
      fx = f(x)
      if verbose:
        print 'f(%s) = %g'%(x,fx)
