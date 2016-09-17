#!/usr/bin/env python

from __future__ import division,print_function
from geode import *

def test_hypot():
  # On some platforms Python.h will try to #define hypot as _hypot
  # Since some platforms define hypot as:
  #   double hypot(double a, double b) { return _hypot(a,b); }
  # As a result calling hypot overflow the stack
  # geode/python/config.h tries to ensure we still have a hypot function but could also break things
  # We just make sure we can call hypot without a stack overflow:
  assert geode_test_hypot(1.,0.) == 1.

if __name__=='__main__':
  test_hypot()
