#!/usr/bin/env python

from other.core.utility import *

def test_curry():
  def f(a,b,c,d=0,e=0):
    return a,b,c,d,e
  g = curry(f,1,2,d=4,e=5)
  assert g(3,e=6)==f(1,2,3,d=4,e=6)

if __name__=='__main__':
  test_curry()
