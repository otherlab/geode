#!/usr/bin/env python

from geode import *
from numpy import random
import base64

def test_curry():
  def f(a,b,c,d=0,e=0):
    return a,b,c,d,e
  g = curry(f,1,2,d=4,e=5)
  assert g(3,e=6)==f(1,2,3,d=4,e=6)

def test_base64():
  random.seed(73101)
  for i in xrange(6):
    for n in xrange(20):
      s = random.bytes(n)
      ours = base64_encode(s)
      theirs = base64.b64encode(s)
      assert ours==theirs
      assert base64_decode(ours)==s
  for s in '====','&&&&','++==','+++=':
    try:
      base64_decode(s)
    except ValueError:
      pass

def test_partition_loop():
  for threads in 1,5,14:
    for count in 0,5,7,71:
      partition_loop_test(count,threads)

def test_format():
  format_test()

def test_cache_method():
  class A(object):
    def __init__(self,x):
      self.x = Prop('x',x) 
      self.n = 0
    @cache_method
    def f(self):
      self.n += 1
      return self.x()
  a = A(5)
  b = A(7)
  assert a.n==b.n==0
  for i in xrange(2):
    assert a.f()==5
    assert b.f()==7
    assert a.n==b.n==1
  a.x.set(1)
  assert a.n==b.n==1
  assert a.f()==1
  assert b.f()==7
  assert a.n==2
  assert b.n==1

if __name__=='__main__':
  test_cache_method()
  test_partition_loop()
  test_base64()
  test_curry()
  test_format()
