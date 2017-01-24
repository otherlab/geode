#!/usr/bin/env python

from __future__ import absolute_import

from numpy import *
from geode import *
import cPickle as pickle
import sys
import py
import unittest

@unittest.skip("module import bugs")
def test_basic():
  a = empty_array()
  assert a.base is None
  assert all(a==array([]))
  assert a.base is None
  b = array_test(array(xrange(5),dtype=int32),-1)
  assert len(b)==5

def test_resize():
  a = array_test(empty_array(),20)
  print type(a.base)
  assert len(a)==20
  assert base_refcnt(a)==1 # Can't use sys.getrefcount(a.base) since it's version dependent
  a = array_test(a,10)
  assert base_refcnt(a)==1
  str(a.base)

def test_sharing():
  a = zeros(5,dtype=int32)
  b = array_test(a,-1)
  assert all(a==b)
  a[3] = 3
  assert b[3]==3

def test_const():
  a = arange(5,dtype=int32)
  b = const_array_test(a)
  assert all(a==b)
  try:
    b[3] = 0
    assert False
  except RuntimeError:
    pass
  except ValueError:
    pass

@unittest.skip("module import bugs")
def test_refcount():
  a = array([],dtype=int32)
  print a.dtype.name
  base = a
  for i in xrange(100):
    a = array_test(a,-1)
    assert a.base is base
    assert sys.getrefcount(a)==2
    assert sys.getrefcount(base)==3

@unittest.skip("module import bugs")
def test_mismatch():
  a = array([1.,2,3])
  py.test.raises(TypeError,array_test,a,-1)

def test_write(filename=None):
  random.seed(1731031)
  data = random.randn(10,7)
  if filename is None:
    file = named_tmpfile(suffix='.npy')
    filename = file.name
  header,size = array_write_test(filename,data)
  assert size==data.nbytes
  data2 = load(filename)
  assert all(data==data2)
  assert header.tostring()==open(filename).read(len(header))

def test_nested():
  nested_test()
  l = [[1,2],[3]]
  a = Nested(l,dtype=int32)
  assert a==nested_convert_test(a)==nested_convert_test(l)
  n = asarray([[1,2,3],[4,5,6]],dtype=int32)
  na = Nested(n)
  print na,nested_convert_test(na),nested_convert_test(n)
  assert na==nested_convert_test(na)==nested_convert_test(n)
  assert a==pickle.loads(pickle.dumps(a))

if __name__=='__main__':
  test_write('array.npy')
