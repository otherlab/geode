#!/usr/bin/env python

from other.core import Object,ClassTest

def test_field():
  c=ClassTest(Object()) 
  c.field=2
  assert c.field==2
  assert c.static_const==17

def test_ref():
  c=ClassTest(Object()) 
  assert isinstance(c.ref,Object)
  o=Object()
  c.ref=o
  assert c.ref is o
  try:
    c.ref=None
    assert False
  except TypeError:
    pass

def test_ptr():
  c=ClassTest(Object()) 
  o=Object()
  assert c.ptr is None
  c.ptr=o
  print c.ptr
  assert c.ptr is o
  c.ptr=None
  assert c.ptr is None

def test_ref2():
  c=ClassTest(Object())
  c.ref2={}
  assert isinstance(c.ref2,dict)
  c.ref2=None
  assert c.ref2 is None

def test_ptr2():
  c=ClassTest(Object()) 
  assert c.ptr2 is None
  c.ptr2={}
  assert isinstance(c.ptr2,dict)
  c.ptr2=None
  assert c.ptr2 is None

def test_methods():
  c=ClassTest(Object()) 
  assert c.normal(1)==2
  assert c.virtual_(2)==6
  assert c.static_(3)==15
  assert c(6)==24

def test_prop():
  c=ClassTest(Object()) 
  assert c.prop==17
  c.data = 78
  assert c.data==78
  try:
    c.prop = 23
    assert False
  except AttributeError:
    pass

def test_inherit():
  class A(ClassTest):
    def __init__(self,o):
      d = self.__dict__
      self.x = 5
      self.y = o
  a = A(Object())
  assert a.x==5
  assert a.prop==17
  assert a.ref is a.y

if __name__=='__main__':
  test_inherit()
  test_field()
