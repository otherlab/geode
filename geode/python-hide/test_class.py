#!/usr/bin/env python

from geode import Object,ClassTest,ClassTest2

def test_extra():
  # Extra arguments should throw exceptions
  try:
    ClassTest(Object(),4)
    assert False
  except TypeError:
    pass

def test_field():
  c = ClassTest(Object())
  c.field = 2
  assert c.field==2
  assert c.static_const==17

def test_attr():
  c = ClassTest(Object())
  assert c.attr==8
  c.attr = 9
  assert c.attr==9
  try:
    c.blah
    assert False
  except AttributeError:
    pass
  try:
    c.blah = 0
    assert False
  except AttributeError:
    pass

def test_ref():
  c = ClassTest(Object())
  assert isinstance(c.ref,Object)
  o = Object()
  c.ref = o
  assert c.ref is o
  try:
    c.ref = None
    assert False
  except TypeError:
    pass

def test_ptr():
  c = ClassTest(Object())
  o = Object()
  assert c.ptr is None
  c.ptr = o
  print c.ptr
  assert c.ptr is o
  c.ptr = None
  assert c.ptr is None

def test_ref2():
  c = ClassTest(Object())
  c.ref2 = {}
  assert isinstance(c.ref2,dict)
  c.ref2 = None
  assert c.ref2 is None

def test_ptr2():
  c = ClassTest(Object())
  assert c.ptr2 is None
  c.ptr2 = {}
  assert isinstance(c.ptr2,dict)
  c.ptr2 = None
  assert c.ptr2 is None

def test_methods():
  c = ClassTest(Object())
  assert c.normal(1)==2
  assert c.virtual_(2)==6
  assert c.static_(3)==15
  assert c(6)==24

def test_prop():
  c = ClassTest(Object())
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
    def __init__(self,o,x):
      d = self.__dict__
      self.x = x
      self.y = o
  a = A(Object(),x=5)
  assert a.x==5
  assert a.prop==17
  assert a.ref is a.y

def test_weakref():
  import weakref
  c = ClassTest(Object())
  r = weakref.ref(c)
  assert r() is c
  del c
  assert r() is None

def test_hash():
  a = ClassTest(Object())
  a.field = 17
  b = ClassTest(Object())
  b.field = 17
  assert hash(a)==hash(b)
  b.field = 18
  assert hash(a)!=hash(b)

def test_compare():
  c = ClassTest2(17)
  assert c==c
  assert not c!=c
  try:
    c<c
    assert False
  except TypeError:
    pass
  a = ClassTest(Object())
  b = ClassTest(Object())
  for i in 0,1:
    a.field = i
    for j in 0,1:
      b.field = j
      assert (a==b)==(a.field==b.field)
      assert (a!=b)==(a.field!=b.field)
      assert (a< b)==(a.field< b.field)
      assert (a> b)==(a.field> b.field)
      assert (a<=b)==(a.field<=b.field)
      assert (a>=b)==(a.field>=b.field)

if __name__=='__main__':
  test_weakref()
