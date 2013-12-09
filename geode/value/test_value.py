#!/usr/bin/env python

from geode.value import *

def test_prop():
  i = Prop('i',3)
  assert i()==3
  assert i.is_prop()
  f = Prop('f',3.4)
  assert f()==3.4
  s = Prop('s','blah')
  assert s()=='blah'
  for p in i,f,s:
    v = p()*2 
    p.set(v)
    assert p()==v
    assert not p.dirty()

def test_array_prop():
  a = Prop('a',zeros(2),shape=(2,))
  b = Prop('b',zeros(2),shape=(-1,))
  c = Prop('c',zeros((2,2)),shape=(-1,2))
  a.set(ones(2))
  b.set(ones(7))
  c.set(ones((7,2)))
  try:
    a.set(ones(3))
    assert False
  except ValueError:
    pass
  try:
    c.set(ones((2,3)))
    assert False
  except ValueError:
    pass
  
def test_unusable():
  unusable = unusable_prop_test()
  try:
    unusable()
    assert False
  except TypeError:
    pass
  try:
    unusable.set(None)
    assert False
  except TypeError:
    pass

def test_compute():
  x = Prop('x',4)
  y = Prop('y',5)
  count = [0]
  def f():
    count[0] += 1
    return x()+y()
  z = cache(f)
  assert not z.is_prop()
  assert count[0]==0
  for i in 0,1:
    assert z()==9
    assert count[0]==1
  x.set(1)
  assert z.dirty()
  assert z()==6
  assert not z.dirty()
  assert count[0]==2

def test_cycle():
  def f():
    return x()
  x = cache(f)
  try:
    x()
    assert False
  except RuntimeError:
    pass

def test_convert():
  x = Prop('x',3)
  xt = value_test(x)
  assert xt()==3
  def f():
    return 4
  y = cache(f) 
  yt = value_test(y)
  assert yt()==4
  value_ptr_test(x)
  value_ptr_test(y)
  value_ptr_test(None)

def test_diamond():
  n = Prop('n',2)
  counts = [0,0,0]
  def f():
    counts[0] += 1
    return 2**n()
  x = cache(f)
  def g():
    counts[1] += 1
    return 3**n()
  y = cache(g)
  def fg():
    counts[2] += 1
    return x()*y() 
  xy = cache(fg)
  assert counts==[0,0,0]
  assert xy()==6**n()
  assert counts==[1,1,1]
  n.set(3)
  assert xy.dirty()
  assert xy()==6**n()
  assert counts==[2,2,2]

def test_exception():
  fail = Prop('fail',0)
  count = [0]
  def f():
    count[0] += 1
    if fail():
      raise RuntimeError('fail')
    return 3
  x = cache(f)
  def g():
    return 2*x()
  y = cache(g)
  assert y()==6
  assert count[0]==1
  fail.set(1)
  for i in 0,1:
    try:
      y()
    except RuntimeError,e:
      assert count[0]==2
      if i:
        assert error is e
      else:
        error = e
  assert not x.dirty()
  assert not y.dirty()

def test_prop_manager():
  pm = PropManager()
  test1 = pm.add("test1", 10)
  test2 = pm.add("test2", "string")

  assert test1() == 10
  assert test2() == "string"

  assert pm.get("test1")() == 10
  assert pm.get("test2")() == "string"
  assert pm.get('test1') is pm.test1
  
  test1.set(15)
  test2.set("blah")
  
  assert test1() == pm.get("test1")()
  assert test2() == pm.get("test2")()
  
  pm.get("test1").set(20)
  assert pm.get("test1")() == 20
  
  try:
    pm.get("test2").set(20)
    assert False
  except:
    pass

if __name__=='__main__':
  test_prop()
  test_compute()
  test_cycle()
  test_convert()
  test_exception()
  test_diamond()
  test_prop_manager()
