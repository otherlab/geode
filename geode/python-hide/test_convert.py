from geode import *
import ast

def test_list():
  x = [1,2,3]
  y = list_convert_test(x)
  assert type(x)==type(y)
  assert x==y

def test_set():
  x = set([1,2,3])
  y = set_convert_test(x)
  assert type(x)==type(y)

def test_dict():
  x = {1:'a',2:'b',3:'c'}
  y = dict_convert_test(x)
  assert type(x)==type(y)
  assert x==y

def test_enum():
  a = EnumTestA
  aa = enum_convert_test(a)
  assert a==aa
  assert a is aa
  assert str(a)=='EnumTestA'

def test_str_repr():
  for i in xrange(256):
    c = chr(i)
    r = str_repr_test(c)
    assert ast.literal_eval(r)==c
    if c != "'":
      assert repr(c)==r
