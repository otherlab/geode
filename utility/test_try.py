from __future__ import absolute_import

from other.core.utility import tryfile
from numpy import *
import py.test

def check(a,b):
  if isinstance(a,dict) and isinstance(b,dict):
    assert len(a)==len(b)
    for n,av in a.iteritems():
      check(av,b[n])
  elif isinstance(a,tuple) and isinstance(b,tuple):
    assert len(a)==len(b)
    for av,bv in zip(a,b):
      check(av,bv)
  elif isinstance(a,ndarray) or isinstance(b,ndarray):
    assert all(a==b)
  else:
    assert False

dir = None
def setup_module(module):
  module.dir = py.test.ensuretemp('try')

def test_try():
  file = dir.strpath+'/try.try'
  value={'a':{'b':array([1,2]),
              'c':array([1.,2.,3.])},
         'd':array([4,5],dtype=float64),
         'e':3.0,
         'f':float64(4),
         'g':5}
  tryfile.write(file,value)
  check(value,tryfile.read(file))
  check(value,tryfile.unpack(tryfile.pack(value)))

def test_version_1():
  file = dir.strpath+'/1.try' 
  open(file,'wb').write('\x03TRY\x03\x01\x0a\x03\x00\x05array\x00\x01\x03\x02\x00\x11')
  check(tryfile.read(file),17)

def test_version_2():
  file = dir.strpath+'/2.try'
  raw = '\x03TRY\x07\x02\x0e\x0e\xe9\x28\xd7\n\x00\x05array\x00\x07\x0eA\xc8~@x\x9cce\x10d``\x00\x00\x00h\x00\x17'
  # Verify that our raw string is a correct version 2 file
  open(file,'wb').write(raw)
  check(tryfile.read(file),17)
  # Verify that deleting any byte or changing any bit is detected
  def bad(s):
    open(file,'w').write(s)
    py.test.raises(Exception,tryfile.read,file)
  for i in xrange(len(raw)):
    bad(raw[:i]+raw[i+1:])
    for j in xrange(8):
      bad(raw[:i]+chr(ord(raw[i])^(1<<j))+raw[i+1:])
