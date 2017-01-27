#!/usr/bin/env python

from __future__ import with_statement
from geode import Prop,PropManager,cache
from geode.value import Worker
import sys
import unittest

@unittest.skip("not a test")
def worker_test_factory(props):
  x = props.get('x')
  y = props.add('y',5)
  return cache(lambda:x()*y())

def remote(conn):
  inputs = conn.inputs
  x = inputs.get('x')
  assert x()==7
  n = Prop('n',-1)
  done = Prop('done',False)
  conn.add_output('n',n)
  conn.add_output('done',done)
  for i in xrange(10):
    n.set(i)
  done.set(True)

@unittest.skip("worker tests do not run yet")
def test_worker():
  command_file = __file__
  if command_file.endswith('.pyc'):
    command_file=command_file[:-3]+'py'
  for command in None,[command_file,'--worker']:
    props = PropManager()
    x = props.add('x',3)
    props.add('y',5)
    with Worker.Worker(debug=0,command=command) as worker:
      worker.add_props(props)
      xy = worker.create('xy',worker_test_factory)
      assert xy() is None
      worker.pull('xy')
      worker.process(timeout=None,count=1)
      assert xy()==3*5
      x.set(7) 
      worker.process(timeout=None,count=1)
      assert xy()==None
      worker.pull('xy')
      worker.process(timeout=None,count=1)
      assert xy()==7*5
      # Test remote function execution
      worker.run(remote)
      n = worker.wait_for_output('n')
      done = worker.wait_for_output('done')
      seen = []
      while not done():
        worker.process(timeout=None,count=1)
        seen.append(n())
      assert seen==range(10)+[9]

if __name__=='__main__':
  if len(sys.argv)==3 and sys.argv[1]=='--worker':
    Worker.worker_standalone_main(sys.argv[2])
  else:
    test_worker()
