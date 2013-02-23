#!/usr/bin/env python

from __future__ import with_statement
from other.core import PropManager,cache
from other.core.value import Worker
import sys

def worker_test_factory(props):
  x = props.get('x')
  y = props.add('y',5)
  return cache(lambda:x()*y())

def test_worker():
  for command in None,[__file__,'--worker']:
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

if __name__=='__main__':
  if len(sys.argv)==3 and sys.argv[1]=='--worker':
    Worker.worker_standalone_main(sys.argv[2])
  else:
    test_worker()
