#!/usr/bin/env python

from __future__ import division,print_function
from geode import *
from numpy import random

def test_heap():
  random.seed(83131)
  for n in xrange(30):
    for m in 2,max(n//3,1),1000:
      x = random.randint(m,size=n).astype(int32)
      y = heapsort_test(x)
      assert all(sort(x)==y)

if __name__=='__main__':
  test_heap()
