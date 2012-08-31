#!/usr/bin/env python

from __future__ import division
from other.core import *

def test_predicates():
  predicate_tests(4096)

if __name__=='__main__':
  Log.configure('predicates',0,0,100)
  test_predicates()
