#!/usr/bin/env python

from __future__ import with_statement
from other.core.python.Frozen import Frozen

def test_frozen():
  a = Frozen()
  try:
    a.x = 0
    assert False
  except TypeError:
    pass
  with a.thaw():
    a.x = 0
  a.x = 1
  try:
    a.y = 0
    assert False
  except TypeError:
    pass
