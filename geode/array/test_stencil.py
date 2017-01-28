#!/usr/bin/env python

from __future__ import absolute_import
from geode import *

def test_stencil():
  for n in 7,11,19:
    for m in 5,9,13:
      for r in 1,2,3,10:
        x = m*ones((m,n),dtype=uint8)-arange(m,dtype=uint8)[:,None]
        xo = x[:,0].astype(int)
        if 0:
          print "before (m = %d, n = %d, r = %d): " % (m,n,r)
          print "x[:,0] =", x[:,0]
        apply_stencil_uint8(MaxStencil_uint8(r),r,x)
        if 0:
          print "after:"
          print "x[:,0] =", x[:,0]
        # Make sure the pattern makes sense
        diff = xo-x[:,0]
        i = arange(min(m,r+1))
        assert all(diff[i]==-i)
        i = arange(r+1,m)
        assert all(diff[i]==-r)

if __name__ == '__main__':
  test_stencil()
