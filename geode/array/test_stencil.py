#!/usr/bin/env python

from __future__ import absolute_import

from numpy import *
from geode import *

def test_stencil():
  for n in [10, 100, 255]:
    for m in [10, 100, 255]:
      for r in [1, 2, 3, 10, 50]:
        x = ndarray(shape=(m,n), dtype=numpy.dtype('u1'))
        for i in range(m):
          for j in range(n):
            x[i,j] = m-i

        xo = array(x[:,0], dtype=int).copy()

        print "before (m = %d, n = %d, r = %d): " % (m,n,r)
        print "x[:,0] =", x[:,0]

        apply_stencil_uint8(MaxStencil_uint8(r), r, x)

        print "after:"
        print "x[:,0] =", x[:,0]

        # make sure the pattern makes sense
        diff = xo - array(x[:,0], dtype=int)
        for i in range(min(m,r+1)):
          assert diff[i] == -i
        for i in range(r+1,m):
          assert diff[i] == -r


if __name__ == '__main__':
  test_stencil()
