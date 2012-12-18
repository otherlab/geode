"""Extended matrix support for numpy

This defines a Matrix class derived from numpy.array which is more
convenient than numpy.matrix in several respects:

1. matrix * vector works correctly: one dimensional arrays are treated
   as column vectors, and the result is returned as a 1D array.
   array of vectors also work as expected.

2. Matrix objects with rank higher than two are treated as arrays of
   matrices.  These work as expected when combined with arrays of
   vectors.  E.g., an array of matrices times a vector transforms the
   vector by each matrix, and an array of $n$ matrices times an array
   of $n$ vectors does the expected thing.

The goal is to be able to think of vectors and matrices as primitive
entities where the normal numpy notions of broadcasting work as
expected.
"""

from __future__ import (division,absolute_import)

import sys
import numpy

class Matrix(numpy.ndarray):
    def __new__(cls, obj, dtype=None):
        return numpy.asarray(obj,dtype=dtype).view(cls)

    __array_priority__ = -1.

    def _T(self):
        return self.swapaxes(-1,-2)
    T = property(_T)

    def __mul__(self,x):
        if not isinstance(x,numpy.ndarray):
            x = numpy.asarray(x)
        if x.ndim==1:
            return numpy.dot(self,x)
        elif not isinstance(x,Matrix):
            if self.ndim==2:
                return numpy.dot(x,self.T)
            else:
                return numpy.multiply(self,x.reshape(x.shape[:-1]+(1,-1))).sum(axis=-1).view(type(x))
        elif x.ndim==2:
            return numpy.dot(self,x)
        elif self.ndim==2:
            return numpy.dot(x.swapaxes(-1,-2),self.T).swapaxes(-1,-2)
        else:
            # It's impossible to implement this case in O(1) numpy operations without
            # potential memory blowup.  We'll assume the matrices are small and ignore
            # the memory issue.
            return numpy.multiply(self.reshape(self.shape+(1,)),x.reshape(x.shape[:-2]+(1,)+x.shape[-2:])).sum(axis=-2)

    def inverse(self):
        return numpy.linalg.inv(self)

import other_core
other_core._set_matrix_type(Matrix)
