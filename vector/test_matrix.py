#!/usr/bin/env python

from __future__ import absolute_import

import other.core
from numpy import *
from other.core.array import *
from other.core.vector import *

def close(a,b):
    assert all(abs(a-b)[:-1]<1e-6)

def check(x,y):
    assert type(x)==type(y)
    close(x,y)

def test_matrix():
    state=random.RandomState(221)
    A=Matrix(state.rand(4,3))
    C=Matrix(state.rand(3,4))
    x=array(state.rand(3))
    B=Matrix(state.rand(5,2,3))
    D=Matrix(state.rand(5,3,2))
    y=array(state.rand(5,3))

    check(B[-3],Matrix(array(B)[-3]))
    check(A.T,Matrix(array(A).T))
    check(B.T,Matrix([array(b).T for b in B]))

    check(A*x,array(dot(A,x)))
    check(A*C,Matrix(dot(A,C)))
    check(B*x,array([b*x for b in B]))
    check(A*y,array([A*v for v in y]))
    check(B*y,array([b*v for b,v in zip(B,y)]))
    check(A*D,Matrix([A*d for d in D]))
    check(B*C,Matrix([b*C for b in B]))
    check(B*D,Matrix([b*d for b,d in zip(B,D)]))

    A=Matrix(state.rand(3,3))
    check(A.inverse(),Matrix(linalg.inv(A)))

def test_conversions():
    A=Matrix([eye(4)]*2,dtype=other.core.real)
    A[0,1,2] = 3
    B=matrix_test(A)
    check(A,B)

def test_sparse():
    J=NestedArray([[1,0],[1,0,2],[1,2]],dtype=int32)
    A=array([-1,2,2,-1,-1,-1,2],dtype=other.core.real)
    M=SparseMatrix(J,A)
    assert M.rows()==M.columns()==3
    assert all(M.J.offsets==J.offsets)
    assert all(M.J.flat==[0,1,0,1,2,1,2])
    assert all(M.A.flat==A)
    print M.J,M.A
    b=array([pi,3,e],dtype=other.core.real)
    C=M.incomplete_cholesky_factorization(0,0)
    t=empty_like(b)
    C.solve_forward_substitution(b,t)
    x=empty_like(b) 
    C.solve_backward_substitution(t,x)
    b2=empty_like(b)
    M.multiply(x,b2)
    print b,b2
    assert all(abs(b-b2)<1e-6)
    b3=2*x-[x[1],x[0]+x[2],x[1]]
    assert all(abs(b2-b3)<1e-6)

def test_cpp():
    run_tests()
