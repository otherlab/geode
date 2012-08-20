#!/usr/bin/env python

from __future__ import absolute_import

from numpy import *
from other.core import real
from other.core.force import *

def test_gravity():
    random.seed(12871)
    X = random.randn(1,3)
    gravity = Gravity([1])
    force_test(gravity,X,verbose=1)

def test_ether_drag():
    random.seed(12871)
    X = random.randn(1,3)
    drag = EtherDrag([1.2],7)
    force_test(drag,X,verbose=1)

def test_springs():
    random.seed(12871)
    X0 = random.randn(2,3)
    X = random.randn(2,3)
    springs = Springs([[0,1]],[1.1,1.2],X0,5,7)
    force_test(springs,X,verbose=1)
    springs = Springs([[0,1]],[1.1,1.2],X0,[5],[7])
    force_test(springs,X,verbose=1)

def test_fvm_2d():
    random.seed(12872)
    model = neo_hookean()
    X = random.randn(3,2)
    dX = .1*random.randn(3,2)
    fvm = finite_volume([(0,1,2)],1000,X,model)
    force_test(fvm,X+dX,verbose=1)

def test_fvm_s3d():
    random.seed(12872)
    model = neo_hookean()
    X = random.randn(3,3)
    dX = .1*random.randn(3,3)
    fvm = finite_volume([(0,1,2)],1000,X,model)
    force_test(fvm,X+dX,verbose=1)

def test_fvm_3d():
    random.seed(12873)
    model = neo_hookean()
    X = random.randn(4,3)
    dX = .1*random.randn(4,3)
    fvm = finite_volume([(0,1,2,3)],1000,X,model)
    force_test(fvm,X+dX,verbose=1)

def test_linear_bending_2d():
    random.seed(7218414) 
    X = random.randn(3,2)
    bend = linear_bending_elements(SegmentMesh([[0,1],[1,2]]),X,7,3)
    dX = .1*random.randn(3,2) 
    force_test(bend,X+dX,verbose=1)

def test_linear_bending_3d():
    random.seed(7218414) 
    X = random.randn(4,3)
    bend = linear_bending_elements(TriangleMesh([[0,1,2],[1,3,2]]),X,7,3)
    dX = .1*random.randn(4,3) 
    force_test(bend,X+dX,verbose=1)

def test_linear_fvm_2d():
    random.seed(12872)
    X = random.randn(3,2)
    dX = .1*random.randn(3,2)
    fvm = linear_finite_volume([(0,1,2)],X,1000)
    force_test(fvm,X+dX,verbose=1)

def test_linear_fvm_s3d():
    random.seed(12872)
    X = random.randn(3,3)
    dX = .1*random.randn(3,3)
    fvm = linear_finite_volume([(0,1,2)],X,1000)
    force_test(fvm,X+dX,verbose=1)

def test_linear_fvm_3d():
    random.seed(12873)
    X = random.randn(4,3)
    dX = .1*random.randn(4,3)
    fvm = linear_finite_volume([(0,1,2,3)],X,1000)
    force_test(fvm,X+dX,verbose=1)

def test_linear_fvm_hex():
    random.seed(12873)
    X = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]+.1*random.randn(8,3)
    dX = .1*random.randn(8,3)
    fvm = linear_finite_volume([arange(8)],X,1000)
    force_test(fvm,X+dX,verbose=1)
