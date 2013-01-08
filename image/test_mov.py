#!/usr/bin/env python

from __future__ import division

import tempfile
from numpy import *
from other.core import MovWriter

def test_mov(filename=None):
    if not filename:
        file = tempfile.NamedTemporaryFile(suffix='.mov')
        filename = file.name
    mov = MovWriter(filename,24) 
    w,h = 60,50
    y,x = meshgrid(arange(h),arange(w))
    assert x.shape==y.shape==(w,h)
    x = (x-(w-1)/2)/h
    y = (y-(h-1)/2)/h
    for f in xrange(100):
        t = 3*f/100
        if t<1:
            c = (t,0,0)
        elif t<2:
            c = (2-t,t-1,0)
        else:
            c = (0,3-t,t-2)
        a = 4*pi*t/3
        image = c*(x*cos(a)+y*sin(a)+.5).reshape(w,h,1)
        mov.add_frame(image)

if __name__=='__main__':
    test_mov('test.mov')
