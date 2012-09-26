#!/usr/bin/env python

from json_conversion import *
from numpy import *
from other.core import *

def main():
    stuff = [ 1, 1., "hi!", True, array([1,2]) , array([1,2,3]) , array([1,2,3,4]) , Matrix([[0,1,2],[1,2,3],[2,3,4]]) , Matrix([[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6]]), Frame.identity(2), Frame.identity(3)]
    for elem in stuff:
        s = to_json(elem)
        print from_json(s)

if __name__=='__main__':
  main()
