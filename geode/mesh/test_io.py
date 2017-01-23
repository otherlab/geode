#!/usr/bin/env python

from __future__ import division,print_function,unicode_literals
from geode import *
import hashlib

def test_io():
  soup = TriangleSoup([(0,1,2),(2,3,4)])
  X = asarray([(0.125,1,2),(3,4,5.5),(7,5,4),(1,2,3),(9,9,9.125)])
  ascii = {'.stl': '''\
solid blah
  facet normal 1 0 0
    outer loop
      vertex 125e-3 1 2
      vertex 3 4 5.5
      vertex 7 5 4
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 7 5 4
      vertex 1 2 3
      vertex 9 9 9.125
    endloop
  endfacet
endsolid
''', '.obj': '''\
# A comment, followed by a blank line

v 0.125 1 2
v 3 4 5.5
v 7 5 4
v 1 2 3
v 9 9 9.125
f 1 2 3
f 3 4 5
''', '.ply': '''\
ply
format ascii 1.0
comment blah
element vertex 5
property float x
property float y
property float z
property uchar red
element face 2
property list uchar float texcoord
property list uchar int vertex_indices
property uchar red
end_header
0.125 1 2 8
3 4 5.5 2
7 5 4 2
1 2 3 0
9 9 9.125 255
0 3 0 1 2 4
1 255 3 2 3 4 0
''', '.x3d': '''\
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.1//EN" "http://www.web3d.org/specifications/x3d-3.1.dtd">
<X3D profile="Immersive" version="3.1" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.1.xsd" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance">
 <head>
  <meta content=".x3d format: http://en.wikipedia.org/wiki/X3d" name="description"/>
 </head>
 <Scene>
  <Shape>
   <IndexedFaceSet coordIndex="0 1 2 -1 2 3 4 -1" solid="false">
    <Coordinate point="0.125 1 2 3 4 5.5 7 5 4 1 2 3 9 9 9.125"/>
   </IndexedFaceSet>
  </Shape>
 </Scene>
</X3D>
'''}
  binary = {'.stl': '9c59baaec644425acfde9e9145ffed1a1f826f4a',
            '.ply': 'd7d04c361254c854bbd2805ea8f8703f9ad907e6'}

  for ext in '.stl .obj .ply .x3d'.split():
    f = named_tmpfile(suffix=ext)
    def check_read():
      soup2,X2 = read_soup(f.name)
      try:
        assert all(soup.elements==soup2.elements)
        assert all(X==X2)
      except:
        print('correct:\nX =\n%s\ntris =\n%s'%(X,soup.elements))
        print('got:\nX =\n%s\ntris =\n%s'%(X2,soup2.elements))
        raise
    write_mesh(f.name,soup,X)
    if ext in binary:
      sha1 = hashlib.sha1(open(f.name,'rb').read()).hexdigest()
      if sha1 != binary[ext]:
        raise RuntimeError('sha1 mismatch: ext %s, expected %s, got %s'%(ext,binary[ext],sha1))
    else:
      written =  open(f.name,'rb').read()
      try:
        if ext=='.obj':
          assert written.split('\n')[6:]==ascii[ext].split('\n')[2:]
        else:
          assert written==ascii[ext]
      except:
        print('\nWRITTEN:\n--------------------\n%s\nCORRECT:\n--------------------\n%s'%(written,ascii[ext]))
        raise
    if ext != '.x3d':
      check_read()
      open(f.name,'w').write(ascii[ext])
      check_read()

if __name__=='__main__':
  test_io()
