from __future__ import absolute_import

from numpy import *
from geode import *
import struct

def linear_subdivide(mesh,X,steps=1):
  for _ in xrange(steps):
    subdivide = TriangleSubdivision(mesh)
    mesh = subdivide.fine_mesh
    X = subdivide.linear_subdivide(X)
  return mesh,X

def loop_subdivide(mesh,X,steps=1,corners=zeros(0,dtype=int32)):
  for _ in xrange(steps):
    subdivide = TriangleSubdivision(mesh)
    subdivide.corners = corners
    mesh = subdivide.fine_mesh
    X = subdivide.loop_subdivide(X)
  return mesh,X

def read_obj(file):
  """Parse an obj file into a mesh and associated properties.
  Returns (mesh,props) where mesh is a PolygonSoup, and props is a dictionary containing some of X,normals,texcoord,material,face_normals,face_texcoords
  """

  # parse file
  material=None
  X,normals,texcoords=[],[],[]
  counts,vertices,face_normals,face_texcoords=[],[],[],[]
  file = open(file)
  while 1:
    line = file.readline()
    if not line:
      break
    line = line.rstrip('\r\n')
    if line.endswith('\\'):
      line = line[:-1] + file.readline()
    values=line.split()
    if not values: continue
    cmd=values[0]
    if cmd.startswith('#'):
      pass
    elif cmd=='v':
      X.append(values[1:4])
    elif cmd=='vn':
      normals.append(values[1:4])
    elif cmd == 'vt':
      texcoords.append(values[1:3])
    elif cmd in ('usemtl','usemat'):
      material=values[1]
    elif cmd=='mtllib':
      # Todo: handle reading the associated material file here
      pass
    elif cmd=='f':
      counts.append(len(values)-1)
      for v in values[1:]:
        w=v.split('/')
        vertices.append(w[0])
        if len(w)>=2 and w[1]:
          face_texcoords.append(w[1])
        if len(w)>=3 and w[2]:
          face_normals.append(w[2])

  # construct mesh
  counts=array(counts,dtype=int32)
  vertices=array(vertices,dtype=int32)-1
  mesh=PolygonSoup(counts,vertices)

  # add vertex properties
  props={}
  props['X']=array(X,dtype=float)
  props['normals']=array(normals,dtype=float)
  props['texcoords']=array(texcoords,dtype=float)

  # add extra face properties
  if face_normals:
    assert len(vertices)==len(face_normals)
    props['face_normals']=array(face_normals,dtype=int)-1
  if face_texcoords:
    assert len(vertices)==len(face_texcoords)
    props['face_texcoords']=array(face_texcoords,dtype=int)-1

  # done
  return mesh,props

def write_obj(file,mesh,X):
  """Write a simple obj file.
  For now, only mesh and positions are supported
  """

  output=open(file,"w")

  # write format
  print>>output,'''\
# simple obj file format:
#   # vertex at coordinates (x,y,z)
#   v x y z
#   # triangle [quad] with vertices a,b,c[,d]
#   f a b c [d]
#   # vertices are indexed starting from 1
'''

  # write vertices
  for x in X:
    output.write("v %.5g %.5g %.5g\n"%tuple(x))

  # write polygons
  if isinstance(mesh,TriangleSoup):
    triangles=mesh.elements+1
    for t in triangles:
      print>>output,'f',' '.join(map(str,t))
  elif isinstance(mesh,PolygonSoup):
    vertices=mesh.vertices+1
    offset=0
    for c in mesh.counts:
      print>>output,'f',' '.join(map(str,vertices[offset:offset+c]))
      offset+=c
  else:
    raise TypeError('unknown mesh type %s'%type(mesh))

def read_stl(file):
  f = open(file)
  header = f.read(80)
  count, = struct.unpack('<I',f.read(4))
  X,triangles = [],[]
  id = {}
  for t in xrange(count):
    nx,ny,nz,v0x,v0y,v0z,v1x,v1y,v1z,v2x,v2y,v2z,c = struct.unpack('<ffffffffffffH',f.read(12*4+2))
    tri = []
    for x in (v0x,v0y,v0z),(v1x,v1y,v1z),(v2x,v2y,v2z):
      try:
        i = id[x]
      except KeyError:
        i = len(X)
        X.append(x)
        id[x] = i
      tri.append(i)
    triangles.append(tri)
  return TriangleSoup(triangles),array(X,dtype=real)

def merge_meshes(surfaces):
  tris = []
  X = []
  total = 0
  for m,x in surfaces:
    tris.append(total+(m.elements if isinstance(m,TriangleSoup) else asarray(m)))
    x = asarray(x)
    X.append(x)
    total += len(x)
  return TriangleSoup(concatenate(tris).astype(int32)),concatenate(X)

