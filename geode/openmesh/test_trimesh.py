#!/usr/bin/env python

from __future__ import division
from geode.vector import *
from geode.openmesh import *
from geode.geometry.platonic import *

if openmesh_enabled():
  def test_trimesh():
    # Make sure we can create one and then check its exposed functions
    trimesh = TriMesh()
    v0 = trimesh.add_vertex((0,0,0))
    v1 = trimesh.add_vertex((1,0,0))
    v2 = trimesh.add_vertex((0,1,0))
    assert (v0,v1,v2)==(0,1,2)
    assert trimesh.n_vertices()==3
    f = trimesh.add_face(v0,v1,v2)
    assert f==0
    assert trimesh.n_faces()==1
  
  def test_curvature():
    r = e
    known = [('sphere',       sphere_mesh(4,radius=r),1/r,1/(r*r))
            ,('cylinder',     open_cylinder_mesh(x0=0,x1=(1,0,0),radius= r,na=1000,nz=10), .5/r,0)]
    for name,(mesh,X),H,K in known:
      for scale in 1,-1:
        print '\n%s %g'%(name,scale)
        H *= scale
        tm = TriMesh()
        tm.add_vertices(scale*X)
        tm.add_faces(mesh.elements)
        Ha = mean_curvatures(tm)
        He = relative_error(H,Ha)
        print 'H error = %g (range %g %g, correct %g)'%(He,Ha.min(),Ha.max(),H)
        assert He<1e-5
        Ka = gaussian_curvatures(tm)
        Ke = relative_error(K,Ka,absolute=1e-6)
        print 'K error = %g (range %g %g, correct %g)'%(Ke,Ka.min(),Ka.max(),K)
        assert Ke<2e-3
  
  # Ensure readers and writers in OpenMesh were installed
  def test_openmesh_formats():
    # Statically linking against OpenMesh requires defining OM_STATIC_BUILD
    # If this isn't defined, readers and writers are never instantiated and thus aren't attached to the IOManager
    # IOManager has a can_read function, but this is broken for many of the formats so we can't use it
    # Checking the qt filters should ensure that we have attached formats to OpenMesh's IOManager
    read_filter = openmesh_qt_read_filters()
    write_filter = openmesh_qt_write_filters()
    all_extensions = ['*.off ', '*.obj ', '*.ply ', '*.stl ', '*.stla ', '*.stlb ', '*.om ']
    for ext in all_extensions:
      assert read_filter.find(ext) != -1
      assert write_filter.find(ext) != -1 or (ext == '*.stl ') # .stl appears to work, but filter seems to want stla/stlb (alpha/binary)

if __name__=='__main__':
  test_trimesh()
  test_curvature()
  test_openmesh_formats()
