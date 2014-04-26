#!/usr/bin/env python

from __future__ import division

from numpy import *
from geode import Nested, PolygonSoup, SegmentSoup, TriangleSoup
from geode import vertex_position_id, write_mesh, TriangleTopology, MutableTriangleTopology, csg
from geode.geometry.platonic import icosahedron_mesh, sphere_mesh
from geode.vector import relative_error

def test_intersection(output=False):
  imesh,iX = icosahedron_mesh()
  smesh,sX = sphere_mesh(1)

  im = MutableTriangleTopology()
  im.add_vertices(max(imesh.elements.flat)+1)
  im.add_faces(imesh.elements)
  ip = im.add_vertex_field("3d", vertex_position_id)
  im.field(ip)[:] = iX

  sm = MutableTriangleTopology()
  sm.add_vertices(max(smesh.elements.flat)+1)
  sm.add_faces(smesh.elements)
  sp = sm.add_vertex_field("3d", vertex_position_id)

  sm2 = MutableTriangleTopology()
  sm2.add_vertices(max(smesh.elements.flat)+1)
  sm2.add_faces(smesh.elements)
  sp2 = sm2.add_vertex_field("3d", vertex_position_id)

  for d in [.3, .4, .5]: 
    sm.field(sp)[:] = sX + d

    # compute the union
    u = im.copy()
    u.add(sm)

    if output:
      write_mesh("input-%f.obj" % d, u, u.field(ip))

    csg(u, 0, vertex_position_id);
    if output:
      write_mesh("union-%f.obj" % d, u, u.field(ip))

    # compute the intersection
    i = im.copy()
    i.add(sm)
    csg(i, 1, vertex_position_id);
    if output:
      write_mesh("intersection-%f.obj" % d, i, i.field(ip))

  for d in [1.55]:
    sm.field(sp)[:] = sX + .86
    sm2.field(sp2)[:] = sX + tile(array([0,d,0.1]), (sm.field(sp2).shape[0], 1))

    # compute the union
    u = im.copy()
    u.add(sm)
    u.add(sm2)

    if output:
      write_mesh("input2-%f.obj" % d, u, u.field(ip))

    csg(u, 0, vertex_position_id);
    if output:
      write_mesh("union2-%f.obj" % d, u, u.field(ip))

    # compute the intersection
    i = im.copy()
    i.add(sm)
    i.add(sm2)
    csg(i, 1, vertex_position_id);
    if output:
      write_mesh("intersection2-%f.obj" % d, i, i.field(ip))


if __name__ == "__main__":
  test_intersection(output=True)