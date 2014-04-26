#!/usr/bin/env python

from __future__ import division

from numpy import *
from geode import Nested, PolygonSoup, SegmentSoup, TriangleSoup
from geode import vertex_position_id, write_mesh, TriangleTopology, MutableTriangleTopology, csg
from geode.geometry.platonic import icosahedron_mesh, sphere_mesh
from geode.vector import relative_error

def test_intersection():
  imesh,iX = icosahedron_mesh()
  smesh,sX = sphere_mesh(1)

  im = MutableTriangleTopology()
  im.add_vertices(max(imesh.elements.flat)+1)
  im.add_faces(imesh.elements)
  ip = im.add_vertex_field("3d", vertex_position_id)
  im.field(ip)[:] = iX

  write_mesh("input-i.obj", im, im.field(ip))

  sm = MutableTriangleTopology()
  sm.add_vertices(max(smesh.elements.flat)+1)
  sm.add_faces(smesh.elements)
  sp = sm.add_vertex_field("3d", vertex_position_id)

  for d in [.3, .4, .5]: 
    sm.field(sp)[:] = sX + d

    write_mesh("input-s-%f.obj" % d, sm, sm.field(sp))

    # compute the union
    u = im.copy()
    u.add(sm)
    csg(u, 0, vertex_position_id);
    write_mesh("union-%f.obj" % d, u, u.field(ip))

    # compute the intersection
    i = im.copy()
    i.add(sm)
    csg(i, 1, vertex_position_id);
    write_mesh("intersection-%f.obj" % d, i, i.field(ip))

if __name__ == "__main__":
  test_intersection()