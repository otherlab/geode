import json

from numpy import *
from geode import *

def from_ndarray(v, typ = float):
  return map(typ, v.flatten())

def from_array(v, typ = float):
  return map(typ, v)


to_json_fn = {}
from_json_fn = {}

from_json_fn['int']    = lambda v: int(v)
from_json_fn['real']   = lambda v: real(v)
from_json_fn['float']  = lambda v: float(v)
from_json_fn['string'] = lambda v: str(v)
from_json_fn['bool']   = lambda v: bool(v)

from_json_fn['ndarray'] = lambda v : array(v)

from_json_fn['mat22'] = lambda v: Matrix(array(v).reshape(2, 2))
from_json_fn['mat33'] = lambda v: Matrix(array(v).reshape(3, 3))
from_json_fn['mat44'] = lambda v: Matrix(array(v).reshape(4, 4))

from_json_fn['frame2'] = lambda v: Frames(v['t'], Rotation.from_sv(array(v['r'])))
from_json_fn['frame3'] = lambda v: Frames(v['t'], Rotation.from_sv(array(v['r'])))

from_json_fn['box2'] = from_json_fn['box3'] = lambda v: Box(v['min'], v['max'])
from_json_fn['TriangleSoup'] = from_json_fn['SegmentSoup'] = lambda v: v

from_json_fn['dict'] = lambda v: v

to_json_fn[dict] = lambda v: { 't': 'dict', 'v': v }

to_json_fn[int]   = lambda v: { 't': 'int',    'v': v }
to_json_fn[real]  = lambda v: { 't': 'real',   'v': v }
to_json_fn[float] = lambda v: { 't': 'float',  'v': v }
to_json_fn[str]   = lambda v: { 't': 'string', 'v': v }
to_json_fn[bool]  = lambda v: { 't': 'bool',   'v': v }

to_json_fn[Box2d] = to_json_fn[Box3d] = lambda v: {
  't': ('box%s') % len(v.min),
  'v': {
    'min': from_array(v.min),
    'max': from_array(v.max)
  }
}

to_json_fn[list] = lambda v: {
  't': 'list',
  'v': v # let's hope this works on the client...
}

to_json_fn[ndarray] = lambda v: {
  't': 'ndarray',
  'v': {
    'shape': v.shape,
    'data': from_ndarray(v)
  }
}
to_json_fn[Matrix] = lambda v: {
  't': ('mat%s%s') % (len(v), len(v[0])),
  'v': from_ndarray(v)
}
to_json_fn[Frames] = lambda v: {
  't': ('frame%s') % (len(v.t)),
  'v': {
    't': map(float, v.t),
    'r': map(float, v.r.sv)
  }
}

to_json_fn[TriangleSoup] = lambda v: {
  't': 'TriangleSoup',
  'v': from_ndarray(v.elements, int)
}
to_json_fn[SegmentSoup] = lambda v: {
  't': 'SegmentSoup',
  'v': from_ndarray(v.elements, int)
}

to_json_fn[MutableTriangleTopology] = lambda v: {
  't': 'TriangleTopology',
  'v': {
    'vertices': from_ndarray(v.vertex_field(vertex_position_id)),
    'elements': from_ndarray(v.elements(), int)
  }
}

if openmesh_enabled():
  to_json_fn[TriMesh] = lambda v: {
    't': 'TriMesh',
    'v': {
      'vertices': from_ndarray(v.X()),
      'elements': from_ndarray(v.elements(), int)
    }
  }
  from_json_fn[TriMesh] = lambda d: d['v']

def to_json(v):
  fn = to_json_fn.get(type(v), None)
  if callable(fn):
    return fn(v)
  else:
    raise TypeError("Don't know how to transscribe type %s to json." % type(v))

def to_json_string(v):
  return json.dumps(to_json(v), allow_nan = False, separators = (',', ':'))

def from_json(d):
  fn = from_json_fn.get(d['t'])
  return fn(d['v']) if callable(fn) else None

def from_json_string(s):
  return from_json(json.loads(s))

def register(typ, name, to_fn, from_fn):
  to_json_fn[typ] = to_fn
  from_json_fn[name] = from_fn
