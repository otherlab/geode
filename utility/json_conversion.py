import json
from numpy import *
from itertools import chain
from other.core import *


def flatten(l):
  return list(chain.from_iterable(l))

def from_ndarray(v, typ = float):
  return map(typ, flatten(v))

def from_array(v, typ = float):
  return map(typ, v)


to_json_fn = {}
from_json_fn = {}

from_json_fn['int']    = lambda v: int(v)
from_json_fn['real']   = lambda v: real(v)
from_json_fn['float']  = lambda v: float(v)
from_json_fn['string'] = lambda v: str(v)
from_json_fn['bool']   = lambda v: bool(v)

from_json_fn['vec2'] = from_json_fn['vec3'] = from_json_fn['vec4'] = lambda v: array(v)

from_json_fn['mat22'] = lambda v: Matrix(array(v).reshape(2, 2))
from_json_fn['mat33'] = lambda v: Matrix(array(v).reshape(3, 3))
from_json_fn['mat44'] = lambda v: Matrix(array(v).reshape(4, 4))

from_json_fn['frame2'] = lambda v: Frames(v['t'], Rotation.from_matrix(array(v['r']).reshape(2, 2)))
from_json_fn['frame3'] = lambda v: Frames(v['t'], Rotation.from_matrix(array(v['r']).reshape(3, 3)))

from_json_fn['box2'] = from_json_fn['box3'] = lambda v: Box(v['min'], v['max'])
from_json_fn['TriangleMesh'] = from_json_fn['SegmentMesh'] = lambda v: v


to_json_fn[int]   = lambda v: { "t": "int",    "v": v }
to_json_fn[real]  = lambda v: { "t": "real",   "v": v }
to_json_fn[float] = lambda v: { "t": "float",  "v": v }
to_json_fn[str]   = lambda v: { "t": "string", "v": v }
to_json_fn[bool]  = lambda v: { "t": "bool",   "v": v }

to_json_fn[Box2d] = to_json_fn[Box3d] = lambda v: {
  "t": ('box%s') % len(v.min),
  "v": {
    "min": from_array(v.min),
    "max": from_array(v.max)
  }
}

to_json_fn[ndarray] = lambda v: {
  "t": ('vec%s') % len(v),
  "v": from_array(v)
}
to_json_fn[Matrix] = lambda v: {
  "t": ('mat%s%s') % (len(v), len(v[0])),
  "v": from_ndarray(v)
}
to_json_fn[Frames] = lambda v: { # send matrix over the wire or make javascript compose t and r?
  "t": ('frame%s') % (len(v.t)),
  "v": {
    "t": map(float, v.t),
    "r": from_ndarray(v.r.matrix()),
    "m": from_ndarray(v.matrix())
  }
}

to_json_fn[TriangleMesh] = lambda v: {
	"t": "TriangleMesh",
	"v": from_ndarray(v.elements, int)
}
to_json_fn[SegmentMesh] = lambda v: {
	"t": "SegmentMesh",
	"v": from_ndarray(v.elements, int)
}


# def dictionaried(v):
#   return to_json_fn[type(v)](v)

def to_json(v):
  fn = to_json_fn.get(type(v), None)
  return fn(v) if callable(fn) else None

def to_json_string(v):
  return json.dumps(to_json(v), allow_nan = False, separators = (",", ":"))

def from_json(d):
  fn = from_json_fn.get(d['t'])
  return fn(d['v']) if callable(fn) else None

def from_json_string(s):
  return from_json(json.loads(s))


def register(typ, name, to_fn, from_fn):
  to_json_fn[typ] = to_fn
  from_json_fn[name] = from_fn
