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


to_jsons = {}
from_jsons = {}

from_jsons['int']    = lambda d: int(d['v'])
from_jsons['real']   = lambda d: real(d['v'])
from_jsons['float']  = lambda d: float(d['v'])
from_jsons['string'] = lambda d: str(d['v'])
from_jsons['bool']   = lambda d: bool(d['v'])

from_jsons['vec2'] = from_jsons['vec3'] = from_jsons['vec4'] = lambda d: array(d['v'])

from_jsons['mat22'] = lambda d: Matrix(array(d['v']).reshape(2, 2))
from_jsons['mat33'] = lambda d: Matrix(array(d['v']).reshape(3, 3))
from_jsons['mat44'] = lambda d: Matrix(array(d['v']).reshape(4, 4))

from_jsons['frame2'] = lambda d: Frames(d['v']['t'], Rotation.from_matrix(array(d['v']['r']).reshape(2, 2)))
from_jsons['frame3'] = lambda d: Frames(d['v']['t'], Rotation.from_matrix(array(d['v']['r']).reshape(3, 3)))

from_jsons['box2'] = from_jsons['box3'] = lambda d: Box(d['v']['min'], d['v']['max'])


to_jsons[int]   = lambda v: { "t": "int",    "v": v }
to_jsons[real]  = lambda v: { "t": "real",   "v": v }
to_jsons[float] = lambda v: { "t": "float",  "v": v }
to_jsons[str]   = lambda v: { "t": "string", "v": v }
to_jsons[bool]  = lambda v: { "t": "bool",   "v": v }

to_jsons[Box2d] = to_jsons[Box3d] = lambda v: {
  "t": ('box%s')%len(v.min),
  "v": {
    "min": from_array(v.min),
    "max": from_array(v.max)
  }
}

to_jsons[ndarray] = lambda v: {
  "t": ('vec%s') % len(v),
  "v": from_array(v)
}
to_jsons[Matrix] = lambda v: {
  "t": ('mat%s%s') % (len(v), len(v[0])),
  "v": from_ndarray(v)
}
to_jsons[Frames] = lambda v: { # send matrix over the wire or make javascript compose t and r?
  "t": ('frame%s') % (len(v.t)),
  "v": {
    "t": map(float, v.t),
    "r": from_ndarray(v.r.matrix()),
    "m": from_ndarray(v.matrix())
  }
}


def to_json(v):
  fn = to_jsons.get(type(v), None)
  return fn(v) if callable(fn) else None

def to_json_string(v):
  return json.dumps(to_json(v), allow_nan = False, separators = (",", ":"))

def from_json(d):
  fn = from_jsons.get(d['t'])
  return fn(d) if callable(fn) else None

def from_json_string(s):
  return from_json(json.loads(s))


def register(typ, name, tofn, fromfn):
  to_jsons[typ] = tofn
  from_jsons[name] = fromfn
