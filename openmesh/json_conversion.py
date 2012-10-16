from other.core.utility.json_conversion import *
from other.core.openmesh import *

to_json_fn[TriMesh] = lambda v: {
  't': 'TriMesh',
  'v': {
    'vertices': from_ndarray(v.X()),
    'elements': from_ndarray(v.elements(), int)
  }
}

from_json_fn = lambda d: d['v']

register(TriMesh, 'TriMesh', to_json_fn, from_json_fn)
