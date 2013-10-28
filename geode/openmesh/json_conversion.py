from geode.utility.json_conversion import register
from geode.openmesh import *

to_fn = lambda v: {
  't': 'TriMesh',
  'v': {
    'vertices': from_ndarray(v.X()),
    'elements': from_ndarray(v.elements(), int)
  }
}

from_fn = lambda d: d['v']

register(TriMesh, 'TriMesh', to_fn, from_fn)
