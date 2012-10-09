from other.core.utility.json_conversion import *
from other.core.openmesh import *

tofn = lambda v: {"t":"TriMesh","v":{'X':from_ndarray(v.X()),'elements':from_ndarray(v.elements(),int)}}
fromfn =  lambda d: d['v']

register(TriMesh,'TriMesh',tofn,fromfn)
