import json
from numpy import *
from other.core import *

to_jsons = {}
from_jsons = {}

from_jsons['int']=(lambda d: int(d['value']))
from_jsons['real']=lambda d: real(d['value'])
from_jsons['float']=lambda d: float(d['value'])
from_jsons['string']=lambda d: str(d['value'])
from_jsons['bool']=lambda d: bool(d['value'])

from_jsons['vec2']=lambda d: array(d['value'])
from_jsons['vec3']=lambda d: array(d['value'])
from_jsons['vec4']=lambda d: array(d['value'])

from_jsons['mat22']=lambda d: Matrix([d['value'][0],
                                      d['value'][1]
                                      ])

from_jsons['mat33']=lambda d: Matrix([d['value'][0],
                                      d['value'][1],
                                      d['value'][2]])

from_jsons['mat44']=lambda d: Matrix([d['value'][0],
                                      d['value'][1],
                                      d['value'][2],
                                      d['value'][3]
                                     ])

from_jsons['frame2']=lambda d: Frames(d['value']['t'],Rotation.from_matrix(d['value']['r']) )
from_jsons['frame3']=lambda d: Frames(d['value']['t'],Rotation.from_matrix(d['value']['r']) )


to_jsons[int] = lambda v: {"type":"int","value":v}
to_jsons[real] = lambda v:{"type":"real","value":v}
to_jsons[float] = lambda v:{"type":"float","value":v}
to_jsons[str] = lambda v: {"type":"string","value":v}
to_jsons[bool] = lambda v: {"type":"bool","value":v}

to_jsons[ndarray] = lambda v: {"type":('vec%s')%len(v),"value": map(float,v)}
to_jsons[Matrix] = lambda v: {"type":('mat%s%s')%(len(v),len(v[0])),"value":map(lambda x:map(float,x),v)}
to_jsons[Frames] = lambda v: {"type":('frame%s')%(len(v.t)),"value":{"t":map(float,v.t),"r":map(lambda x:map(float,x),v.r.matrix()),"m":map(lambda x:map(float,x),v.matrix())} } #send matrix over the wire or make javascript compose t and r?

def dictionaried(v):
    return to_jsons[type(v)](v)

def to_json(v):
    tj = dictionaried(v)
    js = json.dumps(tj,allow_nan = False, separators = (",", ":"))
    return js

def from_json(j):
    d = json.loads(j)
    return from_jsons[d['type']](d)
