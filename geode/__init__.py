from __future__ import absolute_import

import platform
def is_windows():
  return platform.system()=='Windows'

# Import xdress bindings
from .xdress.dtypes import *
from .xdress.stlcontainers import *
from .xdress.xdress_extra_types import *
from .xdress.wrap import *

# Import children
from .utility import *
from .array import *
if has_exact():
  from .exact import *
from .geometry import *
from .value import *
from .vector import *
from .mesh import *
real = geode_wrap.real
