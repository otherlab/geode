from __future__ import absolute_import

import platform
def is_windows():
  return platform.system()=='Windows'

if is_windows():
  import other_all as other_core
  from other_all import *
else:
  import other_core
  from other_core import *
from .python import real
from .utility import *
from .array import *
from .geometry import *
from .vector import *
from .mesh import *
real = other_core.real
