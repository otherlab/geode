from __future__ import absolute_import

import platform
def is_windows():
  return platform.system()=='Windows'

# Import geode_wrap, possibly as geode_all on windows
if is_windows():
  from . import geode_all as geode_wrap
  from .geode_all import *
else:
  from . import geode_wrap
  from .geode_wrap import *

# py.test overrides AssertionError, so make sure C++ knows about it
geode_wrap.redefine_assertion_error(AssertionError)

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
