from __future__ import absolute_import

import platform
def is_windows():
  return platform.system()=='Windows'

# Import other_core, possibly as other_all on windows
if is_windows():
  import other_all as other_core
  from other_all import *
else:
  import other_core
  from other_core import *

# py.test overrides AssertionError, so make sure C++ knows about it
other_core.redefine_assertion_error(AssertionError)

# Import children
from .utility import *
from .array import *
from .geometry import *
from .vector import *
from .mesh import *
real = other_core.real
