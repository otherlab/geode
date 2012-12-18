from __future__ import absolute_import

from other_core import *

# py.test overrides AssertionError, so make sure C++ knows about it
redefine_assertion_error(AssertionError)
