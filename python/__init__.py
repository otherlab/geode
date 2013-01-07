from __future__ import absolute_import

from other.core import (real,redefine_assertion_error)

# py.test overrides AssertionError, so make sure C++ knows about it
redefine_assertion_error(AssertionError)
