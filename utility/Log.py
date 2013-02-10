"""log module"""

from __future__ import (with_statement,absolute_import)
from contextlib import contextmanager
import platform

if platform.system()=='Windows':
  import other_all as other_core
else:
  import other_core

configure = other_core.log_configure
initialized = other_core.log_initialized
cache_initial_output = other_core.log_cache_initial_output
copy_to_file = other_core.log_copy_to_file
finish = other_core.log_finish
write = other_core.log_print
error = other_core.log_error
flush = other_core.log_flush

@contextmanager
def scope(format,*args):
  other_core.log_push_scope(format%args)
  try:
    yield
  finally:
    other_core.log_pop_scope()
