"""log module"""

from __future__ import (with_statement,absolute_import)
from contextlib import contextmanager
from ..xdress import wrap

configure = wrap.log_configure
initialized = wrap.log_initialized
cache_initial_output = wrap.log_cache_initial_output
copy_to_file = wrap.log_copy_to_file
finish = wrap.log_finish
write = wrap.log_print
error = wrap.log_error
flush = wrap.log_flush

@contextmanager
def scope(format,*args):
  wrap.log_push_scope(format%args)
  try:
    yield
  finally:
    wrap.log_pop_scope()
