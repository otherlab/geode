"""log module"""

from __future__ import (with_statement,absolute_import)
from contextlib import contextmanager
import platform

if platform.system()=='Windows':
  from ..import geode_all as geode_wrap
else:
  from .. import geode_wrap

configure = geode_wrap.log_configure
initialized = geode_wrap.log_initialized
cache_initial_output = geode_wrap.log_cache_initial_output
copy_to_file = geode_wrap.log_copy_to_file
finish = geode_wrap.log_finish
write = geode_wrap.log_print
error = geode_wrap.log_error
flush = geode_wrap.log_flush

@contextmanager
def scope(format,*args):
  geode_wrap.log_push_scope(format%args)
  try:
    yield
  finally:
    geode_wrap.log_pop_scope()
