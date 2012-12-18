"""log module"""

from __future__ import (with_statement,absolute_import)
from contextlib import contextmanager

from other_core import (
  log_configure as configure,
  log_cache_initial_output as cache_initial_output,
  log_copy_to_file as copy_to_file,
  log_finish as finish,
  log_print as write,
  log_error as error,
  log_flush as flush,
  log_push_scope,
  log_pop_scope)

@contextmanager
def scope(format,*args):
  log_push_scope(format%args)
  try:
    yield
  finally:
    log_pop_scope()
