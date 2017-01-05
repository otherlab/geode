#!/usr/bin/env python

import sys
import os.path
from distutils.sysconfig import get_python_lib
print(os.path.dirname(get_python_lib(standard_lib=True)))
