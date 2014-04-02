#!/usr/bin/env python

from setuptools import setup,find_packages,Extension
from Cython.Build import cythonize
from Cython.Build.Dependencies import create_extension_list
import glob
import os

# TODO: Don't hardcode these
arch = 'native'
type = 'release'

# Work around bugs in cythonize: https://groups.google.com/forum/#!topic/cython-users/b7Kvw2sbbEA
os.chdir(os.path.dirname(__file__))
build = 'build/%s/%s'%(arch,type)
extensions = []
for e in glob.glob(build+'/geode/xdress/*.pyx'):
  name = os.path.splitext(os.path.basename(e))[0]
  extensions.append(Extension('geode.xdress.'+name,[e],
                              language='c++',
                              extra_compile_args=['-std=c++11'],
                              include_dirs=['.'],
                              library_dirs=[build+'/lib'],
                              libraries=['geode']))

setup(
  # Basics
  name='geode',
  version='0.0-dev',
  description='A computational geometry library',
  author='Otherlab et al.',
  author_email='geode-dev@googlegroups.com',
  url='http://github.com/otherlab/geode',

  # Installation
  packages=find_packages(),
  package_data={'geode':['*.py','*.so']},
  ext_modules=cythonize(extensions)
)
