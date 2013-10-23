#!/usr/bin/env python

from setuptools import setup,find_packages

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
)
