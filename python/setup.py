#!/usr/bin/env python

from setuptools import setup, Command, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='deep_calibration',
      version='0.0',
      description='Deep Calibration PRR 2020',
      license='MIT License',
      author='Nicolas Makaroff',
      author_email='nicolas.makaroff@ensiie.fr',
      url='https://github.com/https://github.com/NicolasMakaroff/deep_calibration',
      packages=find_packages(),
      install_requires = required,
      long_description= (" ")
     )
