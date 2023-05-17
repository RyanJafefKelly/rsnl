import os
from io import open

from setuptools import find_packages, setup


with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(name='rsnl',
      version='0.0.1',
      description='Package for RSNL algorithm for \
                   simulation-based inference',
      url='https://github.com/RyanJafefKelly/rsnl',
      author='Ryan Kelly',
      author_email='ryan@kiiii.com',
      license='GPL',
      packages=['rsnl'],
      zip_safe=False,
      python_requires='>=3.7',
      install_requires=requirements
      )
