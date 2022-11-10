import setuptools
import codecs

with codecs.open("README.rst", encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(name='rsnl',
                 version='0.0.1',
                 description='Package for RSNL algorithm for \
                              simulation-based inference',
                 long_description=long_description,
                 long_description_content_type="text/x-rst",
                 url='https://github.com/RyanJafefKelly/rsnl',
                 author='Ryan Kelly',
                 author_email='ryan@kiiii.com',
                 license='GPL',
                 packages=['rsnl'],
                 zip_safe=False,
                 python_requires='>=3.5',
                 install_requires=[
                     'jax>=0.2.0',
                     'numpyro>=0.7.0'
                 ])
