#! /usr/bin/env python
from setuptools import setup
import codecs
import os

# get the version (don't import, so dependencies are not needed)
version = None
with open('microstates.py') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name='mne_microstates',
        maintainer='Marijn van Vliet',
        maintainer_email='w.m.vanvliet@gmail.com',
        description=('Code for microstate analysis, in combination with '
                     'MNE-Python.'),
        license='BSD-3',
        url='https://github.com/wmvanvliet/mne_microstates',
        version=version,
        long_description=codecs.open('README.md', encoding='utf8').read(),
        long_description_content_type='text/markdown',
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        platforms='any',
        py_modules=['mne_microstates'],
        install_requires=['numpy', 'scipy', 'mne'],
    )
