#! /usr/bin/env python
from setuptools import setup
import os

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name='mne_microstates',
        maintainer='Marijn van Vliet',
        maintainer_email='w.m.vanvliet@gmail.com',
        description='Code for microstate analysis, in combination with MNE-Python.',
        license='BSD-3',
        url='https://github.com/wmvanvliet/mne_microstates',
        version='0.1',
        long_description=open('README.md').read(),
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
        py_modules=['microstates'],
        install_requires=['numpy', 'scipy', 'mne'],
    )
