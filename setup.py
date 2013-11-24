import os

# Usage: python setup.py install
#        to install this package in your Python environment.
#
# Note for future packages updates:
#
# Use 'python setup.py check sdist --formats=gztar,zip upload' to update PyPi with new
# .tar.gz and .zip release files.

from distutils.core import setup

setup(
    name = 'pyaixi',
    version = '1.0.0',
    author = 'Geoff Kassel',
    author_email = 'geoffkassel_at_gmail_dot_com',
    package_dir = {'pyaixi': os.curdir},
    packages = ['pyaixi', 'pyaixi.agents', 'pyaixi.environments', 'pyaixi.prediction', 'pyaixi.search'],
    scripts = ['aixi.py'],
    url = 'https://github.com/gkassel/pyaixi',
    license = 'Creative Commons Attribution-ShareAlike 3.0 Unported License',
    description = 'A pure Python implementation of the Monte Carlo-AIXI-Context Tree Weighting (MC-AIXI-CTW) artificial intelligence algorithm.',
    long_description = open('README.md').read(),
    keywords = ["AIXI", "agent", "artificial intelligence", "general learning", "machine learning",
                "MC-AIXI-CTW", "model based", "reinforcement learning", "prediction", "search"],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'License :: Freely Distributable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)