# Usage:
#
#   python setup.py install
#
# to install this package in your Python environment.
#
# Note for future packages updates:
#
# Use
#
#   python setup.py check sdist --formats=gztar,zip upload
#
# to update PyPi with new .tar.gz and .zip release files.

from distutils.core import setup

setup(
    name="pyaixi",
    version="2.0.0",
    author="SG Kassel",
    maintainer="nbro",
    packages=[
        "pyaixi",
        "pyaixi.agents",
        "pyaixi.environments",
        "pyaixi.prediction",
        "pyaixi.search",
    ],
    scripts=["aixi.py"],
    url="https://github.com/sgkasselau/pyaixi",
    license="Creative Commons Attribution-ShareAlike 3.0 Unported License",
    description="A pure Python implementation of the "
    "Monte Carlo-AIXI-Context Tree Weighting (MC-AIXI-CTW) "
    "artificial intelligence algorithm.",
    long_description=open("README.md").read(),
    keywords=[
        "MC-AIXI-CTW",
        "AIXI",
        "universal intelligence",
        "artificial general intelligence",
        "AGI",
        "intelligent agent",
        "artificial intelligence",
        "reinforcement learning",
        "machine learning",
        "prediction",
        "search",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "License :: Creative Commons Attribution-ShareAlike 3.0 Unported "
        "License",
        "Intended Audience :: Developers",
        "Intended Audience :: Computer Scientists",
        "Intended Audience :: Students",
        "Intended Audience :: Researchers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
