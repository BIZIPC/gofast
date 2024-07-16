# -*- coding: utf-8 -*-
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# List all the .pyx files you want to compile
pyx_files = [
    "enrichment.pyx",
    "inspection.pyx",
    "management.pyx",
    "preprocessing.pyx",
    "quality.pyx",
    "transformation.pyx"
]

# Create a list of extensions
extensions = [
    Extension("gofast.dataops." + file.replace(".pyx", ""), [file],
              include_dirs=[numpy.get_include()]) for file in pyx_files
]

# Setup configuration
setup(
    name="gofast.dataops",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
# python setup.py build_ext --inplace
