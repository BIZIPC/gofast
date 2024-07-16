# -*- coding: utf-8 -*-
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# List all the .pyx files you want to compile
pyx_files = [
    # "adaline.pyx",
    "cluster_based.pyx",
    "_cluster_based.pyx", 
    # "ensemble.pyx",
    # "neural.pyx",
    # "tree.pyx",
    # "base.pyx",
    # "benchmark.pyx",
    # "boosting.pyx",
    # "dynamic_system.pyx",
    # "perceptron.pyx",
    # "tree.pyx",
    # "util.pyx"
]

# Create a list of extensions
module = '' # "gofast.estimators." 
extensions = [
    Extension(module + file.replace(".pyx", ""), [file],
              include_dirs=[numpy.get_include()]) for file in pyx_files
]

# Setup configuration
setup(
    name="gofast.estimators",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()],
)
