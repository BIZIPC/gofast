# -*- coding: utf-8 -*-
import os # noqa
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# List all the .pyx files you want to compile
pyx_files = [
    "feature_engineering.pyx",
]

# Ensure the target directory exists
# target_dir = os.path.join('gofast', 'transformers')
# os.makedirs(target_dir, exist_ok=True)

# Create a list of extensions
extensions = [
    Extension( file.replace(".pyx", ""), [file],
              include_dirs=[numpy.get_include()]) for file in pyx_files
]

# Setup configuration
setup(
    name="gofast.transformers",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()],
)
