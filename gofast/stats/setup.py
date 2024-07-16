# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from setuptools import Extension
# Explicitly list .pyx files if needed
pyx_files = [
    'descriptive.pyx',
    'utils.pyx'
    # Add other .pyx files explicitly if needed
]

ext_modules = [
    Extension(pyx_file.replace('/', '.').replace('.pyx', ''),
              [pyx_file], include_dirs=[np.get_include()])
    for pyx_file in pyx_files
]

setup(
    name="gofast_tools",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
)