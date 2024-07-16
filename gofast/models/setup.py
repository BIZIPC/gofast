# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from setuptools import Extension
# from gofast.util import make_extensions

# # List of files or modules to be converted and compiled
# files_or_modules = [
#     'gofast/models/optimize.py',
#     'gofast.models.another_module'
# ]

# # Construct the extension modules
# extensions = make_extensions(files_or_modules, rename=False, verbose=True)

# # Setup configuration
# setup(
#     name="gofast_models_optimize",
#     ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
#     include_dirs=[np.get_include()],
# )

# Explicitly list .pyx files if needed
pyx_files = [
    'optimize.pyx',
    # Add other .pyx files explicitly if needed
]

ext_modules = [
    Extension(pyx_file.replace('/', '.').replace('.pyx', ''),
              [pyx_file], include_dirs=[np.get_include()])
    for pyx_file in pyx_files
]

setup(
    name="gofast_models_optimize",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
)