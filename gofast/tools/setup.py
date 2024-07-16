# -*- coding: utf-8 -*-
"""
Cythonize  "_openmp_helpers" 
=============================
we have choice to use 'pyximport' (Cython Compilation for Developers) or setup 
configuration. the latter one as recommended so. For further details 
refer to  http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html

"""
#import pyximport; pyximport.install(pyimport=True)
# from numpy.distutils.misc_util import Configuration

# def configuration(parent_package="", top_path=None):
#     """ Cythonize _openmp_helpers """
#     config = Configuration("utils", parent_package, top_path)
    
#     libraries=[]
#     config.add_extension(
#       "_openmp_helpers", sources=["_openmp_helpers.pyx"], libraries=libraries
#       )
#     config.add_subpackage("tests")
    
#     return config 

# if __name__ == "__main__":
#     from numpy.distutils.core import setup

#     setup(**configuration(top_path="").todict())

from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from setuptools import Extension
# Explicitly list .pyx files if needed
pyx_files = [
    'coreutils.pyx',
    'mlutils.pyx', 
    'validator.pyx', 
    'mathex.pyx', 
    'funcutils.pyx',
    'baseutils.pyx', 
    
    # Add other .pyx files explicitly if needed
]

ext_modules = [
    Extension(pyx_file.replace('/', '.').replace('.pyx', ''),
              [pyx_file], include_dirs=[np.get_include()])
    for pyx_file in pyx_files
]

setup(
    name="gofast_tools",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3", 'binding': True}, annotate=True),
    include_dirs=[np.get_include()],
)
