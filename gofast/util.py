# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""Provides utility functions for the GOFast package, including logging 
setup, file conversion from Python to Cython, and creation of Cython 
extension modules.

The module initializes the logging configuration to ensure
consistent logging across all modules within the package.
"""

import os
import logging
from ._gofastlog import gofastlog


__all__ = [ 'make_extensions', 'to_pyx' , 'initialize_logging', 'get_logger']


# Determine the directory where __init__.py resides
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the default logging configuration file path
DEFAULT_LOG_CONFIG = os.path.join(PACKAGE_DIR, '_gflog.yml')

def initialize_logging(
    config_file: str = DEFAULT_LOG_CONFIG,
    use_default_logger: bool = True,
    verbose: bool = False
) -> None:
    """
    Initializes the logging configuration for the GOFast package.

    This function configures logging based on a YAML configuration file located
    within the package directory. If the configuration file is not found or an
    error occurs during loading, it falls back to a default logger setup.

    Parameters
    ----------
    config_file : str, optional
        Path to the logging configuration YAML file. Defaults to
        `_gflog.yml` located in the package directory.
    
    use_default_logger : bool, optional
        Whether to use the default logger configuration if the specified
        `config_file` is not found or fails to load. Defaults to `True`.
    
    verbose : bool, optional
        If `True`, prints additional information during the logging setup.
        Useful for debugging purposes. Defaults to `False`.
    
    Raises
    ------
    FileNotFoundError
        If the specified `config_file` does not exist and `use_default_logger`
        is set to `False`.
    
    yaml.YAMLError
        If there is an error parsing the YAML configuration file.
    """
    try:
        # Attempt to load and configure 
        # logging using the specified config file
        gofastlog.load_configuration(
            config_path=config_file,
            use_default_logger=use_default_logger,
            verbose=verbose
        )
    except FileNotFoundError as e:
        if use_default_logger:
            logging.warning(
                f"Logging configuration file not found: {config_file}. "
                "Falling back to default logger."
            )
            gofastlog.set_default_logger()
        else:
            raise e
    except Exception as e:
        if use_default_logger:
            logging.error(
                f"Failed to load logging configuration from {config_file}: {e}. "
                "Falling back to default logger."
            )
            gofastlog.set_default_logger()
        else:
            raise e

def get_logger(logger_name: str = '') -> logging.Logger:
    """
    Retrieves a logger with the specified name.

    Parameters
    ----------
    logger_name : str, optional
        The name of the logger. If empty, returns the root logger.
        Defaults to `''`.
    
    Returns
    -------
    logging.Logger
        The logger instance with the specified name.
    """
    return gofastlog.get_gofast_logger(logger_name)

  
def to_pyx(*files_or_modules: str, rename: bool = False, verbose: bool = False):
    for file_or_module in files_or_modules:
        if file_or_module.endswith('.py'):
            python_file = file_or_module
        else:
            # Convert module path to file path
            python_file = file_or_module.replace('.', '/') + '.py'

        if not os.path.exists(python_file):
            raise FileNotFoundError(f"The specified file {python_file} does not exist")

        # Construct the .pyx filename
        pyx_file = python_file.replace('.py', '.pyx')

        # Copy the contents of the .py file to the .pyx file
        with open(python_file, 'r', encoding="utf8" ) as f_py:
            content = f_py.read()

        with open(pyx_file, 'w',  encoding="utf8" ) as f_pyx:
            f_pyx.write(content)

        # Rename (delete) the original .py file if specified
        if rename:
            os.remove(python_file)

        if verbose:
            print(f"Converted {python_file} to {pyx_file}")
            if rename:
                print(f"Deleted the original file {python_file}")


to_pyx.__doc__="""\
Converts Python files to Cython files (.pyx).

Parameters
----------
files_or_modules : str
    The paths to the Python files or the module names. If a file path is
    provided, it must end with `.py`. If a module name is provided,
    it should follow the dot notation, e.g., `gofast.models.optimize`.

rename : bool, optional
    If `True`, deletes the original `.py` file after conversion.
    Default is `False`.

verbose : bool, optional
    If `True`, prints detailed messages about the conversion process.
    Default is `False`.

Returns
-------
None

Examples
--------
>>> from gofast.util import to_pyx
>>> to_pyx('gofast/models/optimize.py', 'gofast.models.another_module', 
           rename=False, verbose=True)
Converted gofast/models/optimize.py to gofast/models/optimize.pyx
Converted gofast/models/another_module.py to gofast/models/another_module.pyx

Notes
-----
The `to_pyx` function converts Python files (`.py`) into Cython files
(`.pyx`) by copying the content of the `.py` file to a new `.pyx` file
with the same name. If the `rename` parameter is `True`, the original
`.py` file is deleted after conversion.

Mathematically, this process can be represented as:
.. math::

    \text{content}(\text{.py}) \rightarrow \text{content}(\text{.pyx})

Where :math:`\rightarrow` denotes the copying of content from the
`.py` file to the `.pyx` file. The original `.py` file is optionally
deleted based on the value of the `rename` parameter.

See Also
--------
- `Cython <https://cython.readthedocs.io/en/latest/>`_ : Official
  Cython documentation.

References
----------
.. [1] Cython, "Cython: C-Extensions for Python." Available:
   https://cython.readthedocs.io/en/latest/
"""

def make_extensions(
        *files_or_modules, rename: bool = False, verbose: bool = False):
    from setuptools import Extension
    import numpy as np
    # Convert Python files to Cython files
    to_pyx(*files_or_modules, rename=rename, verbose=verbose)

    # Construct the extension modules
    extensions = []
    for file_or_module in files_or_modules:
        if file_or_module.endswith('.py'):
            module_path = file_or_module.replace('.py', '')
            module_name = module_path.replace('/', '.')
            pyx_file = file_or_module.replace('.py', '.pyx')
        else:
            module_name = file_or_module
            module_path = file_or_module.replace('.', '/')
            pyx_file = module_path + '.pyx'

        extension = Extension(
            module_name,
            [pyx_file],
            include_dirs=[np.get_include()]
        )
        extensions.append(extension)

    return extensions

make_extensions.__doc__="""\
Converts Python files to Cython files and constructs a list of 
setuptools-Extension objects.

Parameters
----------
files_or_modules : list of str
    The list of paths to the Python files or the module names. If a file path is
    provided, it must end with `.py`. If a module name is provided,
    it should follow the dot notation, e.g., `gofast.models.optimize`.

rename : bool, optional
    If `True`, deletes the original `.py` file after conversion. Default is `False`.

verbose : bool, optional
    If `True`, prints detailed messages about the conversion process. Default is `False`.

Returns
-------
extensions : list of setuptools.Extension
    A list of Extension objects constructed from the given files or modules.

Examples
--------
>>> from gofast.util import make_extensions
>>> files_or_modules = ['gofast/models/optimize.py', 'gofast.models.another_module']
>>> extensions = make_extensions(files_or_modules, rename=False, verbose=True)
>>> for ext in extensions:
>>>     print(ext.name, ext.sources)

Notes
-----
The `make_extensions` function first converts Python files (`.py`) into Cython files
(`.pyx`) using the `to_pyx` function. It then constructs Extension objects for the given
files or modules, preparing them for Cython compilation.

See Also
--------
- `to_pyx` : Converts Python files to Cython files.
- `Cython <https://cython.readthedocs.io/en/latest/>`_ : Official Cython documentation.

References
----------
.. [1] Cython, "Cython: C-Extensions for Python." Available:
   https://cython.readthedocs.io/en/latest/
"""

if __name__ == "__main__":
    # Initialize logging when the package is imported
    initialize_logging()



