# util.py

"""provides utility functions for the GOFast package, including logging setup,
file conversion from Python to Cython, and creation of Cython extension modules."""
import os
import sys 
import logging
import logging.config  
import yaml

__all__ = ['create_log_files', 'ensure_logging_directory',
           'load_logging_configuration',  'make_extensions',
          'setup_gofast_logging', 'setup_logging', 'to_pyx' 
          ]

def ensure_logging_directory(log_path):
    """Ensure that the logging directory exists."""
    os.makedirs(log_path, exist_ok=True)

def setup_logging(default_path='_gflog.yml', default_level=logging.INFO):
    """Setup logging configuration with fallback."""
    package_dir = os.path.dirname(__file__)
    log_path = os.environ.get('LOG_PATH', os.path.join(package_dir, 'gflogs'))
    ensure_logging_directory(log_path)
    create_log_files(log_path)  # Ensure log files are created
    config_file_path = os.path.join(package_dir, default_path)
    try:
        load_logging_configuration(config_file_path, default_level)
    except Exception as e:
        logging.basicConfig(level=default_level)
        logging.warning(f"Failed to load logging configuration from {config_file_path}."
                        f" Error: {e}. Using basicConfig with level={default_level}.")

def create_log_files(log_path):
    """Create log files if they do not exist."""
    for log_file in ['infos.log', 'warnings.log', 'errors.log']:
        full_path = os.path.join(log_path, log_file)
        if not os.path.exists(full_path):
            with open(full_path, 'w'):  # This will create the file if it does not exist
                pass

def load_logging_configuration(config_file_path, default_level):
    """Load and interpolate environment variables in logging configuration."""
    if os.path.exists(config_file_path):
        with open(config_file_path, 'rt') as f:
            config = yaml.safe_load(f.read())
            # Interpolate environment variables
            for handler in config.get('handlers', {}).values():
                if 'filename' in handler:
                    handler['filename'] = os.path.expandvars(handler['filename'])
            logging.config.dictConfig(config)
    else:
        raise FileNotFoundError(f"Logging configuration file not found: {config_file_path}")

def setup_gofast_logging (default_path='_gflog.yml'): 
    "Setup gofast logging config YAML file."
    # Only modify sys.path if necessary, avoid inserting unnecessary paths
    package_dir = os.path.dirname(__file__)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
        
    # Set a default LOG_PATH if it's not already set
    os.environ.setdefault('LOG_PATH', os.path.join(package_dir, 'gflogs'))
    
    # Import the logging setup function from _gofastlog.py
    from ._gofastlog import gofastlog
    
    # Define the path to the _gflog.yml file
    config_file_path = os.path.join(package_dir, default_path)
    
    # Set up logging with the path to the configuration file
    gofastlog.load_configuration(config_file_path)
  

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
    setup_logging()


