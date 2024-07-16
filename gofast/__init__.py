# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
GOFast: Accelerate Your Machine Learning Workflow
=================================================

:code:`gofast` is designed to streamline and accelerate every step of your 
data science workflow, enhancing productivity, ease of use, and community-driven
improvements.
"""
import os
import logging
import warnings
import importlib

# Configure basic logging and suppress certain third-party library warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True

# Dynamic import function
def _lazy_import(module_name, alias=None):
    """Lazily import a module to reduce initial package load time."""
    def _lazy_loader():
        return importlib.import_module(module_name)
    if alias:
        globals()[alias] = _lazy_loader
    else:
        globals()[module_name] = _lazy_loader

# Define the version
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0"

# Dependency check
_required_dependencies = [
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("matplotlib", None),
    ("seaborn", None),
    ("tqdm", None), 
    ("sklearn", "scikit-learn"),
    ("statsmodels", None)
]

_missing_dependencies = []
for package, import_name in _required_dependencies:
    try:
        if import_name:
            _lazy_import(import_name, package)
        else:
            _lazy_import(package)
    except ImportError as e:
        _missing_dependencies.append(f"{package}: {str(e)}")

if _missing_dependencies:
    warnings.warn("Some dependencies are missing. GOFast may not function correctly:\n" +
                  "\n".join(_missing_dependencies), ImportWarning)

# Suppress FutureWarnings if desired, but allow users to re-enable them
_warnings_state = {"FutureWarning": "ignore"}
def suppress_warnings(suppress=True):
    """Function to suppress or re-enable future warnings."""
    for warning, action in _warnings_state.items():
        if suppress:
            warnings.filterwarnings(action, category=FutureWarning)
        else:
            warnings.filterwarnings("default", category=FutureWarning)

suppress_warnings()

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Setup logging configuration
from .util import setup_logging
setup_logging()

# Public API flag
PUBLIC = False

def check_public_api():
    """Check if public API should be made available."""
    global __all__
    if config._set_public:
        from . import _public  # noqa
        from .assistance import assist_me, gofast_explorer as explore
        __all__.extend(["assist_me", "explore"])
        globals().update({"assist_me": assist_me, "explore": explore})
        # print("Public API has been enabled.")
        warnings.warn("Public API has been enabled.", UserWarning)

# Property to automatically check public API when __set_public__ changes
class GoFastConfig:
    def __init__(self):
        self._set_public = False

    @property
    def PUBLIC(self):
        return self._set_public

    @PUBLIC.setter
    def PUBLIC(self, value):
        self._set_public = value
        check_public_api()

config = GoFastConfig()

# Update the module to use the new property
__builtins__['PUBLIC'] = config.PUBLIC

__doc__ += f"\nVersion: {__version__}\n"

# Public API
__all__ = ["setup_logging", "__version__", "check_public_api"]

# Make sure to extend the `__all__` list correctly
check_public_api()
