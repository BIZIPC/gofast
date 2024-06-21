# -*- coding: utf-8 -*-
# File: gofast/model_selection.py
"""
This module provides simplified access to the model selection tools available
in the :mod:`gofast.models.selection` module. It allows for direct imports of key
model selection strategies into user space, streamlining the import process
and enhancing code readability.

Use this module to import model selection classes like :class:`SwarmSearchCV`, 
:class:`GradientSearchCV`, and others directly into your projects. It acts as 
a facade over the more complex structure within the :mod:`gofast.models.selection`
module, making these tools more accessible.

Additionally, the module guards experimental features such as 
:class:`HyperbandSearchCV`, ensuring they are accessed only through explicit 
experimental pathways, safeguarding against inadvertent use of unstable APIs.
"""

# Import the necessary classes from the original module
from gofast.models.selection import (
    SwarmSearchCV,
    GradientSearchCV,
    AnnealingSearchCV,
    GeneticSearchCV,
    EvolutionarySearchCV,
    SequentialSearchCV
)

# Define what is available to import from this module when using
# 'from gofast.model_selection import *'
__all__ = [
    "SwarmSearchCV",
    "GradientSearchCV",
    "AnnealingSearchCV",
    "GeneticSearchCV",
    "EvolutionarySearchCV",
    "SequentialSearchCV",
]

import typing
if typing.TYPE_CHECKING:
    from ._deep_selection import HyperbandSearchCV  # noqa

def __getattr__(name):
    """
    Intercept attribute access attempts for undefined attributes in this module.
    This method is primarily used to manage the exposure of experimental features,
    ensuring that they are only used consciously and explicitly by the end-users.

    This function is invoked automatically when an attribute that does not exist
    in the module's namespace is accessed. The primary use in this context is to
    throw informative errors when users attempt to access certain experimental
    features, guiding them towards the correct way to access these features safely
    and deliberately.

    Parameters:
    - name (str): The name of the attribute being accessed.

    Raises:
    - ImportError: If the attribute name matches an experimental feature that
      requires explicit enabling. This error instructs the user on how to properly
      enable access to the feature.
    - AttributeError: If the attribute name does not correspond to any handleable
      case, indicating that the attribute genuinely does not exist in the module.

    Returns:
    - None: This function does not return. It either raises an exception or would
      result in a continuation of the attribute resolution process if it were not
      to handle the attribute.

    Example:
    Trying to access `HyperbandSearchCV` directly from the module without proper
    enabling will result in an ImportError with instructions on how to enable it:

        >>> from gofast.model_selection import HyperbandSearchCV
        ImportError: HyperbandSearchCV is experimental and the API might change
        without any deprecation cycle. To use it, you need to explicitly import
        `enable_hyperband_selection`:
        `from gofast.experimental import enable_hyperband_selection`

    Notes:
    - This method is part of Python's module handling system and is only called
      during attribute lookup when normal attribute access fails.
    - Using `__getattr__` for managing experimental features allows for a dynamic
      response to attribute access, which can be tailored to provide additional
      information and control the usage patterns of the module.
    """
    if name == "HyperbandSearchCV":
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            "deprecation cycle. To use it, you need to explicitly import "
            "`enable_hyperband_selection`:\n"
            "`from gofast.experimental import enable_hyperband_selection`"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
