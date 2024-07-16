# -*- coding: utf-8 -*-

"""
This module provides compatibility tools for bridging differences between 
Python 2 and 3, enhancing cross-version code maintainability. It includes 
utility functions for type conversions, list-producing versions of built-in 
Python functions, and mechanisms for class creation with metaclasses across
Python versions.

Functions:

- asbytes(s): Converts a string or bytes object to bytes. For string inputs,
  it encodes them to bytes using 'latin1' encoding, facilitating handling of
  byte data.

- asstr(s): Converts a string or bytes object to a string. For bytes inputs, 
  it decodes them to a string using 'latin1' encoding, ensuring text data 
  is properly handled.

- lrange(*args, **kwargs): A list-producing version of `range()`, returning a list 
  instead of a range object for compatibility with code expecting a list.

- lzip(*args, **kwargs): A list-producing version of `zip()`, combining several 
  iterables into a list of tuples, where each tuple contains elements from all 
  iterables at the same position.

- lmap(*args, **kwargs): A list-producing version of `map()`, applying a given
  function to all items in an input list and returning a list of the results.

- lfilter(*args, **kwargs): A list-producing version of `filter()`, filtering
  elements from an iterable for which a function returns true and returning
  a list of the results.

- with_metaclass(meta, *bases): A utility function for class creation with a
  metaclass, addressing compatibility issues between Python 2 and 3's metaclass
  syntax.

Compatibility:

- Provides a forward-compatible `Literal` type from the `typing` module for 
  Python versions that support it (>=3.8), using the `typing_extensions` 
  module as a fallback.

Notes:

- The `asunicode` function has been unified under Python 3's `str` type, reflecting 
  the transition away from separate unicode and string types in Python 2.

- These tools are designed to simplify the development process across Python versions, 
  minimizing the need for version-specific code paths and facilitating cleaner, more 
  readable codebases.
"""
import sys
from typing import TYPE_CHECKING

# Detect Python version for conditional compatibility
PY37 = sys.version_info[:2] == (3, 7)

# Define string and byte conversion functions
# Convert anything to string, ignoring the second parameter
asunicode = lambda x, _: str(x)  

# Functions to be exposed in the module's public interface
__all__ = [
    "asunicode",
    "asstr",
    "asbytes",
    "Literal",
    "lmap",
    "lzip",
    "lrange",
    "lfilter",
    "with_metaclass",
]

def asbytes(s):
    """Converts strings to bytes, assuming 'latin1' encoding 
    for string inputs."""
    if isinstance(s, bytes):
        return s
    return s.encode("latin1")

def asstr(s):
    """Converts bytes to strings, assuming 'latin1' encoding for 
    bytes inputs."""
    if isinstance(s, str):
        return s
    return s.decode("latin1")

# list-producing versions of the major Python iterating functions
def lrange(*args, **kwargs):
    """Equivalent to range(), but returns a list."""
    return list(range(*args, **kwargs))

def lzip(*args, **kwargs):
    """Equivalent to zip(), but returns a list of tuples."""
    return list(zip(*args, **kwargs))

def lmap(*args, **kwargs):
    """Equivalent to map(), but returns a list of the results."""
    return list(map(*args, **kwargs))

def lfilter(*args, **kwargs):
    """Equivalent to filter(), but returns a list of items for which the
    function returns True."""
    return list(filter(*args, **kwargs))

def with_metaclass(meta, *bases):
    """Facilitates class creation with a metaclass across Python versions."""
    class metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, "temporary_class", (), {})

# Compatibility handling for Literal type from typing module
if sys.version_info >= (3, 8):
    from typing import Literal
elif TYPE_CHECKING:
    from typing_extensions import Literal
else:
    from typing import Any as Literal
