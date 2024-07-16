# -*- coding: utf-8 -*-
#Created on Sat Feb  7 20:58:51 2024
# @author: LKouadio <etanoyau@gmail.com>

"""Enables GeoIntelligentSystem.

GeoIntelligentSystem class and its methods are part of an experimental 
API that is still under development. As such, they may undergo significant 
changes in future releases of the 'gofast' library.

Importing this file dynamically sets :class:`~gofast.geo.GeoIntelligentSystem`
as an attribute of the impute module::

    >>> # explicitly require this experimental feature
    >>> from gofast.experimental import enable_geo_intel_system  # noqa
    >>> # now we can import normally from geo.system
    >>> from gofast.geo  import GeoIntelligentSystem

"""

from ..geo.system import GeoIntelligentSystem
from .. import geo 

# use settattr to avoid mypy errors when monkeypatching
setattr(geo, "GeoIntelligentSystem" , GeoIntelligentSystem )

geo.__all__ += ["GeoIntelligentSystem"]