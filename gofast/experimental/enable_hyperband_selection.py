# -*- coding: utf-8 -*-
"""Enables models Selection Optimizer

The API and results of this estimator might change without any deprecation
cycle.

Importing this file dynamically sets :class:`~gofast.models.selection`
as an attribute of the selection module::

    >>> # explicitly require this experimental feature
    >>> from gofast.experimental import enable_hyperband_selection  # noqa
    >>> # now you can import normally from HyperbandSearchCV
    >>> from gofast.models.selection import HyperbandSearchCV
    
Created on Sat Feb  3 20:58:51 2024
@author: LKouadio<etanoyau@gmail.com>
"""

from ..models._deep_selection import HyperbandSearchCV
from ..models import selection 
from .. import models 
from .. import model_selection 

# use settattr to avoid mypy errors when monkeypatching
setattr ( models, 'selection', selection )
setattr ( models.selection, "HyperbandSearchCV", HyperbandSearchCV )
setattr ( model_selection, "HyperbandSearchCV", HyperbandSearchCV )

models.selection.__all__ += ["HyperbandSearchCV"]
model_selection.__all__.extend(["HyperbandSearchCV"])