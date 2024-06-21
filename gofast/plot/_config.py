# -*- coding: utf-8 -*-

# config.py in the gofast.plot package

class PlotConfig:
    """
    Configuration class to manage settings for handling plot dependencies
    within the plotting system.

    This class provides static variables to configure global settings for
    automatic installation of dependencies required by the plotting functions,
    and to determine the installation method (pip or conda).

    Attributes
    ----------
    install_dependencies : bool
        If set to True, enables automatic installation of missing packages
        required by plot functions. Defaults to False.
    use_conda : bool
        If set to True, uses conda for automatic installation, otherwise uses pip.
        Applies only if `install_dependencies` is True. Defaults to False.

    Examples
    --------
    To enable automatic installation of dependencies using pip:

    >>> from gofast.plot._config import PlotConfig
    >>> PlotConfig.install_dependencies = True
    >>> PlotConfig.use_conda = False

    To enable automatic installation using conda:

    >>> PlotConfig.install_dependencies = True
    >>> PlotConfig.use_conda = True
    """
    install_dependencies = False
    use_conda = False
