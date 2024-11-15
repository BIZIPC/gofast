# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Track bugs and issues, and handle all GOFast exceptions.

This module provides a logging utility class `gofastlog` to configure and manage
logging across the `gofast` package. It supports various logging configurations,
including YAML and INI formats, and offers methods to set up default loggers,
retrieve named loggers, and configure log file outputs.

The module also includes helper functions to set up logging with environment
variable substitutions, enhancing flexibility in different deployment scenarios.
"""

import os
import yaml
import logging
import logging.config
from string import Template
from typing import Optional

__all__ = ["gofastlog"]


class gofastlog:
    """
    A class to configure logging for the `gofast` module, facilitating the tracking
    of all exceptions and log messages within the system.
    """

    @staticmethod
    def load_configuration(
        config_path: Optional[str] = None,
        use_default_logger: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Configures logging based on a specified configuration file.

        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file. Supports `.yaml`, `.yml`, `.ini`, 
            and `.json` formats. If `None` or not provided, uses basic logging 
            configuration or a default logger setup, depending on 
            `use_default_logger`.
        
        use_default_logger : bool, optional
            Whether to use the default logger configuration if no `config_path` 
            is provided. Defaults to `True`.
        
        verbose : bool, optional
            If `True`, prints additional information during configuration. 
            Defaults to `False`.

        Raises
        ------
        FileNotFoundError
            If the specified configuration file does not exist.
        
        ValueError
            If the configuration file format is unsupported.
        """
        if not config_path:
            if use_default_logger:
                gofastlog.set_default_logger()
            else:
                logging.basicConfig()
            return

        if verbose:
            print(f"Configuring logging with: {config_path}")

        if config_path.endswith((".yaml", ".yml")):
            gofastlog._configure_from_yaml(config_path, verbose)
        elif config_path.endswith(".ini"):
            logging.config.fileConfig(config_path, disable_existing_loggers=False)
        else:
            logging.warning(
                f"Unsupported logging configuration format: {config_path}"
            )

    @staticmethod
    def _configure_from_yaml(yaml_path: str, verbose: bool = False) -> None:
        """
        Configures logging from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file.
        
        verbose : bool, optional
            If `True`, prints additional information during configuration.
            Defaults to `False`.

        Raises
        ------
        FileNotFoundError
            If the YAML configuration file does not exist.
        
        yaml.YAMLError
            If there is an error parsing the YAML file.
        """
        full_path = os.path.abspath(yaml_path)
        if not os.path.exists(full_path):
            logging.error(f"The YAML config file {full_path} does not exist.")
            raise FileNotFoundError(f"The YAML config file {full_path} does not exist.")

        if verbose:
            print(f"Loading YAML config from {full_path}")

        try:
            with open(full_path, "rt") as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML config file: {e}")
            raise

    @staticmethod
    def set_default_logger() -> None:
        """
        Sets up a default logger configuration for basic logging needs.
        
        This default configuration logs messages with level INFO and above to the
        console with a simple format.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @staticmethod
    def get_gofast_logger(logger_name: str = '') -> logging.Logger:
        """
        Retrieves a logger with a specified name.

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
        return logging.getLogger(logger_name)

    @staticmethod
    def load_configure_set_logfile(
        config_file: str = '_gflog.yml',
        app_name: str = 'gofast'
    ) -> None:
        """
        Configures logging from a YAML file located in a specific directory 
        defined by an environment variable.

        Parameters
        ----------
        config_file : str, optional
            The name of the configuration file. Defaults to `'_gflog.yml'`.
        
        app_name : str, optional
            The application name used to derive the path from environment 
            variables. Defaults to `'gofast'`.

        Raises
        ------
        EnvironmentError
            If the environment variable for the log config path is not set.
        
        FileNotFoundError
            If the specified configuration file does not exist.
        
        ValueError
            If the configuration file format is unsupported.
        """
        env_var = f"{app_name.upper()}_LOG_CONFIG_PATH"
        config_path_env = os.getenv(env_var, '')
        if config_path_env:
            full_path = os.path.join(config_path_env, config_file)
        else:
            raise EnvironmentError(
                f"{env_var} environment variable is not set."
            )

        if not full_path.endswith(('.yaml', '.yml')):
            raise ValueError("Only `.yaml` or `.yml` config files are supported.")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The config file {full_path} does not exist.")

        try:
            with open(full_path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML config file: {e}")
            raise

    @staticmethod
    def set_logger_output(
        log_filename: str = "gofast.log",
        date_format: str = '%Y-%m-%d %H:%M:%S',
        file_mode: str = "w",
        format_: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level: int = logging.DEBUG
    ) -> None:
        """
        Configures the logging output, including the file name, date format, 
        logging level, and message format.

        Parameters
        ----------
        log_filename : str, optional
            The name of the log file. Defaults to `"gofast.log"`.
        
        date_format : str, optional
            The date format used in log messages. Defaults to 
            `'%Y-%m-%d %H:%M:%S'`.
        
        file_mode : str, optional
            The mode for opening the log file (`'a'` for append, `'w'` for 
            overwrite). Defaults to `'w'`.
        
        format_ : str, optional
            The format of the log messages. Defaults to 
            `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`.
        
        level : int, optional
            The logging level (e.g., `logging.DEBUG`, `logging.INFO`). 
            Defaults to `logging.DEBUG`.
        """
        handler = logging.FileHandler(log_filename, mode=file_mode)
        handler.setLevel(level)
        formatter = logging.Formatter(format_, datefmt=date_format)
        handler.setFormatter(formatter)

        logger = gofastlog.get_gofast_logger()
        logger.setLevel(level)
        logger.addHandler(handler)

        # Remove duplicate handlers to prevent repeated log messages.
        logger.handlers = list(set(logger.handlers))


def setup_logging(config_path: str = 'path/to/_gflog.yml') -> None:
    """
    Sets up logging configuration from a YAML file and applies the necessary 
    environment variable overrides for log paths.

    Parameters
    ----------
    config_path : str, optional
        Path to the logging configuration YAML file. Default is 
        `'path/to/_gflog.yml'`.
    
    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    
    yaml.YAMLError
        If there is an error parsing the YAML file.
    """
    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f.read())

    # Use the default path from the YAML file or override it with an environment variable
    log_path = os.getenv('LOG_PATH', config.get('default_log_path', '/fallback/path'))

    # Replace the placeholder with the actual log path in handlers
    if 'handlers' in config:
        for handler in config['handlers'].values():
            if 'filename' in handler:
                handler['filename'] = handler['filename'].replace('${LOG_PATH}', log_path)

    logging.config.dictConfig(config)


def setup_logging_with_template(config_path: str = 'path/to/_gflog.yml') -> None:
    """
    Sets up logging configuration from a YAML file with environment variable 
    placeholders that are replaced using a Template.

    Parameters
    ----------
    config_path : str, optional
        Path to the logging configuration YAML file. Default is 
        `'path/to/_gflog.yml'`.
    
    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    
    KeyError
        If required environment variables for substitution are missing.
    
    yaml.YAMLError
        If there is an error parsing the YAML file.
    """
    # Load the YAML configuration file
    with open(config_path, 'rt') as f:
        config_text = f.read()

    # Substitute environment variable placeholders using Template
    config_text = Template(config_text).substitute(
        LOG_PATH=os.getenv('LOG_PATH', 'default/log/path')
    )

    # Load the substituted configuration as a dictionary
    config = yaml.safe_load(config_text)

    # Configure logging
    logging.config.dictConfig(config)


def setup_logging_with_expandvars(config_path: str = 'path/to/_gflog.yml') -> None:
    """
    Sets up logging configuration from a YAML file with environment variable 
    placeholders that are expanded using `os.path.expandvars`.

    Parameters
    ----------
    config_path : str, optional
        Path to the logging configuration YAML file. Default is 
        `'path/to/_gflog.yml'`.
    
    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    
    yaml.YAMLError
        If there is an error parsing the YAML file.
    """
    # Load the YAML configuration file
    with open(config_path, 'rt') as f:
        config_text = f.read()

    # Substitute environment variable placeholders using expandvars
    config_text = os.path.expandvars(config_text)

    # Load the substituted configuration as a dictionary
    config = yaml.safe_load(config_text)

    # Configure logging
    logging.config.dictConfig(config)


if __name__ == '__main__':
    """
    Entry point for testing the `gofastlog` module.

    Prints the absolute path of the `gofastlog` module.
    """
    print(os.path.abspath(gofastlog.__name__))
