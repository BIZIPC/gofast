# util.py

import os
import sys 
import logging
import logging.config  
import yaml

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
    

def make_public_api (): 
    # make summary API public 
    import gofast._public  # noqa 

make_public_api () 

if __name__ == "__main__":
    setup_logging()


