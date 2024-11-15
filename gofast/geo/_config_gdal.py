# -*- coding: utf-8 -*-
"""
Configures GDAL and PyProj for the application. It checks for the 
availability of GDAL and PyProj libraries, determines the version of GDAL if 
available, and loads EPSG codes from a local file or a numpy backup, ensuring
 the application can correctly handle geospatial data.

Attributes:
    HAS_GDAL (bool): Indicates if GDAL is available.
    NEW_GDAL (bool): True if the GDAL version is 3 or higher.
    HAS_PROJ (bool): True if PyProj is available.
    EPSG_DICT (dict): Dictionary of EPSG codes mapped to their definitions.
    
Created on Sat Feb  3 21:33:42 2024
@author: LKouadio <etanoyau@gmail.com>
"""

import os
import re
import warnings

import numpy as np

def suppress_warnings():
    """
    Suppresses UserWarning warnings.

    This function is used to ignore specific warnings, particularly from the 
    GDAL and PyProj libraries, that do not affect the application's functionality.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

def check_gdal_availability():
    """
    Checks for the availability and version of GDAL.

    Returns:
        tuple: A tuple containing two boolean values. The first indicates if 
        GDAL is available, and the second if the GDAL version is 3 or higher.
    """
    from ..decorators import gdal_data_check
    try:
        has_gdal = gdal_data_check(None)._gdal_data_found
        new_gdal = False
        if has_gdal:
            from osgeo import __version__ as osgeo_version
            if int(osgeo_version[0]) >= 3:
                new_gdal = True
        return has_gdal, new_gdal
    except ImportError:
        return False, False

def check_proj_availability():
    """
    Checks for the availability of PyProj.

    Returns:
        bool: True if PyProj is available, False otherwise.
    """
    try:
        import pyproj
        pyproj.Proj(init="epsg:4326")  # Example use to ensure pyproj works
        return True
    except ImportError:
        return False

def load_epsg_codes(has_proj):
    """
    Loads EPSG codes from the PyProj data directory or a numpy backup file.

    Parameters:
        has_proj (bool): Indicates if PyProj is available.

    Returns:
        dict: A dictionary mapping EPSG codes to their definitions.
    """
    EPSG_DICT = {}
    if has_proj:
        try:
            import pyproj
            epsg_file_path = os.path.join(pyproj.pyproj_datadir, 'epsg')
            with open(epsg_file_path, 'r') as f:
                for line in f:
                    if '#' in line or not re.search('<(\d+)>', line):
                        continue
                    epsg_code, epsg_string = parse_epsg_line(line)
                    if epsg_code:
                        EPSG_DICT[epsg_code] = epsg_string
        except Exception:
            EPSG_DICT = load_epsg_from_backup()
    return EPSG_DICT

def parse_epsg_line(line):
    """
    Parses a single line from an EPSG file to extract the EPSG code and definition.

    Parameters:
        line (str): A line from an EPSG file.

    Returns:
        tuple: A tuple containing the EPSG code (int) and its definition (str) if found,
        otherwise (None, None).
    """
    epsg_code_val = re.findall('<(\d+)>', line)
    if epsg_code_val and epsg_code_val[0].isdigit():
        epsg_code = int(epsg_code_val[0])
        epsg_string = re.findall('>(.*)<', line)[0].strip()
        return epsg_code, epsg_string
    return None, None

def load_epsg_from_backup():
    """
    Loads EPSG codes from a numpy backup file.

    Returns:
        dict: A dictionary mapping EPSG codes to their definitions, loaded 
        from a numpy file.

    Raises:
        RuntimeError: If the numpy file could not be found or loaded.
    """
    try:
        path = os.path.dirname(os.path.abspath(__file__))
        epsg_dict_fn = os.path.join(path, 'epsg.npy')
        return np.load(epsg_dict_fn, allow_pickle=True).item()
    except Exception as e:
        raise RuntimeError("Failed to load EPSG codes from backup file.") from e

# Main execution block
if __name__ == "__main__":
    suppress_warnings()
    HAS_GDAL, NEW_GDAL = check_gdal_availability()
    HAS_PROJ = check_proj_availability() or HAS_GDAL  # Assume PyProj is available if GDAL is
    if not (HAS_GDAL or HAS_PROJ):
        raise RuntimeError("Either GDAL or PyProj must be installed and functional.")
    
    EPSG_DICT = load_epsg_codes(HAS_PROJ)

