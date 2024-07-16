# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides utility functions for data management, simulation adjustments,
distribution checks, and extraction of geographical data.
"""

import re
import copy
import random
import warnings 
import itertools
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Tuple, Optional, Union
import numpy as np 
import pandas as pd 

from ..api.formatter import DescriptionFormatter
from ..api.structures import Boxspace
from ..compat.sklearn import  train_test_split 
from ..tools.coreutils import is_in_if, add_noises_to, validate_noise 
from ..tools.coreutils import smart_format, is_iterable, validate_ratio 
from .metadata import SimulationMetadata

__all__= [
     'apply_scaling',
     'build_continent_country_dict',
     'build_dataset_description',
     'build_distributions_from',
     'build_reserve_details_by_country',
     'check_distributions',
     'extract_minerals_from_countries',
     'extract_minerals_from_regions',
     'fetch_simulation_metadata',
     'find_countries_and_minerals_by_region',
     'find_countries_by_distributions',
     'find_countries_by_minerals',
     'find_countries_by_region',
     'find_mineral_by_country',
     'find_mineral_distributions',
     'find_mineral_location',
     'generate_regression_output',
     'generate_synthetic_values',
     'is_structure_nested',
     'manage_data',
     'manage_nested_lists',
     'rename_data_columns',
     'select_location_for_mineral',
     'validate_noise_level',
  ]

def validate_region(region, mode="strict"):
    """
    Validates and normalizes a region name based on the specified mode.

    Parameters
    ----------
    region : str
        The name of the region to validate and normalize. The region can
        include cardinal points (North, South, East, West) and their variants
        (Northern, Southern, Eastern, Western), followed by the continent name.
    mode : str, optional
        The mode of operation, either 'strict' or 'soft'. In 'strict' mode,
        any cardinal points are removed, and the region is normalized to the
        continent name only. In 'soft' mode, cardinal points are normalized
        and retained in the output. The default is 'strict'.

    Returns
    -------
    str
        The normalized region name, with or without cardinal directions,
        depending on the mode.

    Raises
    ------
    ValueError
        If the region name is not valid based on the mode of operation or if an
        unknown mode is specified.

    Examples
    --------
    >>> from gofast.datasets.util import validate_region 
    
    >>> validate_region("North America", mode="strict")
    'America'
    
    >>> validate_region("Southern Africa", mode="soft")
    'South Africa'
    
    >>> validate_region("Northern America", mode="soft")
    'North America'
    
    >>> validate_region("Western Asia", mode="strict")
    'Asia'
    
    In 'strict' mode, invalid region names raise an error:

    >>> validate_region("Central Europe", mode="strict")
    ValueError: Invalid region name 'Central Europe'. Expected one of the continents.

    In 'soft' mode, unrecognized region parts are ignored, and an attempt is made
    to normalize and retain valid parts:

    >>> validate_region("Eastern Central Europe", mode="soft")
    'East Europe'
    """

    cardinal_mapping = {
        "northern": "north",
        "southern": "south",
        "western": "west",
        "eastern": "east"
    }
    continents = ["africa", "america", "asia", "europe", "oceania"]
    cardinal_points = ["north", "south", "east", "west"]

    # Normalize input
    region_lower = region.lower()
    for variant, cardinal in cardinal_mapping.items():
        region_lower = region_lower.replace(variant, cardinal)
    
    # Split region name to handle cardinal directions
    parts = region_lower.split()

    # Normalize based on mode
    error_msg = f"Invalid region name '{region}'. Expected one of the continents."
    if mode == "strict":
        # Remove cardinal directions if present
        normalized_region = next((part for part in parts if part in continents), None)
        if normalized_region is None:
            raise ValueError(error_msg)
    elif mode == "soft":
        # Keep the cardinal direction but ensure it's the correct form
        normalized_region = " ".join(part.capitalize() for part in parts 
                                     if part in continents + cardinal_points)
        if not normalized_region:
            raise ValueError(error_msg)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'strict' or 'soft'.")

    return normalized_region.title()

def validate_country(name, error="warn"):
    """
    Validates a given country name against a global dictionary of mineral 
    production by country, considering alternative names and abbreviations.

    Parameters
    ----------
    name : str
        The country name to validate. The function is case-insensitive and
        supports common abbreviations and alternative names (e.g., "USA" or
        "U.S.A" for "United States").
    error : str, optional
        Specifies the error handling behavior when the country name is not
        found. Options are "raise" to raise a ValueError, "warn" to issue a
        warning, or "ignore" to silently ignore the error. The default is "warn".

    Returns
    -------
    str
        The validated country name as it appears in the mineral production
        dictionary. Returns an empty string if the name is not found and
        the error parameter is set to "warn" or "ignore".

    Raises
    ------
    ValueError
        If the country name does not exist in the mineral production
        dictionary and the error parameter is set to "raise".

    Examples
    --------
    >>> from gofast.datasets.util import validate_country
    >>> validate_country("USA")
    'United States'

    >>> validate_country("U.S.A")
    'United States'

    >>> validate_country("PRC")
    'China'

    >>> validate_country("UK", error="raise")
    ValueError: The country name 'UK' does not exist in the mineral ...

    >>> validate_country("UK", error="warn")
    UserWarning: The country name 'UK' does not exist in the mineral ...
    ''

    Note
    ----
    This function relies on a predefined mapping of common abbreviations and
    alternative country names to their official names as listed in the global
    dictionary `MINERAL_PROD_BY_COUNTRY`. It is designed to facilitate user
    input by allowing flexibility in how country names are specified.
    """
    from ._globals import MINERAL_PROD_BY_COUNTRY
    map_names = {
        "USA": "United States",
        "PRC": "China",
        "DRC": "Democratic Republic of Congo",
        "IVORY COAST": "Cote d'Ivoire",
        "UAE": "United Arab Emirates"
    }
    # Normalize name to handle common variations and casing
    normalized_name = name.upper().replace('.', '')
    if normalized_name in map_names:
        normalized_name = map_names[normalized_name].upper()

    # Check if the normalized name is in the MINERAL_PROD_BY_COUNTRY keys
    if any(normalized_name == country.upper() 
            for country in MINERAL_PROD_BY_COUNTRY.keys()):
        # Return the proper name from MINERAL_PROD_BY_COUNTRY
        return next((country for country in MINERAL_PROD_BY_COUNTRY 
                      if country.upper() == normalized_name), None)
    else:
        # Handle error based on the 'error' parameter
        message =( 
            f"The country name '{name}' does not "
            "exist in the mineral production dictionary."
            )
        if error == "raise":
            raise ValueError(message)
        elif error == "warn":
            import warnings
            warnings.warn(message)
        elif error == "ignore":
            pass  # Do nothing
        # Return an empty string for 'warn' and 'ignore' if the name is not found
        return ""  

def find_countries_by_region(
    region, 
    country_region_dict=None 
    ):
    """
    Find all countries in the given region. The function is case-insensitive and
    considers partial matches, allowing for flexible searches such as 
    'america' matching both 'North America' and 'South America', or 'Europe'
    matching 'Europe/Asia'.

    Parameters:
    ----------
    region : str
        The region to search for. The search is case-insensitive and supports
        partial matches. Expected regions include 'Africa', 'America', 'Asia',
        'Oceania', and 'Europe', but also recognizes combined regions 
        like 'Europe/Asia'.
        
    country_region_dict : dict
        A dictionary mapping country names (keys) to their respective 
        regions (values). Regions should accommodate combined entries like
        'Europe/Asia' for accurate matching.

    Returns:
    -------
    list
        A list of country names that are located within the specified region 
        or match the region criteria.

    Raises:
    ------
    ValueError
        If the specified region does not partially match any of the region
        entries in the country_region_dict values, indicating an unexpected
        or misspelled region name.

    Examples:
    --------
    >>> find_countries_by_region('America', country_region_dict)
    ['United States', 'Canada', 'Brazil', 'Chile', 'Peru', ...]

    >>> find_countries_by_region('europe', country_region_dict)
    ['Russia', 'Sweden', 'Ukraine', 'Poland', 'Norway', ...]

    Note:
    -----
    The function is designed to be flexible with region names, accommodating 
    scenarios where regions may overlap or be part of a combined region 
    (e.g., 'Europe/Asia'). This is particularly useful for geopolitical regions
    and territories that do not fit neatly into a single continent.
    """
    if country_region_dict is None: 
        from ._globals import COUNTRY_REGION 
        country_region_dict = copy.deepcopy(COUNTRY_REGION)
    
    # Simplifying region matching by only checking for substrings in region names,
    # without pre-validating against a list of normalized regions.
    region_lower = region.lower()
    
    # Compiling a list of unique region parts to avoid errors 
    # on combined or cardinal directions
    unique_region_parts = set(
        part for regions in country_region_dict.values() 
        for part in regions.replace('/', ' ').lower().split()
        )
    # Validate if the provided region or any part of it matches the 
    # known regions or cardinal points
    if not any(part in unique_region_parts for part in region_lower.split('/')):
        raise ValueError(
            f"Unexpected region '{region}'. Please check the region"
            " name for typos. Expected regions include: smart_format(regions)"
            )
    # Filter countries by region, accommodating partial and case-insensitive matches
    countries = [country for country, reg in country_region_dict.items()
                 if region_lower in reg.lower()]
    if len(countries) == 0:
         warnings.warn(f"No countries found for '{region}'. Falling back to"
                       " default: returning all countries.")
         countries = list(country_region_dict.keys())
        
    return countries

def extract_minerals_from_regions(
    regions, 
    detailed_by_region=False, 
    separate_regions_and_minerals=False
):
    """
    Extracts and compiles a list of minerals from specified global regions. 
    This function can return data in different formats based on the input 
    parameters, offering flexibility in how the extracted information is 
    presented.

    Parameters
    ----------
    regions : str or list of str
        A single region name or a list of region names for which to extract 
        mineral information. The function fetches mineral distribution data 
        for these regions.
    detailed_by_region : bool, optional
        If True, returns a detailed list of tuples, each containing a region 
        name and a list of corresponding minerals. Default is False.
    separate_regions_and_minerals : bool, optional
        If True, returns two separate lists: one for all the region names and 
        another for lists of minerals corresponding to each region. Only effective 
        if `detailed_by_region` is False. Default is False.

    Returns
    -------
    list or tuple
        Depending on the parameters, this function can return:
        - A list of all minerals across specified regions if both 
          `detailed_by_region` and `separate_regions_and_minerals` are False.
        - A list of tuples with detailed region-mineral information if 
          `detailed_by_region` is True.
        - A tuple containing two lists (regions and their minerals) if 
          `separate_regions_and_minerals` is True.

    Examples
    --------
    Extracting minerals without details:

    >>> from gofast.datasets.util import extract_minerals_from_regions
    >>> extract_minerals_from_regions(["Europe", "Asia"])
    ['palladium', 'nickel', 'potash', 'uranium', 'coal', 'iron', 'silver']

    Extracting detailed mineral information by region:

    >>> extract_minerals_from_regions(["Europe"], detailed_by_region=True)
    [('Europe', ['palladium', 'nickel', 'potash', 'uranium'])]

    Separating region names and minerals into separate lists:

    >>> regions, minerals = extract_minerals_from_regions(
    ...     ["Europe", "Asia"], separate_regions_and_minerals=True)
    >>> print(regions)
    ['Europe', 'Asia']
    >>> print(minerals)
    [['palladium', 'nickel', 'potash', 'uranium'], ['coal', 'iron', 'silver']]
    """
    if isinstance(regions, str):
        regions = [regions]
    # Fetching distributions for all regions
    mineral_list_dicts = [find_mineral_distributions(
        region, include_region_summary=True) for region in regions]

    if detailed_by_region:
        detailed_info = [(region, minerals[1]) for mineral_info in mineral_list_dicts
                         for region, minerals in mineral_info.items()]
        return detailed_info
        
    elif separate_regions_and_minerals:
        regions_list = [list(mineral_info.keys())[0] for mineral_info in mineral_list_dicts] 
        minerals_lists = [mineral_info[region][1] for mineral_info, region in zip(
            mineral_list_dicts, regions_list)]
        return regions_list, minerals_lists

    else:
        # Extract all minerals and directly split by both ',' and ';' in one step
        split_minerals = itertools.chain.from_iterable(
            # Split each mineral string by ',' then by ';' to handle both cases
            itertools.chain.from_iterable(
                min_str.split(';') for min_str in mineral.split(',')
            ) for mineral in itertools.chain.from_iterable(
                mineral_info[region][1] for mineral_info in mineral_list_dicts 
                for region in mineral_info
            )
        )
        # Remove duplicates and return the list
        return list(set([mineral.strip() for mineral in  split_minerals]))

def check_distributions(distributions, error='warn'):
    """
    Validates the structure of the distributions dictionary to ensure it
    adheres to the expected format: a dictionary with region names as keys
    and lists of minerals as values.

    Parameters
    ----------
    distributions : dict
        The distributions dictionary to validate.
    error : str, optional
        Controls error handling behavior. Accepts 'raise' to raise exceptions,
        'warn' to issue a warning, or 'ignore' to do nothing on failure.
        Default is 'warn'.

    Returns
    -------
    bool
        True if the distributions dictionary fits the expected arrangement,
        otherwise False if 'error' is set to 'ignore' or 'warn'.
    """

    error_message = ""

    if not isinstance(distributions, dict):
        error_message = ( 
            "Distributions must be a dictionary with regions"
            " as keys and lists of minerals as values."
        )

    else:
        for region, minerals in distributions.items():
            if not isinstance(region, str):
                error_message = ( 
                    "All keys in the distributions dictionary must be"
                    " strings representing region names."
                    )
                break
            if isinstance(minerals, str):
                minerals = [minerals]  # Normalize single string to list
            if not isinstance(minerals, list) or not all(
                    isinstance(mineral, str) for mineral in minerals):
                error_message = ( 
                    "Each value in the distributions dictionary must"
                    " be a list of strings representing minerals."
                    )
                break

    if error_message:
        if error == 'raise':
            raise ValueError(error_message)
        elif error == 'warn':
            warnings.warn(error_message)
            return False
        elif error == 'ignore':
            return False
    
    return True
  
def build_distributions_from(
    regions, 
    minerals, 
    error='warn',  
    ):
    """
    Constructs a dictionary mapping each region to its corresponding list of 
    minerals.
    This function ensures that each region is paired with a specific list of 
    minerals, facilitating the creation of detailed regional mineral 
    distributions.

    Parameters
    ----------
    regions : list or str
        A list of region names or a single region name. If a single region name is
        provided, it is automatically transformed into a list. Regions define the
        geographical areas for the mineral distribution mapping.
    minerals : list
        A list of lists, where each sublist contains minerals associated with
        the corresponding region in the `regions` list. If `regions` contains
        only one region, `minerals` can be a single list which will be automatically
        nested into a list of lists.
    error : {'warn', 'ignore', 'raise'}, optional
        Controls the error handling behavior when the lengths of `regions` and
        `minerals` do not match. 'warn' issues a warning and returns `None`,
        'ignore' silently ignores the discrepancy and also returns `None`, and
        'raise' raises a `ValueError`. The default is 'warn'.

    Returns
    -------
    dict or None
        A dictionary where keys are region names and values are lists of minerals
        associated with each region. Returns `None` if there is a discrepancy
        between the lengths of `regions` and `minerals` and `error` is
        set to 'warn' or 'ignore'.

    Raises
    ------
    ValueError
        If `error` is set to 'raise' and the lengths of `regions` and
        `minerals` do not match, indicating an inability to map each region to
        a specific list of minerals.

    Examples
    --------
    Basic usage with matching lengths of `regions` and `minerals`:

    >>> from gofast.datasets.util import build_distributions_from
    >>> build_distributions_from(['Africa', 'Asia'], [['Gold', 'Diamond'], ['Coal']])
    {'Africa': ['Gold', 'Diamond'], 'Asia': ['Coal']}

    Handling a single region and its minerals:

    >>> build_distributions_from('Europe', ['Uranium', 'Nickel'])
    {'Europe': ['Uranium', 'Nickel']}

    Handling mismatched lengths with 'raise':

    >>> build_distributions_from(['Africa', 'Asia'], ['Gold'], error='raise')
    ValueError: Mismatch in list lengths: expected regions and minerals to ...

    Using 'warn' with mismatched lengths (will print a warning and return `None`):

    >>> build_distributions_from(['Africa', 'Asia'], ['Gold'], error='warn')
    UserWarning: Mismatch in list lengths: expected regions and minerals to  ...
    """
    regions = is_iterable(regions, exclude_string=True, transform=True)
    minerals = is_iterable(minerals, exclude_string=True, transform=True)
    
    # Ensuring minerals list is nested to correspond to regions if necessary
    if len(regions) == 1 and not is_structure_nested(minerals, list):
        minerals = [minerals]
    
    # Error handling for mismatched lengths
    if len(regions) != len(minerals):
        message = (f"Mismatch in list lengths: expected regions and minerals "
                   f"to have the same length, but got {len(regions)} region(s)"
                   f" and {len(minerals)} mineral list(s). This discrepancy"
                   f" prevents accurate distribution mapping. Please ensure"
                   " each region is paired with a corresponding list of minerals."
                   )
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
        elif error == 'ignore':
            pass  # Explicitly handling 'ignore' for clarity
        return None
    
    minerals = minerals_normalizer(minerals ,)
    # Building the distributions dictionary
    distributions = {
        region: [mineral] if isinstance ( mineral, str) else mineral 
        for region, mineral in zip(regions, minerals)
    }
    
    return distributions

def minerals_normalizer(minerals, tolist=False, unpack=False):
    """
    Normalize a given input of minerals to lowercase. Can output as a single 
    string, a list of strings, or a flattened list from nested lists depending
    on parameters.

    Parameters:
    ----------
    minerals : str or list
        The input minerals, can be a string, a list of strings, or a nested 
        list of strings.
    tolist : bool, optional
        If True, the output will be a list even if the input is a single string.
        Default is False.
    unpack : bool, optional
        If True and the input is a nested list, the output will be a flattened
        list. Default is False.

    Returns:
    -------
    str or list
        The normalized minerals, either as a lowercase string, a list of 
        lowercase strings, or a flattened list of lowercase strings, 
        based on input parameters.
    """

    # Normalize single string directly
    if isinstance(minerals, str):
        return [minerals.lower()] if tolist else minerals.lower()

    # Handle list input, including nested lists
    if isinstance(minerals, list):
        if unpack:
            # Flatten and normalize nested list
            flattened_list = [item.lower() for sublist in minerals for item in sublist]
            return flattened_list
        else:
            # Normalize list without unpacking
            return [mineral.lower() if isinstance(mineral, str) else
                    [m.lower() for m in mineral] for mineral in minerals]
    # Fallback for unrecognized input types
    return minerals


def manage_nested_lists(elements, mode='unpack', pack_criteria=None):
    """
    A versatile function that either packs elements into nested lists based 
    on a given criterion or unpacks nested lists into a unique, flat list.

    Parameters:
    - elements (list): The list of elements to be either packed or unpacked.
    - mode (str): Operation mode, either 'pack' for creating nested lists or 
      'unpack'
      for flattening nested lists. Default is 'unpack'.
    - pack_criteria (function): A function used as a criterion for packing 
      elements into
      nested lists. Only relevant if mode is 'pack'. It should accept an 
      element as input and return a boolean indicating whether the element 
      belongs to the current nested list.

    Returns:
    - list: Depending on the mode, either a nested list structure or a flat 
      list of unique elements.

    Example Usage:
    >>> manage_nested_lists([1, 2, [3, 4], [5, [6, 7]], 8], mode='unpack')
    [1, 2, 3, 4, 5, 6, 7, 8]
    
    >>> manage_nested_lists([1, 2, 3, 4, 5], mode='pack',
    ...                     pack_criteria=lambda x: x % 2 == 0)
    [[1], [2], [3], [4], [5]]
    """
    elements = is_iterable(elements, exclude_string=True, transform=True )
    mode_lower = str(mode).lower() 
    if mode_lower == 'unpack':
        if not is_structure_nested(elements, check_for='list'):
            return elements 
        # Flatten the list and remove duplicates
        return list(set(itertools.chain.from_iterable(elements) if isinstance(
            elements, list) else [elements]
                       for elements in elements))
    elif mode_lower == 'pack':
        if not pack_criteria:
            raise ValueError(
                "pack_criteria function must be provided when mode is 'pack'.")

        packed_list = []
        for element in elements:
            if isinstance(element, list):
                # Recursively pack sublists
                packed_list.append(manage_nested_lists(
                    element, mode='pack', pack_criteria=pack_criteria))
            elif pack_criteria(element):
                packed_list.append([element])
            else:
                if not packed_list or not pack_criteria(packed_list[-1][-1]):
                    packed_list.append([])
                packed_list[-1].append(element)
        return packed_list
    else:
        raise ValueError("Invalid mode. Choose either 'pack' or 'unpack'.")
        
def is_structure_nested(input_structure, check_for=list):
    """
    Checks if the given structure (list or dict) is nested with specified 
    types.

    Parameters:
    - input_structure (list or dict): The structure to check for nestedness.
    - check_for (str, list, dict, or tuple): Specifies the types to check for
      nesting. Can be 'list', 'dict', the built-in list or dict types, or a 
      tuple containing these types.

    Returns:
    - bool: True if the structure is nested with the specified type(s), 
      False otherwise.

    Example Usage:
    >>> is_structure_nested([1, 2, 3], check_for='list')
    False
    >>> is_structure_nested([1, [2, 3], 4], check_for=list)
    True
    >>> is_structure_nested({'key': {'nested_key': 'value'}}, check_for=dict)
    True
    >>> is_structure_nested([1, {'nested_dict': 2}], check_for='dict')
    True
    >>> is_structure_nested([1, 2, 3], check_for=('dict',))
    False
    """
    type_map = {'list': list, 'dict': dict}
    # Normalize the check_for parameter to handle strings or direct type references
    if not isinstance(check_for, (list, tuple)):
        check_for = [check_for]
    check_for_types = [type_map.get(cf, cf) for cf in check_for]

    def is_nested(element):
        return any(isinstance(element, cf_type) for cf_type in check_for_types)
    
    if isinstance(input_structure, dict):
        return any(is_nested(value) for value in input_structure.values())
    
    if isinstance(input_structure, list):
        return any(is_nested(element) for element in input_structure)
    
    return False

def extract_minerals_from_countries(
    mineral_prod_by_country=None,
    use_default=False,
    split_minerals=True,
):
    """
    Extracts mineral reserves information from a provided dictionary mapping
    countries to their mineral production and reserves information, or from a
    default global dictionary. The function can parse the mineral reserves
    from structured textual descriptions and optionally split the extracted
    minerals into a list.

    Parameters
    ----------
    mineral_prod_by_country : dict, optional
        A dictionary where each key is a country name and each value is a list
        of strings or a single string describing the mineral production and
        reserves of that country. Minerals are expected to be within square
        brackets in the first element of the list for the default dictionary.
        If None and `use_default` is True, attempts to import a default
        dictionary from a global module.
    use_default : bool, optional
        Indicates whether to use the default mineral production dictionary if
        `mineral_prod_by_country` is not provided. Default is True.
    split_minerals : bool, optional
        If True, splits the extracted mineral reserves string into a list based
        on commas or semicolons. Default is True.

    Returns
    -------
    dict
        A new dictionary where each key is a country name and the corresponding
        value is a string or list of the mineral reserves for that country,
        extracted and optionally split from the provided descriptions.

    Raises
    ------
    ImportError
        If `mineral_prod_by_country` is None, `use_default` is True, and the
        global dictionary `MINERAL_PROD_BY_COUNTRY` cannot be imported.

    Examples
    --------
    Using the default dictionary and extracting minerals as a list:

    >>> from gofast.datasets.util import extract_minerals_from_countries
    >>> extract_minerals_from_countries()
    {'Australia': ['iron ore', 'lithium'], 'China': ['coal', 'rare earth elements']}

    Providing a custom dictionary and extracting minerals as a string:

    >>> mineral_prod_by_country = {
    ...     "Australia": "Vast reserves of bauxite, iron ore, lithium",
    ...     "China": "Large reserves of coal; rare earth elements"
    ... }
    >>> extract_minerals_from_countries(mineral_prod_by_country, 
    ...                                  use_default=False, split_minerals=False)
    {'Australia': 'iron ore, lithium', 'China': 'coal, rare earth elements'}

    Providing a custom dictionary and splitting minerals:

    >>> extract_minerals_from_countries(mineral_prod_by_country, split_minerals=True)
    {'Australia': ['bauxite', 'iron ore', 'lithium'], 
     'China': ['coal', 'rare earth elements']}

    Notes
    -----
    This function is particularly useful for quickly extracting key data points
    from structured textual data where specific information is enclosed in
    square brackets or separated by commas/semicolons.
    """
    if mineral_prod_by_country is None: 
        mineral_prod_by_country = get_mineral_production_by_country()
        use_default=True 
        
    country_minerals_dict = {}
    
    for country, info_list in mineral_prod_by_country.items():
        if use_default:
            minerals_text = info_list[0]
            minerals = minerals_text[minerals_text.find('[') + 1: minerals_text.find(']')]
        else:
            if isinstance(info_list, list):
                info_list = ', '.join(info_list)  # Join list elements into a single string
            minerals = info_list
        
        # If 'split_minerals' is True, split the minerals string into a list,
        # considering ',' or ';'
        if split_minerals:
            separators = (',', ';')
            for sep in separators:
                if sep in minerals:
                    minerals = [mineral.strip() for mineral in minerals.split(sep)]
                    break  # Break after the first separator is found and used
        if split_minerals and isinstance ( minerals, str): 
            minerals =[minerals] # put mineral on list even single 
        country_minerals_dict[country] = minerals

    return country_minerals_dict


def select_location_for_mineral(
    mineral,
    mineral_country_mapping=None,
    fallback_countries=None,
    selected_region=None, 
    error='ignore', 
    substitute_for_missing='Unknown'
):
    """
    Selects a random country that produces a specified mineral.
    
    The function can operate based on a provided mapping of minerals to 
    producing countries or fallback to a predefined list of countries if the 
    mapping is not provided or the mineral is not found.

    Parameters
    ----------
    mineral : str
        The name of the mineral for which to find a producing country.
    mineral_country_mapping : dict, optional
        A dictionary mapping minerals (str) to lists of countries (list of str)
        that produce them.
        If not provided, `fallback_countries` is used instead. Default is None.
    fallback_countries : list of str or str, optional
        A list of countries to randomly choose from if `mineral_country_mapping`
        is not provided or the mineral is not found within it. Can also be 
        a single country as a string.
        Default is None.
    selected_region : str, optional
        The geographical region within which to limit the country selection. 
        This parameter is considered only if `mineral_country_mapping` is provided. 
        Default is None.
    error : str, optional
        Determines how to handle cases when the mineral is not found in 
        `mineral_country_mapping`. Options are 'ignore', 'warn', and 'raise'.
        Default is 'ignore'.
    substitute_for_missing : str, optional
        The substitute value to return when the mineral is not found and 
        `error_handling` is set to 'ignore' or 'warn'. Default is 'Unknown'.

    Returns
    -------
    str
        A random country that produces the specified mineral. 
        Returns `substitute_for_missing` if the mineral is not found and 
        error handling is set to 'ignore' or 'warn'.

    Raises
    ------
    ValueError
        If both `mineral_country_mapping` and `fallback_countries` are None,
        or if `error_handling` is set to 'raise' and the mineral is not 
        found in the provided mapping.

    Examples
    --------
    Using a mineral-country mapping:

    >>> from gofast.datasets.util import select_location_for_mineral
    >>> mineral_country_mapping = {'gold': ['Canada', 'South Africa', 'Australia']}
    >>> select_location_for_mineral(
    ... 'gold', mineral_country_mapping=mineral_country_mapping)
    'Canada'  # Randomly selected from the list

    Fallback to a list of countries when the mapping is not provided:

    >>> select_location_for_mineral('gold', fallback_countries=['Canada', 'France'])
    'France'  # Randomly selected from the fallback list

    Handling missing minerals with warning:

    >>> select_location_for_mineral(
    ... 'diamond', mineral_country_mapping=mineral_country_mapping,
    ...  error='warn', substitute_for_missing='None')
    Warning: Mineral 'diamond' not found. Returning substitute location: 'None'.
    'None'
    """
    from ._globals import COUNTRY_REGION
    mineral_country_mapping = ( 
        mineral_country_mapping if mineral_country_mapping else None 
    )

    # Ensure either mineral_country_mapping or fallback_countries is provided
    if mineral_country_mapping is None and fallback_countries is None:
        raise ValueError("Either mineral_country_mapping or "
                         "fallback_countries must be provided.")
    
    # Normalize the mineral name to lowercase
    mineral_lower = str(mineral).lower()
    
    # Randomly select a mineral if it's listed with multiple options (';' or ',')
    if ';' in mineral_lower or ',' in mineral_lower:
        delimiter = ';' if ';' in mineral_lower else ','
        mineral_lower = np.random.choice(mineral_lower.split(delimiter)).strip()
    
    # If only fallback_countries is provided, select a location from them
    if mineral_country_mapping is None:
        if isinstance(fallback_countries, str):
            fallback_countries = [fallback_countries]
        selected_location = np.random.choice(fallback_countries)
        selected_region = COUNTRY_REGION.get(
            selected_location, substitute_for_missing)
        return selected_region, selected_location
    
    # Check if the mineral is in the mapping
    if mineral_lower not in {key.lower() for key in mineral_country_mapping}:
        if error == 'raise':
            raise ValueError(f"Mineral '{mineral}' not found in "
                             "the provided mineral-country mapping.")
        elif error == 'warn':
            print(f"Warning: Mineral '{mineral}' not found. Returning"
                  f" substitute location: '{substitute_for_missing}'.")
        return 'Unknown', substitute_for_missing
    # Select and return a random country from the list associated with the mineral
    return _select_location_from (
        mineral, selected_region, mineral_country_mapping, 
        error_handling= error,
        substitute_for_missing= substitute_for_missing
        )

def _select_location_from(
    mineral, selected_region, mineral_country_mapping, 
    error_handling='ignore',
    substitute_for_missing='Unknown'
    ):
    """
    Selects a random location from countries that produce the specified 
    mineral within a selected region.
    """
    # Normalize the mineral name to lowercase for consistency
    mineral_lower = mineral.lower()
    # Ensure the mineral is present in the mapping; otherwise, return an empty list
    mineral_countries_list = mineral_country_mapping.get(mineral_lower, [])
    
    selected_region = validate_region(selected_region , mode='soft')
    if selected_region is None: 
        # randomly select a location from mineral countries list 
        return np.random.choice(mineral_countries_list)
    # Build the continent to country mapping
    region_map_list = build_continent_country_dict()

    # Extract the list of countries for the selected region
    # Capitalize for matching
    countries_list = region_map_list.get(selected_region.capitalize(), [])  
    # Calculate the intersection if a region is specified and intersection is possible
    intersection_list =[]
    if selected_region and mineral_countries_list:
        intersection_list = list(set(mineral_countries_list) & set(countries_list))
        # Handle cases with no intersection
        if not intersection_list:
            message = f"No countries found for '{mineral}' in region '{selected_region}'."
            if error_handling == 'warn':
                print(f"Warning: {message}" )
                # then find a country location in this mineral in the location 
            elif error_handling == 'raise':
                raise ValueError(message)
            # Fallback to the mineral's countries list if 'ignore'
   
            selected_country , intersection_list = find_mineral_location(
                mineral, excluded_region= selected_region, 
                error_handling=error_handling , 
                substitute_for_missing= substitute_for_missing
                ) 
            intersection_list = mineral_countries_list
   
    # If there's no country to select (list is empty), return None or a meaningful default
    if not intersection_list:
        return "Unknown",  [get_default_location(strategy='global_hq' )] 
    
    # Randomly select and return a country from the intersection list
    return selected_region, np.random.choice(intersection_list)

def find_mineral_location(
    mineral, 
    excluded_region=None,
    mineral_prod_by_country=None, 
    error_handling='warn', 
    region_preference='change', 
    substitute_for_missing='Unknown'
    ):
    continents = ['Africa', 'Europe', "America", "Asia", "Oceania"]
    
    if excluded_region: 
        excluded_region = validate_region(excluded_region)
    # Determine the regions to consider based on exclusions
    
    regions_to_consider = [continent for continent in continents 
                           if continent != excluded_region]

    mineral = mineral.lower()
    potential_locations = {}

    for region in regions_to_consider:
        distributions = find_mineral_distributions(region, detailed_by_country=True)
        # Skip the region if no distribution data is found
        if not distributions:
            continue

        region_countries = distributions.get(region, {})
        for country, c_minerals in region_countries.items():
            # Normalize the list of minerals for comparison
            c_minerals = [m.lower() for m in minerals_normalizer(c_minerals, tolist=True)]
            if mineral in c_minerals:
                potential_locations[region] = country.capitalize()
                break  # Found a location, no need to search more in this region

    # Handle the case when no potential locations are found
    if not potential_locations:
        message =f"Mineral '{mineral}' not found on Earth in our actual database."
        if error_handling == 'warn':
            print(f"Warning: {message}")
        elif error_handling == 'raise':
            raise ValueError(message)
        return "Unknown", substitute_for_missing

    # Randomly select one of the potential locations
    selected_region = np.random.choice(list(potential_locations.keys()))
    selected_country = potential_locations[selected_region]

    # Handle region status preferences
    if region_preference.lower() == 'change':
        final_region = selected_region
    elif region_preference.lower() == 'keep':
        final_region = excluded_region
    else:
        # Default or 'unknown' case
        final_region = "Outer Space"

    return final_region, selected_country

def get_default_location(
    strategy='global_hq',
    mineral_country_mapping=None, 
    mineral=None
    ):
    """
    Returns a default location based on the specified strategy when there's no
    intersection between a mineral's producing countries and a selected region.

    Parameters
    ----------
    strategy : str, optional
        The strategy to use for determining the default location. Options include:
        'global_hq' for a global default location,
        'unknown' for a generic unknown placeholder,
        'largest_producer' for selecting the country that is the largest producer
        of the mineral,
        'random_from_all' for a random selection from all producing countries.
        Default is 'global_hq'.
    mineral_country_mapping : dict, optional
        A dictionary mapping minerals to lists of countries that produce them. 
        Required for 'largest_producer' and 'random_from_all' strategies.
    mineral : str, optional
        The mineral in question. Required for the 'largest_producer' strategy.

    Returns
    -------
    str
        The default location based on the chosen strategy.
    """
    # XXX TODO 
    # largerst producer stragey not implement yet 
    if strategy == 'largest_producer' : 
        strategy = 'unknown'
    if strategy == 'global_hq':
        return "Global HQ"
    elif strategy == 'unknown':
        return "Unknown"
    elif ( 
            strategy == 'largest_producer' 
            and mineral_country_mapping is not None 
            and mineral is not None
            ):
        # Assuming additional data exists mapping minerals to their largest producer
        largest_producer_mapping = {m: max(countries, key=lambda k: k.production)
                                    for m, countries in mineral_country_mapping.items()}
        return largest_producer_mapping.get(mineral.lower(), "Unknown")
    elif strategy == 'random_from_all' and mineral_country_mapping is not None:
        all_countries = sum(mineral_country_mapping.values(), [])
        return np.random.choice(all_countries) if all_countries else "Unknown"
    else:
        return "Invalid Strategy or Missing Data"

def find_countries_by_distributions(distributions, return_countries_only=False):
    """
    Aggregates countries based on their mineral distributions across different
    regions. The function can return either a list of countries per region
    or a detailed mapping of minerals to countries within each region,
    depending on the `return_countries_only` flag.

    Parameters
    ----------
    distributions : dict
        A dictionary where keys are region names and values are lists of 
        minerals significant to those regions. The structure should strictly 
        follow {'RegionName': ['mineral1', 'mineral2', ...]} format.
    return_countries_only : bool, optional
        Determines the format of the returned data. If True, the function 
        returns a list of countries for each region that produce any of the 
        specified minerals. If False, the function provides a detailed 
        mapping of each mineral to its producing countries within the region. 
        The default is False.

    Returns
    -------
    dict
        Depending on `return_countries_only`, returns:
        - If False (default): A dictionary with regions as keys and dictionaries
          as values, where each key-value pair within the inner dictionary 
          maps a mineral to a list of countries producing it in that region.
        - If True: A dictionary with regions as keys and lists of countries 
          as values, each list representing countries in that region producing
          any of the specified minerals.

    Raises
    ------
    ValueError
        If `distributions` is not a dictionary or if any of the values in the
        `distributions` dictionary are not lists, indicating an incorrect
        format of the input data.

    Examples
    --------
    >>> from gofast.datasets.util import find_countries_by_distributions
    >>> distributions = {
    ...     "Oceania": ["iron ore", "gold"],
    ...     "Africa": ["diamonds", "gold"]
    ... }
    >>> find_countries_by_distributions(distributions, return_countries_only=True)
    {'Oceania': ['Australia', 'Papua New Guinea', 'New Zealand'],
     'Africa': ['South Africa', 'Botswana', 'Ghana', 'Zimbabwe']}

    >>> find_countries_by_distributions(distributions, return_countries_only=False)
    {'Oceania': {'iron ore': ['Australia'], 
                 'gold': ['Australia', 'Papua New Guinea', 'New Zealand']},
     'Africa': {'diamonds': ['Botswana', 'South Africa'], 
                'gold': ['South Africa', 'Ghana', 'Zimbabwe']}}

    Notes
    -----
    The function is designed to process structured data where regions are
    mapped to their significant minerals. This information is then used to
    aggregate country-level data based on regional mineral distributions.
    The actual country and mineral data processing is handled by the
    `find_countries_and_minerals_by_region` function, which is not defined
    in this snippet.
    """
    if not isinstance(distributions, dict):
        raise ValueError("Distributions must be a dictionary with regions as keys"
                         " and lists of minerals as values.")

    region_countries = defaultdict(list)  # More Pythonic and clear initialization

    for region, minerals_list in distributions.items():
        # Ensure minerals_list is actually a list for consistency
        if not isinstance(minerals_list, list):
            raise ValueError(f"Minerals for region '{region}' must be provided as a list.")

        # Simplified logic by eliminating redundancy
        countries_or_mineral_countries = find_countries_and_minerals_by_region(
            region, minerals=minerals_list,
            return_countries_only=return_countries_only,
        )
        # Directly use the result without unnecessary list wrapping
        region_countries[region] = countries_or_mineral_countries
    # Convert defaultdict back to dict for the return value, if necessary
    return dict(region_countries)  
            
def find_countries_by_minerals(
    minerals, 
    mineral_prod_by_country=None, 
    return_countries_only=False, 
    return_minerals_and_countries=False, 
    ):
    """
    Finds countries that produce specified minerals, either returning a list
    of countries or a mapping of minerals to countries based on the input
    parameters. It can accept a single mineral or a list of minerals.

    Parameters
    ----------
    minerals : str or list of str
        The name(s) of the mineral(s) to find the producing countries for.
        Can be a single mineral name as a string or a list of mineral names.
    mineral_prod_by_country : dict, optional
        A dictionary mapping country names to a string of minerals they produce,
        separated by commas. If not provided, the function attempts to use
        a default global dictionary. Default is None.
    return_countries_only : bool, optional
        If True, the function returns a list of unique countries that produce
        any of the specified minerals. If False, returns a dictionary mapping
        each specified mineral to a list of countries that produce it.
        Default is False.

    Returns
    -------
    list or dict
        If return_countries_only is False, returns a dictionary where each key
        is a mineral, and each value is a list of countries that produce that
        mineral. If return_countries_only is True, returns a list of unique
        countries that produce any of the specified minerals.

    Raises
    ------
    ImportError
        If `mineral_prod_by_country` is not provided and the global dictionary
        `MINERAL_PROD_BY_COUNTRY` cannot be imported.

    Examples
    --------
    >>> from gofast.datasets.util import find_countries_by_minerals
    >>> find_countries_by_minerals('Gold')
    {'gold': ['Australia', 'South Africa', 'Canada']}

    >>> find_countries_by_minerals(['Gold', 'Silver'], return_countries_only=True)
    ['Australia', 'South Africa', 'Canada', 'Mexico']

    Providing a dictionary directly and looking for multiple minerals:

    >>> minerals = ['coal', 'rare earth elements']
    >>> mineral_data = {'Australia': 'iron ore, lithium', 
    ...                 'China': 'coal, rare earth elements'}
    >>> find_countries_by_minerals(minerals, mineral_prod_by_country=mineral_data)
    {'coal': ['China'], 'rare earth elements': ['China']}
    """
    # Normalize minerals input to a set of lowercase mineral names
    normalized_minerals = {min.lower() for min in minerals} if isinstance(
        minerals, list) else {minerals.lower()}

    # Extract mineral production information from countries
    countries_minerals = extract_minerals_from_countries(
        mineral_prod_by_country, split_minerals=True)

    # Initialize containers for results
    countries = set()
    mineral_by_country = defaultdict(list)
    
    # Loop through the country-mineral data to populate results
    for country, country_minerals in countries_minerals.items():
        country_mineral_set = set(m.lower() for m in country_minerals)
        intersecting_minerals = normalized_minerals & country_mineral_set
        if intersecting_minerals:
            countries.add(country)
            for mineral in intersecting_minerals:
                mineral_by_country[mineral].append(country)
    
    # Error handling for no matches found
    if not countries:
        message = ( 
            "No countries found producing the specified minerals:"
            f" {', '.join(normalized_minerals)}. Consider providing your"
            " own mineral production dictionary."
         )
        raise ValueError(message)
    
    # Decide return value based on parameters
    if return_countries_only:
        return list(countries)
    elif return_minerals_and_countries:
        return list(countries), dict(mineral_by_country)
    else:
        return dict(mineral_by_country)
       
def find_mineral_by_country(country, mineral_prod_by_country=None):
    """
    Finds and returns the mineral production data for a given country, 
    allowing for case-insensitive matching of country names. If the country 
    is not found within the provided or default mineral production data, 
    an informative error is raised.

    Parameters
    ----------
    country : str
        The name of the country for which to find mineral production data.
    mineral_prod_by_country : dict, optional
        A dictionary mapping country names to their mineral production data. 
        If not provided, the function attempts to use a default global 
        dictionary. If the global dictionary is also unavailable, 
        an ImportError is raised.

    Returns
    -------
    dict
        The mineral production data for the specified country.

    Raises
    ------
    ImportError
        If `mineral_prod_by_country` is not provided and the global dictionary
        `MINERAL_PROD_BY_COUNTRY` cannot be imported.
    ValueError
        If the specified country is not found in the mineral production data.

    Examples
    --------
    Assuming a global dictionary `MINERAL_PROD_BY_COUNTRY` is available 
    and contains relevant data:

    >>> from gofast.datasets.util import find_mineral_by_country
    >>> find_mineral_by_country('Canada')
    ['potash', 'uranium']

    Providing a dictionary directly:

    >>> mineral_data = {'USA': {'Coal': 500}, 'Canada': {'Gold': 100, 'Silver': 200}}
    >>> find_mineral_by_country('canada', mineral_prod_by_country=mineral_data)
    {'Gold': 100, 'Silver': 200}

    Case where the country is not found, illustrating the error message:

    >>> find_mineral_by_country('Atlantis')
    ValueError: Country 'Atlantis' not found in the list of minerals. 
    Sample countries include: Canada, USA, France, Germany, Japan.

    Note: The function is case-insensitive and will normalize the country name.
    """
    use_default=False 
    if mineral_prod_by_country is None:
        mineral_prod_by_country = get_mineral_production_by_country()
        use_default=True 
    # Transform the country names in the dictionary to lowercase once
    country_minerals_dict = {k.lower(): v for k, v in extract_minerals_from_countries(
        mineral_prod_by_country, use_default=use_default, split_minerals=True).items()}
    # Perform a case-insensitive lookup by converting the input country to lowercase
    country_lower = country.lower()
    if country_lower not in country_minerals_dict:
        # Generate an informative error message
        # Just as an example, take the first 5 countries
        sample_countries = list(country_minerals_dict.keys())[:5]  
        sample_countries_str = ', '.join([ c.title () for c in sample_countries])
        raise ValueError(f"Country '{country}' not found in the list of minerals. "
                         f"Sample countries include: {sample_countries_str}.")
    
    return country_minerals_dict.get(country_lower)

def find_mineral_distributions(
    region,
    mineral_prod_by_country=None, 
    detailed_by_country=False, 
    include_region_summary=False
    ):
    """
    Aggregates and presents mineral distribution data for specified geographical
    regions, with options for detailed data by country and a summary of
    minerals found in the region.

    This function first retrieves countries within the specified region, then 
    extracts and aggregates mineral reserves information from these countries. 
    It offers flexibility  in the output format, allowing for detailed 
    information by country, a summary of unique minerals in the region, or both.

    Parameters:
    ----------
    region : str
        The geographical region of interest. The function is case-insensitive and
        supports partial matches to accommodate broad searches.
    mineral_prod_by_country : dict, optional
        A dictionary mapping countries to their mineral production and reserves.
        If None, the function attempts to use a default dictionary defined 
        in the scope.
    detailed_by_country : bool, optional
        If set to True, the function returns a dictionary with countries in 
        the specified region and their corresponding minerals. Default is False.
    include_region_summary : bool, optional
        If True, the function appends a summary list of unique minerals found across
        the entire region to the output. This parameter requires `detailed_by_country`
        to be True to take effect. Default is False.

    Returns:
    -------
    dict
        Depending on the parameters, the function returns:
        - A dictionary with a single key-value pair, where the key is the region
          and the value
          is a list of unique minerals, if `detailed_by_country` is False.
        - A dictionary with a single key-value pair, where the key is the region
          and the value is a detailed mapping of each country to its minerals,
          if `detailed_by_country` is True.
        - If `include_region_summary` is also True, the value includes both 
          the detailed mapping and a list of unique minerals in the region.

    Examples:
    --------
    >>> from gofast.datasets.util import find_mineral_distributions
    >>> find_mineral_distributions('Europe')
    {'Europe': ['palladium', 'nickel', 'potash', 'uranium', ...]}
    
    >>> find_mineral_distributions('Europe', detailed_by_country=True)
    {'Europe': {'Russia': 'palladium, nickel', 'Sweden': 'iron ore, copper', ...}}
    
    >>> find_mineral_distributions('Europe', detailed_by_country=True,
    ...                             include_region_summary=True)
    {'Europe': [{'Russia': 'palladium, nickel', 'Sweden': 'iron ore, copper', ...},
                ['palladium', 'nickel', 'iron ore', 'copper', ...]]}

    Note:
    -----
    The function is designed to provide flexibility in accessing mineral 
    distribution data across different geographical regions. It can be used 
    for data analysis, educational purposes, or informing policy and investment
    decisions related to mineral resources.
    """
    countries = find_countries_by_region(region)

    # If no custom dictionary is provided, use a default
    mineral_info = extract_minerals_from_countries(
        mineral_prod_by_country, split_minerals=True )
    #
    # If detailed_by_country is True or include_region_summary is True,
    # prepare country-specific data
    if detailed_by_country or include_region_summary:
        country_minerals = {
            country: mineral_info.get(country, None) for country in countries
        }
    
    # Aggregate unique minerals found across the region
    region_minerals = {
        mineral for country in countries 
        for mineral in (
            [mineral_info.get(country, [])] if isinstance(
                mineral_info.get(country, []), str) 
            else mineral_info.get(country, [])
        )
    }
    # Assemble the return data based on the function parameters
    if include_region_summary:
        # Returns both country-specific data and a summary of minerals in the region
        return {region: [country_minerals, list(region_minerals)]}
    
    if detailed_by_country:
        # Returns a dictionary mapping each country in the region to its minerals
        return {region: country_minerals}
    
    # Default return: a list of unique minerals found in the region
    return {region: list(region_minerals)}

def find_countries_and_minerals_by_region (
    region, 
    minerals=None, 
    return_countries_only=False
    ):
    """
    Finds countries within a specified region that are associated with
    given minerals.

    Parameters:
    ----------
    region : str
        The geographical region to search within.
    minerals : str or list of str, optional
        A single mineral or a list of minerals to find countries for. 
        If None, the function returns all countries in the region without 
        filtering by mineral.
    return_countries_only : bool, optional
        If True, returns a list of unique countries associated with the minerals.
        If False, returns a dictionary mapping each mineral to a list of countries.

    Returns:
    -------
    dict or list
        Depending on `return_countries_only`, returns either:
        - A dictionary mapping each specified mineral to a list of countries in the region
          that have reserves of that mineral.
        - A list of unique countries in the region associated with the specified minerals.

    Example:
    --------
    >>> from gofast.datasets.util import find_countries_and_minerals_by_region
    >>> find_countries_and_minerals_by_region('Europe', minerals=['gold', 'nickel'], 
    ...                             return_countries_only=True)
    ['Russia', 'Finland']

    >>> find_countries_and_minerals_by_region('Europe', minerals='gold')
    {}
    """
    extracted_minerals = find_mineral_distributions(region, detailed_by_country=True)
    
    if minerals is None: 
        return list( extracted_minerals.get(region).keys() ) 
    
    # Ensure minerals is a list for iteration
    if isinstance(minerals, str):
        minerals = [minerals]
    
    # Using defaultdict to automatically initialize lists for new keys
    mineral_by_country = defaultdict(list)
    
    # Loop through each mineral and aggregate countries
    for reg, countries_minerals in extracted_minerals.items():
        for country, country_minerals in countries_minerals.items():
            country_minerals= minerals_normalizer(country_minerals)
            for mineral in minerals:
                # Normalize case for comparison
                if mineral.lower() in ''.join(country_minerals):
                    mineral_by_country[mineral].append(country)
    
    if return_countries_only:
        # Combine all lists and remove duplicates
        all_countries = set(itertools.chain(*mineral_by_country.values()))
        return list(all_countries)
    
    return dict(mineral_by_country)

def generate_ore_infos(countries=None, error='warn'):
    """
    Generates informational summaries about ore reserves and production
    capabilities from a predefined or provided dictionary of mineral
    production by country. This function allows for case-insensitive
    country name comparison and offers flexible error handling.

    Parameters
    ----------
    countries : str or list of str, optional
        A country name or a list of country names for which to generate ore
        information. If None, information for all countries in the
        mineral production dictionary will be generated. Names are
        case-insensitive.
    error : {'warn', 'raise', 'ignore'}, optional
        Specifies how to handle countries not found in the mineral production
        dictionary. 'warn' issues a warning, 'raise' raises a ValueError, and
        'ignore' does nothing. Default is 'warn'.

    Returns
    -------
    dict
        A dictionary where each key is a country name and each value is a
        string summarizing the last two provided information strings about
        mineral production and reserves for that country.

    Raises
    ------
    ImportError
        If the global mineral production dictionary cannot be imported and no
        alternative dictionary is provided.
    ValueError
        If `error` is set to 'raise' and one or more specified countries
        are not found in the mineral production dictionary.

    Examples
    --------
    Assuming `MINERAL_PROD_BY_COUNTRY` is available and formatted correctly:

    >>> from gofast.datasets.util import generate_ore_infos
    >>> generate_ore_infos(['Australia', 'China'])
    {'Australia': 'High production capacity for iron ore, lithium and major\\
     exporter of lithium, iron ore.',
     'China': 'World's top producer of several minerals including rare earths,\\
         coal and significant exporter of rare earth elements, coal.'}

    Specifying a single country and using 'warn' for not found countries:

    >>> generate_ore_infos('Atlantis', error='warn')
    Warning: Countries not found: Atlantis.
    {}

    Providing a list of countries with mixed case and requesting 'raise' for errors:

    >>> generate_ore_infos(['australia', 'Atlantis'], error='raise')
    ValueError: Countries not found: Atlantis.
    """
    mineral_prod_by_country = get_mineral_production_by_country()
    # Normalize countries input to a list of lowercase names
    # for case-insensitive comparison
    if countries is not None:
        countries = [countries.lower()] if isinstance(countries, str) else [
            country.lower() for country in countries]
    else:
        countries = list(mineral_prod_by_country.keys())
        
    country_infos = defaultdict(str)
    for country, info_list in mineral_prod_by_country.items():
        # Skip countries not in the provided list, if applicable
        if countries and country.lower() not in countries:
            continue
        
        # Process the last two items from the info list
        if len(info_list) >= 2:
            infos = " and ".join(info_list[-2:])  # Join the last two descriptions
            infos = infos.replace("[", "").replace("]", "")  # Remove square brackets
            country_infos[country] = infos.capitalize()  # Capitalize the first letter
        else:
            # Handle countries with less than 2 info strings differently
            # This is just an example, adjust based on actual requirements
            country_infos[country] =( 
                info_list[0] if info_list else "No information available."
                )

    # Check for any countries provided but not found in the dictionary
    not_found_countries = [country for country in countries if country.lower() 
                           not in [c.lower() for c in country_infos]]
    if not_found_countries and error != 'ignore':
        message = "Countries not found: {}".format(
                ', '.join([c.capitalize() for c in not_found_countries])) 
        if error == 'warn':
            print(f"Warning: {message}")
        elif error == 'raise':
            raise ValueError(message)

    return dict(country_infos)

def manage_data(
    data, 
    as_frame= False, 
    return_X_y= False, 
    split_X_y= False, 
    target_names= None, 
    test_size= 0.3, 
    noise= None, 
    seed= None, 
    **kwargs
):

    """ Manage the data and setup into an Object 
    
    Parameters
    -----------
    data: Pd.DataFrame 
        The dataset to manage 

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bowlspace object. See
        below for more information about the `data` and `target` object.
        
    split_X_y: bool, default=False,
        If True, the data is splitted to hold the training set (X, y)  and the 
        testing set (Xt, yt) with the according to the test size ratio. 
        
    target_names: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively
        . 
    noise : float, Optional
        The percentage of values to be replaced with NaN in each column. 
        This must be a number between 0 and 1. Default is None.
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns 
    -------
    data : :class:`~gofast.tools.box.Boxspace` object
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} 
            The data matrix. If ``as_frame=True``, `data` will be a pandas DataFrame.
        target: {ndarray, Series} 
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame 
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

    data, target: tuple if `return_X_y` is ``True``
        A tuple of two ndarray. The first containing a 2D array of shape
        (n_samples, n_features) with each row representing one sample and
        each column representing the features. The second ndarray of shape
        (n_samples,) containing the target samples.

    X, Xt, y, yt: Tuple if `split_X_y` is ``True`` 
        A tuple of two ndarray (X, Xt). The first containing a 2D array of:
            
        .. math:: 
            
            \\text{shape}(X, y) =  1-  \\text{test_ratio} *\
                (n_{samples}, n_{features}) *100
            
            \\text{shape}(Xt, yt)= \\text{test_ratio} * \
                (n_{samples}, n_{features}) *100
        
        where each row representing one sample and each column representing the 
        features. The second ndarray of shape(n_samples,) containing the target 
        samples.
    
    """
    # Ensure the correct data types for the parameters
    as_frame, return_X_y, split_X_y = map(
        lambda x: bool(x), [as_frame, return_X_y, split_X_y]
    )
    test_size = float(test_size)
   
    if seed is not None:
        seed = int(seed)
    
    if target_names: 
        target_names = is_iterable (
            target_names, exclude_string=True,transform=True )
    frame = data.copy()

    feature_names = (
        is_in_if(list( frame.columns), target_names, return_diff =True )
        if target_names else list(frame.columns )
    )
    y = None
    if return_X_y:
        y = data [target_names].squeeze ()  
        data.drop( columns = target_names, inplace =True )
        
    # # Apply noises: Noises only in the data not in target
    # add_gaussian_noise=False 
    # noise = validate_noise(noise )
    
    # if noise=='gaussian': 
    #     add_gaussian_noise=True 
    #     #Small value of noise. Do nothing when gaussian noises 
    #     # is applied, just to skip value error. 
    #     noise =.1 
        
    data = add_noises_to(
        data, noise=noise, seed=seed)

    if not as_frame:
        data = np.asarray(data)
        y = np.squeeze(np.asarray(y))
    
    if split_X_y:
        return train_test_split(data, y, test_size=test_size, random_state=seed)
    
    if return_X_y:
        return data, y

    frame[feature_names] = add_noises_to(
        frame[feature_names], 
        noise=noise,
        seed=seed, 
        )

    if as_frame:
        return frame
    
    return Boxspace(
        data=data,
        target=frame[target_names].values if target_names else None,
        frame=frame,
        target_names=[target_names] if target_names else [],
        feature_names=feature_names,
        **kwargs
    )


def get_item_from ( spec , /,  default_items, default_number = 7 ): 
    """ Accept either interger or a list. 
    
    If integer is passed, number of items is randomly chosen. if 
    `spec` is given as a list of objects, then do anything.
    
    Parameters 
    -----------
    spec: int, list 
        number of specimens or speciments to fecth 
    default_items: list 
       Global list that contains the items. 
       
    default_number: int, 
        Randomly select the number of specimens if `spec` is ``None``. 
        If ``None`` then take the length of all default items in 
        `default_items`. 
    Return
    -------
    spec: list 
       List of items retrieved according to the `spec`. 
       
    """
    if default_number is None: 
        default_number= len(default_items )
        
    if spec is None: 
        spec =default_number 
        
    if isinstance ( spec, ( int, float)): 
        spec = np.random.choice (
            default_items, default_number if int(spec)==0 else int (spec) )
    
    spec = is_iterable ( spec, exclude_string= True, transform =True )
    
    return spec 


def generate_synthetic_values(
        samples, range_min, range_max, noise=None, seed=None):
    """
    Generate synthetic data within a given range, optionally adding noise.
    """
    np.random.seed(seed)
    values = np.random.uniform(range_min, range_max, samples)
    if noise:
        values += np.random.normal(0, noise, samples)
    return values
 
def generate_categorical_values(samples, categories, seed=None):
    """
    Generate synthetic categorical data based on specified categories.
    """
    np.random.seed(seed)
    values = np.random.choice(categories, size=samples)
    return values

def generate_regression_output(X, coef, bias, noise, regression_type):
    """
    Generates the regression output based on the specified regression type.
    """
    from ..tools.mathex import linear_regression, quadratic_regression
    from ..tools.mathex import exponential_regression,logarithmic_regression
    from ..tools.mathex import sinusoidal_regression, cubic_regression
    from ..tools.mathex import step_regression
    
    available_reg_types = [ 'linear', 'quadratic', 'cubic','exponential', 
                           'logarithmic', 'sinusoidal', 'step' ]
    regression_dict =  dict ( zip ( available_reg_types, [ 
        linear_regression, quadratic_regression, cubic_regression, 
        exponential_regression,logarithmic_regression, sinusoidal_regression,
        step_regression] ))
                          
    if regression_type not in available_reg_types: 
        raise ValueError(f"Invalid regression_type '{regression_type}'. Expected"
                         f" {smart_format(available_reg_types, 'or')}.")

    return regression_dict[regression_type](X, coef=coef , bias=bias, noise=noise )
        
def apply_scaling(X, y, method):
    """
    Applies the specified scaling method to the data.
    """
    from ..tools.mathex import standard_scaler, minmax_scaler, normalize
    
    scale_dict = {'standard':standard_scaler ,
    'minmax':minmax_scaler , 'normalize':normalize }
    if method not in  (scale_dict.keys()): 
        raise ValueError (f"Invalid scale method '{method}'. Expected"
                          f"{smart_format(scale_dict.keys(),'or')}")
    return scale_dict[method] ( X, y=y )

def rename_data_columns(data, new_columns=None):
    """
    Renames the columns of a pandas DataFrame or the name of a pandas Series.

    This function adjusts the column names of a DataFrame or the name of a Series
    to match the provided list of new column names. If the new column names list is
    shorter than the number of existing columns, the remaining columns are 
    left unchanged.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The DataFrame or Series whose columns or name are to be renamed.
    new_columns : list or iterable, optional
        The new column names or Series name. If None, no changes are made.

    Returns
    -------
    pd.DataFrame or pd.Series
        The DataFrame or Series with updated column names or name.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    >>> rename_data_columns(df, ['X', 'Y'])
       X  Y
    0  1  2
    1  3  4

    >>> series = pd.Series([1, 2, 3], name='A')
    >>> rename_data_columns(series, ['X'])
    Name: X, dtype: int64
    """
    if new_columns is not None and hasattr(data, 'columns'):
        # Ensure new_columns is a list and has the right length
        new_columns = list(new_columns)
        extra_columns = len(data.columns) - len(new_columns)
        if extra_columns > 0:
            # Extend new_columns with the remaining original columns if not 
            # enough new names are provided
            new_columns.extend(data.columns[-extra_columns:])
        data.columns = new_columns[:len(data.columns)]
    elif new_columns is not None and isinstance(data, pd.Series):
        data.name = new_columns[0]
    return data

def get_mineral_production_by_country():
    """
    Attempts to import and return a copy of the MINERAL_PROD_BY_COUNTRY 
    dictionary from the ._globals module. If the import fails, it raises 
    an ImportError with a message indicating that the dictionary is not found 
    and suggesting to provide a mineral production dictionary.

    Returns
    -------
    dict
        A copy of the MINERAL_PROD_BY_COUNTRY dictionary.

    Raises
    ------
    ImportError
        If MINERAL_PROD_BY_COUNTRY cannot be found in the ._globals module.
    """
    try:
        from ._globals import MINERAL_PROD_BY_COUNTRY
        return MINERAL_PROD_BY_COUNTRY.copy()
    except ImportError:
        raise ImportError("MINERAL_PROD_BY_COUNTRY not found. Please provide"
                          " a mineral production dictionary.")

def build_continent_country_dict():
    """
    Constructs a dictionary mapping each of the five continents to a list of
    countries belonging to them based on a global COUNTRY_REGION mapping.

    Returns
    -------
    dict
        A dictionary where keys are continent names (capitalized) and values
        are lists of countries belonging to those continents.
    """
    from ._globals import COUNTRY_REGION 
    region_country_dict = defaultdict(list)
    
    # Define the continents
    continents = ['Africa', 'America', 'Asia', 'Oceania', 'Europe']
    
    # Populate the dictionary
    for country, region in COUNTRY_REGION.items():
        # Split the region by '/' in case a country belongs to multiple regions
        split_regions = region.split('/')
        for continent in continents:
            if any(continent.lower() in region_part.lower() 
                   for region_part in split_regions):
                region_country_dict[continent].append(country)
    
    return region_country_dict


def _parse_numeric_value_with_unit(text):
    # Define conversion for billion and million
    conversions = {
        'billion': 1e9,
        'million': 1e6
    }
    
    # Remove commas for easier processing
    text = text.replace(",", "")
    
    # Search for numeric values and units in the text
    match = re.search(r'(\d+(\.\d+)?)\s*(billion|million)?', text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        if unit:
            value *= conversions[unit.lower()]
        return value
    return np.nan

def _parse_production(text):
    if 'annually' in text or 'annual' in text:
        multiplier = 15  # Estimate over 15 years
    else:
        multiplier = 1
    
    value = _parse_numeric_value_with_unit(text)
    if 'emerging' in text or 'emerge' in text:
        # Randomly generate a positive value
        return  random.choice([1e5, 1e6, 1e7]) * multiplier
    elif 'world-leading' in text:
        # Estimate in billions of tonnes
        return  random.choice([1e9, 2e9, 3e9]) * multiplier
    return value * multiplier

def _parse_feature_value(text, is_numeric=False):
    # Extract value inside brackets
    match = re.search(r'\[(.*?)\]', text)
    if match:
        value = match.group(1).strip()
        if is_numeric:
            # Remove non-numeric characters for numeric processing
            numeric_value = re.sub("[^0-9]", "", value.split()[0])
            if '+' in value:
                # Range from value to 120 years
                start_year = int(numeric_value)
                return random.randint(start_year, 120)
            elif 'Over' in value:
                # Exact or more than the specified number
                value = int(re.search(r'\d+', value).group())
                return random.randint(value, value + 50)
            else:
                return _parse_numeric_value_with_unit(value)
        else:
            if value.lower() == 'none':
                return 'Not Applicable'
            return value.capitalize()
    return np.nan if is_numeric else 'Not Applicable' 

def parse_mineral_reserve_details(details):
    """
    Parses mineral production details from a dictionary, extracting and
    transforming information based on specified rules for each feature.

    This function handles various features differently, parsing numeric values,
    extracting values from brackets, and handling specific textual patterns.
    It specifically treats 'estimated_production' by converting textual
    representations of numbers into numeric values, adjusts 'reserve_life'
    based on the presence of 'years' and certain keywords, and formats
    'associated_minerals' to ensure a consistent list format.

    Parameters
    ----------
    details : dict
        A dictionary where keys represent features of mineral production details
        (e.g., 'estimated_production', 'extraction_cost') and values are strings
        containing the information to be parsed.

    Returns
    -------
    dict
        A new dictionary with the same keys as `details` but with parsed and
        potentially transformed values based on the feature-specific rules.

    Examples
    --------
    >>> from gofast.datasets.util import parse_mineral_reserve_details
    >>> details = {
    ...     "estimated_reserves": "[Approx. 90 tonnes] of gold annually, ...",
    ...     "extraction_cost": "[Varies], with some remote and ...",
    ...     "grade": "[High]-grade gold, with uranium and copper varying ...",
    ...     "accessibility": "[Moderate], with some mines in remote ...",
    ...     "reserve_life": "[20+ years], with ongoing exploration ...",
    ...     "ownership": "[State-controlled] and [foreign joint ventures], ...",
    ...     "regulatory_status": "[Reforming], with recent efforts to ...",
    ...     "market_demand": "[High for all minerals], given their use ...",
    ...     "historical_data": "[Rich in resources], with a long history ...",
    ...     "associated_minerals": "[Silver, lead, zinc], often found ...",
    ...     "technology_used": "[Varied], from traditional to high-tech,..."
    ... }
    >>> parsed_details = parse_mineral_reserve_details(details)
    >>> for key, value in parsed_details.items():
    ...     print(f"{key}: {value}")
    {'estimated_production': 1350.0,
     'extraction_cost': 'Varies',
     'grade': 'High',
     'accessibility': 'Moderate',
     'reserve_life': 20,
     'ownership': 'State-controlled',
     'regulatory_status': 'Reforming',
     'market_demand': 'High for all minerals',
     'historical_data': 'Rich in resources',
     'associated_minerals': 'Silver, lead, zinc',
     'technology_used': 'Varied'}
    
    The output will display parsed and formatted values for each key in the 
    details dictionary, according to the specified parsing rules.
    """
    new_details = {}
    for feature, text in details.items():
        if feature == "estimated_reserves":
            new_details[feature] = _parse_production(text)
        elif feature in ["extraction_cost", "grade", "accessibility", 
                         "ownership", "regulatory_status", "market_demand",
                         "historical_data", "technology_used"]:
            new_details[feature] = _parse_feature_value(text)
        elif feature == "reserve_life":
            new_details[feature] = _parse_feature_value(text, is_numeric=True)
        elif feature == "associated_minerals":
             # if ',,' exist remove one and lower minerals
            new_details[feature] = _parse_feature_value(text).replace(
                ' and ', ', ').replace (',,', ',').lower()
    return new_details


def build_reserve_details_by_country(country, reserves_details=None):
    """
    Retrieves detailed information about the mineral reserves for a specified 
    country.
    
    Function applies default values for missing details and supports both numeric
    and textual information types, ensuring a comprehensive output regardless of
    data availability.

    Parameters
    ----------
    country : str
        The name of the country for which to retrieve mineral reserves details.
    reserves_details : dict, optional
        A custom dictionary mapping countries to their mineral reserves details.
        If not provided, the function uses a predefined global dictionary.

    Returns
    -------
    dict
        A dictionary containing detailed information about the country's mineral reserves.
        Numeric fields are set to NaN and textual fields to 'Not Available' if data is missing.

    Examples
    --------
    >>> from gofast.datasets.util import find_reserves_details_by_country
    >>> build_reserve_details_by_country('Australia')
    {
        'estimated_production': 1e9,
        'extraction_cost': 'Low to medium, depending on the mineral and location',
        'grade': 'High-grade iron ore (>60% Fe content), Lithium concentration varies',
        'accessibility': 'High, well-established mining infrastructure',
        'reserve_life': 50,
        'ownership': 'Various, including BHP, Rio Tinto, and smaller mining companies',
        'regulatory_status': 'Strict environmental and mining regulations,...,
        'market_demand': 'High, especially in Asia for iron ore and globally...',
        'historical_data': 'Decades of extensive mining history, particularly ...,
        'associated_minerals': 'Gold, bauxite, nickel, and many others',
        'technology_used': 'Open-pit mining, underground mining, and various ...'
    }

    >>> build_reserve_details_by_country('Unknown Country')
    {
        'estimated_production': nan,
        'extraction_cost': 'Not Available',
        'grade': 'Not Available',
        'accessibility': 'Not Available',
        'reserve_life': nan,
        'ownership': 'Not Available',
        'regulatory_status': 'Not Available',
        'market_demand': 'Not Available',
        'historical_data': 'Not Available',
        'associated_minerals': 'Not Available',
        'technology_used': 'Not Available'
    }

    Notes
    -----
    Function is designed to provide a robust mechanism for extracting detailed
    mineral reserves information, leveraging both provided and global data sources
    to maximize data completeness and reliability.
    """
    # Define default values for features, both numeric and textual
    default_values = {
        "numeric": np.nan,
        "textual": "Not Available"
    }
    
    features = [
        'estimated_reserves', 'extraction_cost', 'grade', 'accessibility',
        'reserve_life', 'ownership', 'regulatory_status', 'market_demand',
        'historical_data', 'associated_minerals', 'technology_used'
        ]
    default_details =  {
        k: default_values['numeric'] if k in ['estimated_reserves', 'reserve_life']
        else default_values['textual'] for k in features
    }
    # Check if a custom reserves details dictionary is provided,
    # otherwise use the global default
    if reserves_details is None:
        from ._globals import MINERAL_RESERVES_DETAILS
        reserves_details = MINERAL_RESERVES_DETAILS

    # Validate the country name and fetch its reserves details
    validated_country = validate_country(country, error='ignore')
    if not validated_country:
        # Return defaults for an unrecognized country
        return default_details

    country_reserve_details = reserves_details.get(validated_country, {})

    # Parse the details of the validated country, filling in defaults where necessary
    try: 
        parsed_details = parse_mineral_reserve_details ( country_reserve_details)
    except: 
        parsed_details = {}
        for key, default in default_details.items():
            raw_value = country_reserve_details.get(key, None)
            parsed_details[key] = _parse_feature_value(str(raw_value), is_numeric=(
                key in ['estimated_production', 'reserve_life'])) if raw_value else default

    return parsed_details

def get_last_day_of_current_month(date_format='%Y-%m-%d', return_today=False):
    """
    Returns the last day of the current month or today's date, depending on
    the 'return_today' flag, formatted according to the 'date_format' parameter.

    Parameters:
    - date_format (str): Format in which the date should be returned.
      Defaults to '%Y-%m-%d'.
    - return_today (bool): If True, returns today's date instead of the last
      day of the current month. Defaults to False.

    Returns:
    - str: The last day of the current month or today's date, formatted as specified.
    """
    today = datetime.now()
    if return_today:
        # Return today's date in the specified format
        return today.strftime(date_format)

    current_month = today.month
    current_year = today.year

    # Find the first day of the next month
    if current_month == 12:
        next_month_first_day = datetime(year=current_year + 1, month=1, day=1)
    else:
        next_month_first_day = datetime(
            year=current_year, month=current_month + 1, day=1)

    # Subtract one day to get the last day of the current month
    last_day_current_month = next_month_first_day - timedelta(days=1)

    # Return the last day in the specified format
    return last_day_current_month.strftime(date_format)

def adjust_households_and_days(n_samples, initial_guess=5):
    """
    Adjusts the number of households and days to fit a specified number of samples,
    ensuring both values are practical for a dataset.

    Parameters:
    ----------
    n_samples : int
        The desired number of samples in the dataset.
    initial_guess : int, default=5
        An initial guess for the number of households.

    Returns:
    -------
    tuple of (int, int)
        A tuple containing the adjusted number of households and days, ensuring
        the number of samples is distributed in a practical and meaningful way.

    Examples:
    --------
    >>> from gofast.datasets.util import adjust_households_and_days
    >>> adjust_households_and_days(n_samples=3650)
    (10, 365)  # Example output for 10 households over 365 days
    """
    # Determine an appropriate range for the number of days (e.g., between 30 and 365)
    min_days = 30
    max_days = 365
    
    # Starting from the initial guess, incrementally find the best combination
    n_households = initial_guess
    days = max(min_days, min(n_samples // n_households, max_days))
    
    while n_households * days < n_samples:
        n_households += 1
        days = max(min_days, min(n_samples // n_households, max_days))
        
        # Ensure we're not exceeding practical limits for households
        if days == min_days and n_households * days >= n_samples:
            break
        elif n_households * min_days > n_samples:
            n_households -= 1
            break

    # Recalculate days to get as close as possible to n_samples without going under.
    days = n_samples // n_households
    if n_households * days < n_samples:
        days += 1

    return n_households, days

def adjust_parameters_to_fit_samples(n_samples, initial_guesses):
    """
    Adjusts the parameters n_machines, n_sensors, operational_params, and days
    to fit a given number of samples (n_samples).

    Parameters:
    ----------
    n_samples : int
        The target number of samples to fit.
    initial_guesses : dict, optional
        Initial guesses for the parameters. Should contain keys a guess parameters 
        and values .

    Returns:
    -------
    dict
        A dictionary with the adjusted parameters.
    Examples: 
    ---------
    >>> from gofast.datasets.simulate import adjust_parameters_to_fit_samples
    >>> initial_guesses = {
        'n_machines': 50,
        'n_sensors': 20,
        'operational_params': 5,
        'days': 365
    }
    >>> adjust_parameters_to_fit_samples(100, initial_guesses) 
    {'n_machines': 6, 'n_sensors': 2, 'operational_params': 1, 'days': 11}
    """
    from scipy.optimize import minimize

    initial_values = np.array(list(initial_guesses.values()))
    
    # Objective function to minimize: difference between current and desired number of samples
    def objective(x):
        return abs(n_samples - np.prod(x))

    # Bounds to ensure all values are positive and within reasonable ranges
    bounds = [(1, n_samples) for _ in range(len(initial_values))]

    # Optimization
    result = minimize(
        objective, 
        initial_values, 
        method='L-BFGS-B', 
        bounds=bounds
    )

    # Extract the optimized values
    optimized_values = np.round(result.x).astype(int)
    
    # Construct the result dictionary
    adjusted_parameters = dict(zip(initial_guesses.keys(), optimized_values))

    return adjusted_parameters

def validate_noise_level(noise_level, default_value=None):
    """
    Validates the noise level, ensuring it's within the range [0, 1]. Optionally,
    a default value can be used if the noise level is not specified.

    Parameters:
    ----------
    noise_level : float or None
        The noise level to validate. Should be a float between 0 and 1, inclusive.
        If None and a default_value is provided, the default_value is used.
    default_value : float, optional
        A default noise level to use if noise_level is None. Should also be between
        0 and 1, inclusive.

    Returns:
    -------
    float
        The validated noise level, guaranteed to be within the range [0, 1].

    Raises:
    ------
    ValueError
        If the noise level is outside the [0, 1] range or cannot be properly converted.
    """
    # Use the default value if noise_level is None and default_value is provided
    if noise_level is None:
        if default_value is not None:
            noise_level = default_value
        else:
            raise ValueError(
                "Noise level is not specified, and no default value is provided.")
    
    # Validate the noise level
    return validate_ratio(noise_level, bounds=(0, 1), to_percent=False,
                          param_name='noise_level')


def validate_loan_parameters(*params, default_dict=None, error='raise'):
    """
    Validates loan parameters against predefined or custom ranges, adjusting
    parameters according to a specified error handling policy ('raise', 'warn', 
    'ignore'). This function ensures that each parameter falls within an acceptable
    range, providing flexibility in loan simulation tasks.

    Parameters
    ----------
    *params : dict
        A dictionary containing the loan parameters to validate. Each key
        should correspond to a loan parameter name, and each value should be
        a tuple representing the minimum and maximum range.
    default_dict : dict, optional
        A dictionary mapping parameter names to their default (min, max) ranges.
        If not provided, uses a predefined set of common loan parameters.
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Specifies the error handling policy. 'raise' will raise a ValueError,
        'warn' will issue a warning, and 'ignore' will silently ignore issues.

    Returns
    -------
    dict
        A dictionary of validated loan parameters with (min, max) ranges.

    Raises
    ------
    ValueError
        If a parameter is out of the acceptable range and the error policy is 'raise'.

    Examples
    --------
    >>> from gofast.datasets.util import validate_loan_parameters
    >>> validate_loan_parameters(
            {'credit_score_range': (320, 800)},
            error='warn'
        )
    {'credit_score_range': (320, 800), 'age_range': (18, 70), ...}

    >>> validate_loan_parameters(
            {'loan_amount_range': (1000, 70000)},
            default_dict={'loan_amount_range': (5000, 50000)},
            error='raise'
        )
    ValueError: loan_amount_range is out of the acceptable range.

    Notes
    -----
    This function is particularly useful in preparing datasets for simulations
    involving loan defaults, where parameters like credit scores, loan amounts, 
    and annual incomes need to be within realistic bounds.
    """
    from ..tools.validator import validate_length_range

    def handle_error(message, error_policy):
        if error_policy == 'raise':
            raise ValueError(message)
        elif error_policy == 'warn':
            warnings.warn(message)
        elif error_policy != 'ignore':
            raise ValueError(f"Unrecognized error policy: {error_policy}")

    validated_params = {}
    default_dict = default_dict or {
        'credit_score_range': (300, 850),
        'age_range': (18, 70),
        'loan_amount_range': (5000, 50000),
        'interest_rate_range': (5, 20),
        'employment_length_range': (0, 30),
        'annual_income_range': (20000, 150000)
    }

    for param_name, default_range in default_dict.items():
        param_range =  ( 
            params[0].get(param_name, default_range) 
            if  params[0].get(param_name, default_range) 
            else default_range
            ) 
        if not isinstance(param_range, (list, tuple)): 
            # convert to an iterable list bby no consider str.
            param_name= is_iterable(param_range, exclude_string=True, transform=True )
            if  len(param_range) != 2:
                handle_error(f"{param_name} expects a range with two values.", error)
                param_range = [min(param_range), max(param_range)]
        if any(isinstance(s, str) for s in param_range):
            raise TypeError("Expected numeric values in parameter ranges,"
                            " but received a string. Please ensure all values"
                            " within the range are numeric.")

        param_range = sorted(param_range)
        min_val, max_val = validate_length_range( param_range)

        if min_val < default_range[0] or max_val > default_range[1]:
            handle_error(f"{param_name} is out of the acceptable range.", error)

        validated_params[param_name] = (min_val, max_val)

    return validated_params

def select_diagnostic_options(diagnostic_options=None, n_diseases=7):
    """
    Selects diagnostic options for medical diagnosis simulation.

    If no specific diagnostic options are provided, this function randomly selects 
    a specified number of diseases from a predefined list of possible diseases, 
    including 'Healthy' as a non-disease option.

    Parameters:
    ----------
    diagnostic_options : list of str, optional
        A list specifying the diseases to include as diagnostic options. If None, 
        options will be randomly selected from a predefined list.
    n_diseases : int, default=7
        The number of diseases to randomly select if `diagnostic_options` is None. 
        The final list includes 'Healthy', thus n_diseases + 1 options in total.

    Returns:
    -------
    list of str
        The list of diagnostic options including diseases and 'Healthy'.

    Examples:
    --------
    >>> from gofast.dataset.util import select_diagnostic_options
    >>> select_diagnostic_options(None, 3)
    ['Disease A', 'Disease B', 'Disease C', 'Healthy']

    >>> select_diagnostic_options(['Disease X', 'Disease Y'], 5)
    ['Disease X', 'Disease Y', 'Healthy']

    Notes:
    -----
    The predefined list of possible diseases is maintained in a global variable 
    or external configuration for ease of update and management.
    """
    from ._globals import POSSIBLE_DISEASES  

    if diagnostic_options is None:
        # Randomly select n_diseases from the list of possible diseases
        selected_diseases = random.sample(POSSIBLE_DISEASES, n_diseases)
        # Always include 'Healthy' as an option
        diagnostic_options = selected_diseases + ['Healthy']
    else:
        # Ensure input is iterable and exclude 'str' to avoid treating strings as iterables
        if isinstance(diagnostic_options, str):
            diagnostic_options = [diagnostic_options]
        # Ensure all elements are strings for consistency
        diagnostic_options = [str(option) for option in diagnostic_options]
        # Ensure 'Healthy' is included in the options
        if 'Healthy' not in diagnostic_options:
            diagnostic_options.append('Healthy')

    return diagnostic_options
 
def build_dataset_description(
    title: Union[str, Tuple[str, str]] = '',
    dataset_overview: str = '',
    feature_descriptions: dict = None,
    error_handling: str = 'warn',
) -> Tuple[Optional[DescriptionFormatter], Optional[DescriptionFormatter]]:
    """
    Constructs formatted descriptions for datasets and their features.

    Parameters:
    -----------
    title : str or Tuple[str, str], optional
        Title for the dataset or feature description sections, or a tuple 
        containing
        titles for both dataset and feature descriptions, respectively.
        Defaults to an empty string.
    dataset_overview : str, optional
        A general description of the dataset. Defaults to an empty string.
    feature_descriptions : dict, optional
        A dictionary where keys are feature names and values are descriptions.
        Defaults to None.
    error_handling : str, optional
        Error handling strategy ('raise', 'warn', 'ignore') when invalid inputs
        are provided. Defaults to 'warn'.

    Returns:
    --------
    Tuple[Optional[DescriptionFormatter], Optional[DescriptionFormatter]]
        A tuple containing formatted dataset overview and feature descriptions.
        Either part of the tuple can be None if the corresponding content is 
        not provided or invalid.

    Raises:
    -------
    ValueError
        If 'feature_descriptions' is not a dictionary and error_handling 
        is set to 'raise'.

    Example:
    --------
    >>> from gofast.datasets.util import build_dataset_description
    >>> dataset_desc, feature_desc = build_dataset_description(
            title=("My Dataset", "Feature Information"),
            dataset_overview="This dataset includes...",
            feature_descriptions={"Age": "Age of the individual.", 
                                  "Income": "Annual income."},
            error_handling="warn"
        )
    >>> print(dataset_desc)
    >>> print(feature_desc)
    """
    # Support single title or separate titles for dataset and feature descriptions
    dataset_title, feature_title = (title, title) if isinstance(title, str) else title

    # Validate and format feature descriptions
    features_description = _format_feature_descriptions(
        feature_descriptions, feature_title, error_handling
    )
    # Validate and format dataset overview
    dataset_description = _format_dataset_overview(
        dataset_overview, dataset_title
    )

    return dataset_description, features_description

def _format_feature_descriptions(feature_descriptions, title, error_handling):
    """
    Validates and formats feature descriptions.
    """
    if feature_descriptions and not isinstance(feature_descriptions, dict):
        error_msg = "feature_descriptions must be a dictionary."
        if error_handling == 'raise':
            raise ValueError(error_msg)
        elif error_handling == 'warn':
            print(f"Warning: {error_msg}")
            return None
        elif error_handling == 'ignore':
            return None
    return DescriptionFormatter(
        title=title, content=feature_descriptions) if feature_descriptions else None

def _format_dataset_overview(overview, title):
    """
    Formats dataset overview.
    """
    return DescriptionFormatter(title=title, content=overview) if overview else None
    
def fetch_simulation_metadata(simulation_name, titles=None):
    """
    Fetches and formats the dataset and feature descriptions for a given 
    simulation.
    
    Parameters:
    -----------
    simulation_name : str
        The name of the simulation function for which metadata is to be fetched.
    titles : tuple of str, optional
        A tuple containing titles for the dataset and feature descriptions, 
        respectively. Defaults to ("Dataset Overview", "Dataset Features") 
        if not provided.
    
    Returns:
    --------
    tuple
        A tuple containing the formatted dataset and feature descriptions.
    
    Examples:
    ---------
    >>> from gofast.datasets.util import fetch_simulation_metadata
    >>> dataset_description, feature_description = fetch_simulation_metadata(
            simulation_name="simulate_energy_consumption",
            titles=("Energy Consumption Dataset Overview", 
                    "Energy Consumption Features")
        )
    >>> print(dataset_description)
    >>> print(feature_description)
    
    Note:
    -----
    This function assumes that `SimulationMetadata` and `build_dataset_description`
    are implemented in the `.metadata` module and the current module, respectively.
    """
    
    # Provide default titles if not specified
    if titles is None:
        titles = ("Dataset Overview", "Dataset Features")
    
    simulation_metadata = SimulationMetadata()
    
    # Fetch the descriptions from the simulation metadata
    simulation_metadata.add_metadata(simulation_name)
    dataset_description = simulation_metadata.get_dataset_description(simulation_name)
    feature_description = simulation_metadata.get_feature_description(simulation_name)
    
    # Use an existing function to build and format the descriptions
    formatted_dataset_descr, formatted_feature_descr = build_dataset_description(
        title=titles, 
        dataset_overview=dataset_description, 
        feature_descriptions=feature_description, 
        error_handling='ignore'
    )
    
    return formatted_dataset_descr, formatted_feature_descr
    
    
    
    
    