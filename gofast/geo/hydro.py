# -*- coding: utf-8 -*-

"""
The `hydro` module offers specialized tools for hydrogeology analysis.
"""

import math
import pandas as pd
import numpy as np
from collections import Counter
from ..api.types import Tuple, List, Union, Optional, DataFrame

from ..exceptions import StrataError 

__all__=["get_depth_range", "reduce_samples", "calculate_K", "transmissivity", 
         "compress_aquifer_data", "AquiferGroupAnalyzer"]

def compress_aquifer_data(
    *data: DataFrame, 
    base_stratum: str = "Strata",
    depth_column: Optional[str] = None,
    upper_depth: Optional[float] = None,
    lower_depth: Optional[float] = None,  
    objective_column: Optional[str] = 'Thickness',
    base_stratum_value: Optional[str]=None, 
    stack: bool = False
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Compress the missing values in aquifer datasets by averaging the features 
    associated with the most frequent stratum. 
    
    Function filters out samples with missing values in the objective column 
    in the upper section of the first shallowest aquifer into a single vector. 
    It reduces the number of samples by eliminating those with missing objective  
    values in the specified depth range.

    Parameters
    ----------
    *data : Variable number of DataFrame arguments
        Each dataframe represents borehole data with 'Strata' and a depth column, 
        along with other feature columns to be averaged.
    base_stratum : str, optional
        The most frequent stratum to use as a base for compression.
        Defaults to "Strata".
    depth_column : str, optional
        The name of the depth column to look for in the dataframes. 
        If not provided, the function will look for any column that contains 
        'depth' in its name, case-insensitive.
    upper_depth : float, optional
        The depth that marks the upper boundary of the aquifer section to be 
        compressed. If not supplied, it will be calculated as the minimum depth 
        in the dataframes. Default is None.
    lower_depth : float, optional
        The depth that marks the lower boundary of the aquifer section. If not
        supplied, it will be calculated as the maximum depth in the dataframes. 
        Default is None.
    objective_column : str, optional
        The name of the column representing the objective measure (e.g., Thickness) 
        that must not have missing values within the specified depth range. 
        Defaults to 'Thickness'.
    base_stratum_value : Optional[str]
        The specific stratum to use as the base for compression. If None, the
        most frequent
        stratum within the upper section is used. Defaults to None.
    stack : bool, optional
        If True, returns a single DataFrame by stacking all the compressed 
        DataFrames. If False, returns a list of compressed DataFrames. 
        Defaults to False.
        
    Returns
    -------
    Union[pd.DataFrame, List[pd.DataFrame]]
        Depending on the 'stack' parameter, either returns a single DataFrame 
        containing the compressed and filtered data from all input DataFrames, 
        or a list of DataFrames.

    Raises
    ------
    ValueError
        If no depth column is found in any of the provided dataframes.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.geo.hydro import compress_aquifer_data
    >>> df1 = pd.DataFrame({
    ...     'Depth': [10, 20, 30, 40],
    ...     'Strata': ['Sandstone', 'Limestone', 'Sandstone', 'Shale'],
    ...     'Thickness': [10, 10, 5, 15],
    ...     'Porosity': [0.2, 0.15, 0.25, 0.05]
    ... })
    >>> df2 = pd.DataFrame({
    ...     'Depth': [5, 15, 25, 35],
    ...     'Strata': ['Shale', 'Limestone', 'Sandstone', 'Limestone'],
    ...     'Thickness': [5, 10, 10, 10],
    ...     'Porosity': [0.05, 0.15, 0.20, 0.18]
    ... })
    >>> compressed_df = compress_aquifer_data(
    ...     df1, df2, base_stratum='Limestone', depth_column='Depth', upper_depth=25,
    ...     objective_column='Porosity', stack=True)
    >>> print(compressed_df)
    """
    compressed_data = []

    for df in data:
        # Ensure depth column is specified or discoverable
        depth_col = depth_column if depth_column else _validate_depth_column(df)

        # Compute or use provided upper and lower depths
        actual_upper_depth, actual_lower_depth = _compute_upper_lower_depths(
            df, depth_col, upper_depth, lower_depth)

        # Filter the dataframe based on depth range and availability of objective measure
        df_filtered = df[(df[depth_col] >= actual_upper_depth) & (
            df[depth_col] <= actual_lower_depth)]
        df_filtered = df_filtered[pd.notnull(df_filtered[objective_column])]

        # Compress the upper section of the dataset if an upper boundary is defined
        if upper_depth is not None:
            df_compressed_upper = _compress_upper_section(
                df_filtered, depth_col, base_stratum, actual_upper_depth,
                objective_column, base_stratum_value= base_stratum_value)
            # Merge compressed upper section with remaining data
            df_filtered = pd.concat([df_compressed_upper, df_filtered[
                df_filtered[depth_col] > actual_upper_depth]], ignore_index=True)
        
        compressed_data.append(df_filtered)
        
    if stack:
        # If stack is True, concatenate all compressed data into a 
        # single DataFrame
        return pd.concat(compressed_data, ignore_index=True)
    
    return compressed_data

def _validate_depth_column(df: pd.DataFrame) -> str:
    """
    Validate the presence of a depth column in the dataframe, case-insensitive.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check for a depth column.

    Returns
    -------
    str
        The name of the depth column.

    Raises
    ------
    ValueError
        If no depth column is found.
    """
    depth_col = next((col for col in df.columns if 'depth' in col.lower()), None)
    if depth_col is None:
        raise ValueError('No depth column found in dataframe')
    return depth_col

def _compute_upper_lower_depths(
        df: pd.DataFrame, depth_col: str, 
        upper_depth: Optional[float], 
        lower_depth: Optional[float]) -> (float, float):
    """
    Compute the upper and lower depths if not supplied.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which to compute the depths.
    depth_col : str
        The name of the depth column in the dataframe.
    upper_depth : Optional[float]
        The manually supplied upper depth, if available.
    lower_depth : Optional[float]
        The manually supplied lower depth, if available.

    Returns
    -------
    tuple
        The computed or supplied upper and lower depths.
    """
    if upper_depth is None:
        upper_depth = df[depth_col].min()
    if lower_depth is None:
        lower_depth = df[depth_col].max()
    return upper_depth, lower_depth

def _compress_upper_section(
        df: pd.DataFrame, depth_col: str, base_stratum_col: str, 
        upper_depth: float, objective_column: str,
        base_stratum_value: Optional[str] = None
        ) -> pd.DataFrame:
    """
    Compress the upper section of the aquifer into a single vector, 
    optionally using the most frequent stratum as the base for compression 
    if no specific stratum is provided.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe representing borehole data.
    depth_col : str
        The name of the depth column in the dataframe.
    base_stratum_col : str
        The name of the column representing the stratum type.
    upper_depth : float
        The depth that marks the upper boundary of the aquifer section to be compressed.
    objective_column : str
        The name of the column representing the objective measure (e.g., Thickness).
    base_stratum_value : Optional[str]
        The specific stratum to use as the base for compression. If None, the
        most frequent
        stratum within the upper section is used. Defaults to None.

    Returns
    -------
    pd.DataFrame
        The dataframe with the upper section compressed into a single vector.
    """
    # Filter the dataframe for entries with depth less than or equal to the upper_depth
    upper_section = df[df[depth_col] <= upper_depth]
    
    # Compute the base stratum value as the most frequent stratum if not provided
    if base_stratum_value is None:
        # Get the most frequent stratum
        base_stratum_value = upper_section[base_stratum_col].mode()[0]  
    # Filter for rows where the base stratum column matches the computed or 
    # provided base_stratum_value
    base_stratum_data = upper_section[upper_section[base_stratum_col] == base_stratum_value]
    
    # Calculate the average of the objective_column for the specified base stratum
    avg_value = base_stratum_data[objective_column].mean()
    
    # Prepare other averaged data excluding the specified columns
    avg_logging_data = base_stratum_data.drop(
        columns=[base_stratum_col, depth_col, objective_column]).mean()
    
    # Create a new row with the compressed data
    compressed_sample = {base_stratum_col: base_stratum_value, 
                         objective_column: avg_value, **avg_logging_data.to_dict()}
    m_c_columns = [base_stratum_col, objective_column] + list(avg_logging_data.index)
    
    return pd.DataFrame([compressed_sample], columns=m_c_columns)



def get_depth_range(
    data: pd.DataFrame,
    depth_col: Optional[str] = None,
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate the upper and lower depth limits for sample reduction.

    Parameters
    -----------
    dataframe: pd.DataFrame 
        The dataset containing depth information.
    depth_col: str, optional
        The name of the column in the dataset containing depth values.
        If not specified, the function will check for a column named "depth" 
        (case-insensitive).
    top_depth:  float, optional
       The upper depth limit for sample reduction. If not provided, 
       it will be computed.
    bottom_depth: float, optional
       The lower depth limit for sample reduction. If not provided, 
       it will be computed.

    Returns
    -------
    Tuple[float, float]: A tuple containing the calculated upper 
      and lower depth limits.

    Examples
    ---------
    >>> from gofast.geo.hydro import get_depth_range
    >>> import pandas as pd
    >>> data = pd.DataFrame({'Depth': [0, 10, 20, 30, 40]})
    >>> upper, lower = get_depth_range(data)
    >>> print(upper, lower)
    40 0

    >>> upper, lower = get_depth_range(data, top_depth=30)
    >>> print(upper, lower)
    30 0

    >>> upper, lower = get_depth_range(data, depth_col='Depth',
                                       bottom_depth=10, top_depth=30)
    >>> print(upper, lower)
    30 10
    """
    if depth_col is None:
        # Check for a case-insensitive match with "depth"
        depth_columns = [col for col in data.columns if col.lower() == "depth"]
        if len(depth_columns) == 1:
            depth_col = depth_columns[0]
        else:
            raise ValueError("No 'depth_col' specified, and no 'depth' column"
                             " found in the dataframe.")

    if top_depth is None:
        top_depth = data[depth_col].max()

    if bottom_depth is None:
        bottom_depth = data[depth_col].min()

    return top_depth, bottom_depth

def reduce_samples(
    data: pd.DataFrame,
    strata_column: str, 
    k_column: str, 
    depth_column: Optional[str]=None,
    upper_depth: Optional[float]=None, 
    lower_depth: Optional[float]=None, 
    base_stratum: Optional[str] = None
    ) -> pd.DataFrame:
    """
    Reduces a dataset by compressing missing values from the top of the upper 
    section of the first aquifer
    into a single vector, then vertically stacking with the remaining
    valid data.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing borehole logging information.
    strata_column : str
        The name of the column in 'data' that contains the strata values.
    k_column : str
        The name of the column in 'data' that contains the permeability 
        coefficient 'K'.
    depth_column : str, optional 
        The name of the column in 'data' that contains the depth values.
    upper_depth : float, optional
        The depth marking the top of the upper section of the shallowest aquifer.
    lower_depth : float, optional
        The depth marking the bottom of the borehole section to be considered.
    base_stratum : Optional[str], optional
        The name of the base stratum. If None, the most frequent stratum is
        determined from the data.

    Returns
    -------
    pd.DataFrame
        The reduced dataset with missing 'K' values compressed into a single vector.

    Notes
    -----
    This function assumes that the data has been preprocessed to indicate 
    missing 'K' values appropriately.
    The column specified by 'strata_column' should contain categorical strata data.
    The 'upper_depth' and 'lower_depth' parameters define the section of the 
    dataset to be considered for sample reduction.

    Examples
    --------
    >>> dataset = pd.DataFrame({
    ...     'Depth': [0, 50, 100, 150, 200],
    ...     'Strata': ['Topsoil', 'Clay', 'Sand', 'Gravel', 'Rock'],
    ...     'K': [np.nan, 15, 30, 45, np.nan],
    ...     'OtherFeature': [100, 200, 300, 400, 500]
    ... })
    >>> reduced_dataset = reduce_samples(dataset, strata_column='Strata', k_column='K', 
    ...                                  depth_column='Depth', upper_depth=0, lower_depth=200)
    >>> print(reduced_dataset)
    """
    if depth_column is None:
        # Check for a case-insensitive match with "depth"
        depth_columns = [col for col in data.columns if col.lower() == "depth"]
        if len(depth_columns) >= 1:
            depth_column = depth_columns[0]
        else:
            raise ValueError("No 'depth_column' specified, and no 'depth' column"
                             " found in the dataframe.")
    if upper_depth is None or lower_depth is None: 
        top_depth, bottom_depth= get_depth_range(data, depth_column )
        
    # Filter data based on the specified depth range
    data_within_depth = data[(data[depth_column] >= upper_depth) & (
        data[depth_column] <= lower_depth)]

    # Separate data with missing and valid 'K' values
    data_missing_k = data_within_depth[data_within_depth[k_column].isna()]
    data_valid_k = data_within_depth[data_within_depth[k_column].notna()]

    # Determine the base stratum if not provided
    if base_stratum is None:
        base_stratum = data_missing_k[strata_column].mode().iloc[0]  # Most frequent stratum

    # Calculate the average thickness and logging data for the base stratum
    base_stratum_data = data_missing_k[data_missing_k[strata_column] == base_stratum]
    compressed_vector = base_stratum_data.mean(numeric_only=True).to_dict()
    compressed_vector[strata_column] = base_stratum  # Add stratum name to the vector

    # Create a DataFrame from the compressed vector
    compressed_df = pd.DataFrame(compressed_vector, index=[0])

    # Vertically stack the compressed vector with the remaining valid data
    reduced_dataset = pd.concat([compressed_df, data_valid_k], ignore_index=True)

    return reduced_dataset

def calculate_K(Q: float, H: float, h: float, Ry: float, r: float,
    Q_unit: str = 'm3/day', length_unit: str = 'm', 
    output_unit: str = 'm/day', log_base: Optional[float] = math.e
    ) -> float:
    """
    Calculates the hydraulic conductivity (K) for an aquifer based on the
    single-hole pumping test.

    This function employs the formula derived from observations of steady  flow 
    conditions around a single pumping well. It is particularly useful in 
    determining the aquifer's water conductivity under specific test conditions,
    which involves measuring the aquifer thickness before and during the test, 
    as well as the well's influence radius.

    Parameters
    ----------
    Q : float
        The water output in cubic meters per day (m^3/d), representing the 
        volume of water pumped from the well over the duration of the test.
    H : float
        The free aquifer thickness under natural conditions in meters (m), 
        indicating the vertical extent of the aquifer when not being pumped.
    h : float
        The free aquifer thickness during the pumping test in meters (m), 
        showing the reduction in aquifer thickness due to water extraction.
    Ry : float
        The radius of influence in meters (m), which denotes the distance from
        the well at which the pumping effects are observed in the aquifer.
    r : float
        The radius of the filter in meters (m), corresponding to the size of 
        the well screen or filter that allows water into the well.
    Q_unit : str, optional
        The unit for water output Q, default 'm3/day'. Supported units: 
            'm3/s', 'm3/day'.
    length_unit : str, optional
        The unit for H, h, Ry, and r, default 'm'. Supported units: 'm', 'ft'.
    output_unit : str, optional
        The desired unit for the output hydraulic conductivity K, default 
        'm/day'. Supported units: 'm/day', 'ft/day'.
    log_base : Optional[float], optional
        The base of the logarithm used in the calculation, default is the
        natural logarithm (math.e). For base 10,
        specify 10.0.

    Returns
    -------
    float
        The hydraulic conductivity (K) in meters per day (m/d), which quantifies
        the aquifer's ability to transmit water under the condition of a unit 
        hydraulic gradient.

    Formula
    -------
    The formula for calculating K is given by:

    .. math::
        K = \\frac{Q}{\\pi (H^2 - h^2)} \\ln\\left(\\frac{R_y}{r}\\right)

    where the logarithm base is the natural logarithm (ln).

    References
    ----------
    - Men et al. (2012) and Meng et al. (2021) discuss various expressions 
      for calculating K, including empirical  methods and observations from 
      pumping tests.
    - Theis (1935) and Naderi (2019) detail methods for calculating aquifer 
      parameters using pumping test data,including the Theis and Jacob methods.

    Examples
    --------
    >>> Q = 100  # m^3/day
    >>> H = 20   # m
    >>> h = 15   # m
    >>> Ry = 50  # m
    >>> r = 0.5  # m
    >>> K = calculate_K(Q, H, h, Ry, r, Q_unit='m3/day', length_unit='m', 
                        output_unit='m/day')
    >>> print(f"Hydraulic Conductivity (K): {K} m/day")
    """
    # Convert Q if necessary
    if Q_unit == 'm3/s':
        Q = Q * 86400  # Convert m^3/s to m^3/day

    # Convert lengths if necessary
    if length_unit == 'ft':
        # Convert feet to meters
        H, h, Ry, r = [x * 0.3048 for x in [H, h, Ry, r]]

    # Calculate K
    if log_base == math.e:
        log_func = math.log  # Natural logarithm
    else:
        log_func = lambda x: math.log(x, log_base)  # Logarithm with specified base
    
    K = Q / (math.pi * (H**2 - h**2)) * log_func(Ry / r)
    
    # Convert K if necessary
    if output_unit == 'ft/day':
        K = K / 0.3048  # Convert m/day to ft/day

    return K

def transmissivity(K: float, M: float, K_unit: str = 'm/day', 
                   output_unit: str = 'm2/day') -> float:
    """
    Calculates the transmissivity (T) of an aquifer, given its 
    hydraulic conductivity (K) 
    and thickness (M).

    The mathematical formulation for transmissivity is given by:

    .. math::
        T = K \cdot M

    where:
    - \(T\) is the transmissivity,
    - \(K\) is the hydraulic conductivity, and
    - \(M\) is the aquifer thickness.

    Parameters
    ----------
    K : float
        The hydraulic conductivity of the aquifer material. It represents the ease 
        with which water can move through the aquifer's pore spaces or fractures.
    M : float
        The thickness of the aquifer, representing the vertical extent of the aquifer 
        that contributes to the flow.
    K_unit : str, optional
        The unit of hydraulic conductivity. Supported values are 'm/s' (meters per second) 
        and 'm/day' (meters per day). Default is 'm/day'.
    output_unit : str, optional
        The desired unit for the output transmissivity. Supported values are 'm2/day' 
        (square meters per day) and 'm2/s' (square meters per second). Default is 'm2/day'.

    Returns
    -------
    float
        The transmissivity of the aquifer in the specified output unit.

    Examples
    --------
    >>> K = 0.001  # m/s
    >>> M = 20  # m
    >>> transmissivity = transmissivity(K, M, K_unit='m/s', output_unit='m2/s')
    >>> print(f"Transmissivity: {transmissivity} m^2/s")
    Transmissivity: 0.02 m^2/s
    """
    # Convert hydraulic conductivity to meters per day if specified in meters per second
    if K_unit == 'm/s':
        K = K * 86400  # Convert m/s to m/day
    
    # Calculate transmissivity in m^2/day
    T = K * M
    
    # Convert transmissivity to square meters per second if requested
    if output_unit == 'm2/s':
        T = T / 86400  # Convert m^2/day to m^2/s
    
    return T

def select_base_stratum(
    d: Union[pd.Series, np.ndarray, pd.DataFrame],
    sname: Optional[str] = None,
    stratum: Optional[str] = None,
    return_rate: bool = False,
    return_counts: bool = False,
) -> Union[str, float, List[Tuple[str, int]], Tuple[float, List[Tuple[str, int]]]]:
    """
    Selects the base stratum from the strata column in the logging data, finds
    the most recurrent stratum in the data, and computes the rate of occurrence.

    Parameters
    ----------
    d : Union[pd.Series, np.ndarray, pd.DataFrame]
        Valid data containing the strata. If a DataFrame is passed, `sname` is
        needed to fetch strata values.
    sname : Optional[str], optional
        Name of the column in the DataFrame that contains the strata values.
        Do not confuse `sname` with `stratum`, by default None.
    stratum : Optional[str], optional
        Name of the base stratum. If specified, auto-detection of the base stratum
        is not triggered, by default None.
    return_rate : bool, optional
        Returns the rate of occurrence of the base stratum in the data, by default False.
    return_counts : bool, optional
        Returns each stratum name and the occurrences (count) in the data, by default False.

    Returns
    -------
    Union[str, float, List[Tuple[str, int]], Tuple[float, List[Tuple[str, int]]]]
        Depending on `return_rate` and `return_counts`, it returns the base stratum,
        the rate of occurrence, the counts of all strata, or a combination thereof.

    Raises
    ------
    TypeError
        If `d` is a DataFrame but `sname` is not provided.
    ValueError
        If `sname` is not a column in the DataFrame.
    StrataError
        If the specified `stratum` is not found in the data.

    Examples
    --------
    >>> from gofast.datasets import load_hlogs
    >>> from gofast.geo.hydroutils import select_base_stratum
    >>> data = load_hlogs().frame
    >>> select_base_stratum(data, sname='strata_name')
    'siltstone'
    >>> select_base_stratum(data, sname='strata_name', return_rate=True)
    (0.287292817679558, )
    >>> select_base_stratum(data, sname='strata_name', return_counts=True)
    [('siltstone', 52), ...]
    """
    sdata = _extract_strata_data(d, sname)
    bs, r, c = _get_s_occurrence(sdata, stratum)

    if return_rate and return_counts:
        return r, c
    if return_rate:
        return r
    if return_counts:
        return c
    return bs

def _extract_strata_data(d, sname):
    """
    Extracts strata data from the input, validating the input 
    type and parameters.

    Parameters
    ----------
    d : Union[pd.Series, np.ndarray, pd.DataFrame]
        The data from which to extract strata information.
    sname : Optional[str]
        The name of the column containing strata values, required 
        if `d` is a DataFrame.

    Returns
    -------
    pd.Series or np.ndarray
        The extracted strata data.

    Raises
    ------
    TypeError
        If `d` is a DataFrame but `sname` is not provided.
    ValueError
        If `sname` is not a valid column in the DataFrame.
    """
    if isinstance(d, pd.DataFrame):
        if sname is None:
            raise StrataError("'sname' (strata column name) cannot be None when"
                              " a DataFrame is passed.")
        if sname not in d.columns:
            raise ValueError(f"Name {sname!r} is not a valid column strata name."
                             " Please, check your data.")
        return d[sname]
    elif isinstance(d, pd.Series) or isinstance(d, np.ndarray):
        return d
    else:
        raise ValueError("Input data must be a pandas Series, DataFrame, or a numpy array.")

def _get_s_occurrence(
    sd: Union[pd.Series, np.ndarray],
    bs: Optional[str] = None
) -> Tuple[str, float, List[Tuple[str, int]]]:
    """
    Computes the occurrence of each stratum in the data and identifies the 
    base stratum.

    Parameters
    ----------
    sd : Union[pd.Series, np.ndarray]
        Strata data from which to compute occurrences.
    bs : Optional[str], optional
        Specific stratum to treat as the base stratum, by default None.

    Returns
    -------
    Tuple[str, float, List[Tuple[str, int]]]
        The base stratum, its rate of occurrence, and a list of all strata
        with their counts.

    Raises
    ------
    ValueError
        If a specific stratum `bs` is provided but not found in the data.
    """
    counts = Counter(sd)
    if bs and bs not in counts:
        raise ValueError(f"Stratum {bs!r} not found in the data.")
    if not bs:
        bs = counts.most_common(1)[0][0]
    rate = counts[bs] / sum(counts.values())
    counts_list = counts.most_common()
    return bs, rate, counts_list


class AquiferGroupAnalyzer:
    """
    Analyzes and represents aquifer groups, particularly focusing on the 
    relationship between permeability coefficient ``K`` values and aquifer
    groupings. It utilizes a Mixture Learning Strategy (MXS) to impute missing
    'k' values by creating Naive Group of Aquifer (NGA) labels based on 
    unsupervised learning predictions.
    
    This approach aims to minimize bias by considering the permeability coefficient
    'k' closely tied to aquifer groups. It determines the most representative aquifer
    group for given 'k' values, facilitating the filling of missing 'k' values in the dataset.
    
    Parameters
    ----------
    group_data : dict, optional
        A dictionary mapping labels to their occurrences, representativity, and
        similarity within aquifer groups.
    
    Example
    -------
    See class documentation for a detailed example of usage.
    
    Attributes
    ----------
    group_data : dict
        Accessor for the aquifer group data.
    similarity : generator
        Yields label similarities with NGA labels.
    preponderance : generator
        Yields label occurrences in the dataset.
    representativity : generator
        Yields the representativity of each label.
    groups : generator
        Yields groups for each label.
    """

    def __init__(self, group_data=None):
        """
        Initializes the AquiferGroupAnalyzer with optional group data.
        """
        self.group_data = group_data if group_data is not None else {}

    @property
    def similarity(self):
        """Yields label similarities with NGA labels."""
        return ((label, list(rep_val[1])[0]) for label, rep_val in self.group_data.items())

    @property
    def preponderance(self):
        """Yields label occurrences in the dataset."""
        return ((label, rep_val[0]) for label, rep_val in self.group_data.items())

    @property
    def representativity(self):
        """Yields the representativity of each label."""
        return ((label, round(rep_val[1].get(list(rep_val[1])[0]), 2))
                for label, rep_val in self.group_data.items())

    @property
    def groups(self):
        """Yields groups for each label."""
        return ((label, {k: v for k, v in repr_val[1].items()})
                for label, repr_val in self.group_data.items())

    def __repr__(self):
        """
        Returns a string representation of the AquiferGroupAnalyzer object,
        formatting the representativity of aquifer groups.
        """
        formatted_data = self._format(self.group_data)
        return f"{self.__class__.__name__}({formatted_data})"

    def _format(self, group_dict):
        """
        Formats the representativity of aquifer groups into a string.
        
        Parameters
        ----------
        group_dict : dict
            Dictionary composed of the occurrence of the group as a function
            of aquifer group representativity.
        
        Returns
        -------
        str
            A formatted string representing the aquifer group data.
        """
        formatted_groups = []
        for index, (label, (preponderance, groups)) in enumerate(group_dict.items()):
            label_str = f"{'Label' if index == 0 else ' ':>17}=['{label:^3}',\n"
            preponderance_str = f"{'>32'}(rate = '{preponderance * 100:^7}%',\n"
            groups_str = f"{'>34'}'Groups', {groups}),\n"
            representativity_key, representativity_value = next(iter(groups.items()))
            representativity_str = f"{'>34'}'Representativity', ('{representativity_key}', {representativity_value}),\n"
            similarity_str = f"{'>34'}'Similarity', '{representativity_key}')])],\n"

            formatted_groups.extend([
                label_str, preponderance_str, groups_str,
                representativity_str, similarity_str
            ])
        
        return ''.join(formatted_groups).rstrip(',\n')

