# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
load different data as a function. 
Inspired from the machine learning popular dataset loading 
"""
import warnings
import os
import random
import joblib
from importlib import resources 
from importlib.resources import files
import numpy as np
import pandas as pd 

from ..api.structures import Boxspace  
from ..tools.baseutils import fancier_downloader, check_file_exists 
from ..tools.coreutils import  to_numeric_dtypes, smart_format, get_valid_key
from ..tools.coreutils import  random_sampling, assert_ratio, key_checker 
from ..tools.coreutils import  format_to_datetime, is_in_if, validate_feature
from ..tools.coreutils import convert_to_structured_format, resample_data
from ..tools.coreutils import split_train_test_by_id
from .io import csv_data_loader, _to_dataframe, DMODULE 
from .io import description_loader, DESCR, RemoteDataURL  

__all__= [ "load_iris",  "load_hlogs",  "load_nansha", "load_forensic", 
          "load_jrs_bet", "load_statlog", "load_hydro_metrics", "load_mxs", 
          "load_bagoue"]


def load_hydro_metrics(*, return_X_y=False, as_frame=False, tag=None, 
                       data_names=None,  **kws):
    """
    Load and return the Hydro-Meteorological dataset collected in Yobouakro, 
    S-P Agnibilekro, Cote d'Ivoire(West-Africa).

    This dataset encompasses a comprehensive range of environmental and 
    hydro-meteorological variables, including temperature, humidity, wind speed,
    solar radiation, evapotranspiration, rainfall, and river flow metrics. 
    It's instrumental for studies in environmental science, agriculture, 
    meteorology, hydrology, and climate change research, facilitating the analysis 
    of weather patterns, water resource management, and the impacts of climate 
    variability on agriculture.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a Bunch object. Here, 
        `data` includes all features except the target variable(s), and 
        `target` is typically the river flow measurement considered for predictive 
        modeling tasks.
    as_frame : bool, default=False
        If True, returns a pandas DataFrame for `data` and a pandas Series for
        `target`, facilitating direct interaction with pandas functionalities 
        for data manipulation and analysis.
    tag : str, optional
        A tag to add to the dataset loading for user-defined categorization or 
        filtering, not used in this function but maintained for API compatibility.
    data_names : list of str, optional
        Custom names for the data columns if needed to override the default 
        names derived from the dataset; not utilized in this function but
        preserved for future extension or API consistency.
    **kws : dict, optional
        Additional keyword arguments allowing for future enhancements without 
        affecting the current function signature.

    Returns
    -------
    data : ndarray or DataFrame
        The dataset's features, excluding the target variable. If `as_frame=True`,
        `data` is a pandas DataFrame.
    target : ndarray or Series
        The target variable(s), typically representing the river flow measurements.
        If `as_frame=True`, 
        `target` is a pandas Series.
    Boxspace : Bunch object
        A container holding the dataset details. Returned when `return_X_y` is
        False and `as_frame` is False. 
        It includes:
        - `data`: ndarray, shape (n_samples, n_features) for the feature matrix.
        - `target`: ndarray, shape (n_samples,) for the target variable.
        - `frame`: DataFrame, combining `data` and `target` if `as_frame=True`.
        - `DESCR`: str, a detailed description of the dataset and its context.
        - `feature_names`: list, the names of the feature columns.
        - `target_names`: list, the names of the target column(s).

    Examples
    --------
    To load the dataset as a (data, target) tuple for custom analysis:

    >>> from gofast.datasets import load_hydro_metrics
    >>> data, target = load_hydro_metrics(return_X_y=True)
    >>> print(data.shape)
    (276, 8)  # Assuming 276 instances and 8 features.
    >>> print(target.shape)
    (276,)  # Assuming 276 instances of the target variable.

    To load the dataset as a pandas DataFrame for easier data manipulation
    and exploration:

    >>> df, target = load_hydro_metrics(return_X_y=True, as_frame=True)
    >>> print(df.head())
    # Displays the first five rows of the feature data.
    >>> print(target.head())
    # Displays the first five rows of the target data.

    Note
    ----
    The function is designed with flexibility in mind, accommodating various 
    forms of data analysis and machine learning tasks. By providing options to 
    return the data as arrays or a DataFrame, it enables users to leverage the 
    full power of numpy and pandas for data processing and analysis, respectively.
    """
    data_file = "hydro_metrics.csv"
    data, target, target_classes, feature_names, fdescr = csv_data_loader(
        data_file=data_file, include_headline=True, 
        descr_file="hydro_metrics.rst"
    )
    target_columns = ["flow"]
    # get date column and remove it from dataset.
    date_column= pd.to_datetime(pd.DataFrame (data, columns=feature_names)['date'])
    data= data[:, 1:].astype (float)   
    feature_names = feature_names[1:]
    
    frame, data, target = _to_dataframe( data, feature_names=feature_names,
                                        target_names=target_columns, target=target)
    # set date column index 
    frame.set_index ( date_column, inplace =True )
    data.set_index ( date_column, inplace =True )
        
    if kws.get("split_X_y", False): 
        return _split_X_y(frame,target_columns, as_frame=as_frame, 
            test_ratio=kws.pop("test_ratio", None),
        )
    if return_X_y : 
        return _return_X_y( data, target, as_frame) 
    
    if as_frame: 
        return to_numeric_dtypes(frame)

    return Boxspace(
        data=np.array(data),
        target=np.array(target),
        frame=frame,
        target_classes=target_classes,
        target_names=target_columns,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )

def load_statlog(*, return_X_y=False, as_frame=False, tag=None, 
                 data_names=None, **kws):
    """
    Load and return the Statlog Heart Disease dataset.

    The Statlog Heart dataset is a classic dataset in the machine learning 
    community,used for binary classification tasks to predict the presence of 
    heart disease in patients.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a `Boxspace` object.
    as_frame : bool, default=False
        If True, returns a pandas DataFrame object.
    tag : str, optional
        Tag to add to the dataset loading, not used in this function but kept 
        for compatibility.
    data_names : list of str, optional
        Names of the data columns, not used in this function but kept for 
        compatibility.
    **kws : dict, optional
        Additional keyword arguments not used in this function but kept for 
        compatibility.

    Returns
    -------
    data : ndarray or DataFrame
        The data matrix. If `as_frame=True`, `data` is a pandas DataFrame.
    target : ndarray or Series
        The classification targets. If `as_frame=True`, `target` is a pandas Series.
    Boxspace : Boxspace object
        Object with dataset details, returned when `return_X_y` is 
        False and `as_frame` is False.
        - data : ndarray, shape (n_samples, n_features)
          The data matrix with features.
        - target : ndarray, shape (n_samples,)
          The classification targets.
        - frame : DataFrame
          A DataFrame with `data` and `target` when `as_frame=True`.
        - DESCR : str
          The full description of the dataset.
        - feature_names : list
          Names of the feature columns.
        - target_names : list
          Names of the target columns.
    

    Examples
    --------
    To load the dataset as a (data, target) tuple:

    >>> from gofast.datasets import load_statlog
    >>> data, target = load_statlog(return_X_y=True)
    >>> print(data.shape)
    >>> print(target.shape)

    To load the dataset as a pandas DataFrame:

    >>> df, target = load_statlog(return_X_y=True, as_frame=True)
    >>> print(df.head())
    >>> print(target.head())

    """
    data_file = "statlog_heart.csv"
    data, target, target_classes, feature_names, fdescr = csv_data_loader(
        data_file=data_file, descr_file="statlog_heart.rst"
    )
    feature_names=["age", "sex", "cp", "trestbps", "chol", "fbs",
                   "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
                   "thal"]

    frame = None
    target_columns = ["presence"]
    
    frame, data, target = _to_dataframe(
        data, feature_names=feature_names, target_names=target_columns,
        target=target)
    
    if kws.get("split_X_y", False): 
        return _split_X_y(frame,target_columns, as_frame=as_frame, 
            test_ratio=kws.pop("test_ratio", None),
        )
    
    if return_X_y : 
        return _return_X_y(data, target, as_frame)
    
    if as_frame: 
        return to_numeric_dtypes(frame)
    
    return Boxspace(
        data=np.array(data) ,
        target=np.array(target),
        frame=frame,
        target_classes=target_classes,
        target_names=target_columns,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )

def load_dyspnea(
    *, 
    return_X_y=False, 
    as_frame=False, 
    split_X_y=False, 
    test_ratio =.3 , 
    tag=None , 
    data_names=None,
    objective=None, 
    key=None, 
    n_labels=1, 
    **kws
  ):
    """ 
    Load the dyspnea (difficulty in breathing) dataset, which is designed for 
    medical research and predictive modeling in healthcare, particularly for 
    conditions involving dyspnea.
    
    This function allows for flexible data loading tailored for various 
    analysis needs, including diagnostics, severity assessment, outcome 
    prediction, and symptom-based studies.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` as separate objects. Otherwise,
        returns a Bunch object.
    as_frame : bool, default=False
        If True, the data and target are returned as a pandas DataFrame and Series, 
        respectively. This is useful for further analysis and visualization in a 
        tabular format.
    key : {'raw', 'preprocessed'}, default='preprocessed'
        Specifies which version of the dataset to load. 'raw' for the original 
        dataset and 'preprocessed' for the processed dataset with encoded 
        categorical variables.
    split_X_y : bool, default=False
        If True, splits the dataset into training and testing sets based on the 
        `test_size` parameter. This is useful for model training and evaluation 
        purposes.
    test_size : float, default=0.3
        The proportion of the dataset to include in the test split. It should be 
        between 0.0 and 1.0 and represent the size of the test dataset relative to 
        the original dataset.
    tag : str or None, optional
        An optional tag to filter or categorize the data. Useful for specific 
        analyses within the dataset.
    data_names : list of str or None, optional
        Specific names of datasets to be loaded. If None, all datasets are loaded.
        
    objective : str, 
        The analysis objective indicating the type of target variable(s) 
        to select. Supported objectives include ``"Diagnosis Prediction"``, 
        and ``"Severity Prediction"``, ``"Hospitalization Outcome Prediction"``, 
        ``"Symptom-based Prediction"``, with flexibility in matching these
        objectives even without the exact phrase "Prediction".
        
    n_labels : int, optional
        The number of target labels to select. Defaults to 1 for single-label 
        selection. If greater than 1, a multi-label selection is performed, 
        returning multiple target columns based on the specified objective.
        
    **kws : dict, optional
        Additional keyword arguments to pass to the data loading function. Allows 
        customization of the data loading process.

    Returns
    -------
    boxspace : Bunch object or tuple
        The function returns a Bunch object when `return_X_y` is False. When True,
        returns `(data, target)` 
        as separate objects. The Bunch object has the following attributes:
        - data : ndarray, shape (n_samples, n_features)
          The data matrix with features.
        - target : ndarray, shape (n_samples,)
          The classification targets.
        - frame : DataFrame
          A DataFrame with `data` and `target` when `as_frame=True`.
        - DESCR : str
          The full description of the dataset.
        - feature_names : list
          Names of the feature columns.
        - target_classes : list
          Number of classes in the target. 
        - target_names : list
          Column names in the dataset used as targets.
        - labels_descr, columns_descr : dict
          Formatted dictionaries providing descriptions for categorical labels.

    Examples
    --------
    >>> from gofast.datasets import load_dyspnea
    >>> b = load_dyspnea()
    >>> b.frame 
    >>> b.target 
    >>> X, y = load_dyspnea (return_X_y=True )

    """
    from ..tools.dataops import transform_dates 
    from ._globals import DYSPNEA_DICT, DYSPNEA_LABELS_DESCR
    
    key = get_valid_key(key, "pp", {
        "pp": ("pp", 'preprocessed', 'cleaned', 'transformed', 'structured'), 
        "raw": ["raw", "unprocessed", "source", "original"]
        })

    data_file = f"dyspnea{'+'if key=='pp' else ''}.csv"

    with resources.path (DMODULE, data_file) as p : 
        file = str(p)
    data = pd.read_csv ( file )
    
    target_columns = [
        "respiratory_distress",
    ]
    if objective is not None: 
        target_columns= _select_dyspnea_target_variable(objective, n_labels=n_labels )
        
    feature_names= is_in_if(data, items=target_columns, return_diff=True)
    frame, data, target = _to_dataframe(
         data, feature_names = feature_names, target_names = target_columns,
         )
    frame = transform_dates(to_numeric_dtypes(frame),transform=True)
    if split_X_y: 
        return _split_X_y(frame,target_columns, as_frame=as_frame, 
                          test_ratio=test_ratio)
    if return_X_y : 
        return _return_X_y(data, target, as_frame= as_frame)
    
    if as_frame: 
        return to_numeric_dtypes(frame)
    
    fdescr = description_loader(descr_module=DESCR,descr_file="dyspnea.rst")
    
    return Boxspace(
        data=np.array(data),
        target=np.array(target),
        frame=frame,
        target_names=target_columns,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
        labels_descr=DYSPNEA_LABELS_DESCR,
        columns_descr=DYSPNEA_DICT
    )

def load_hlogs(
    *,
    return_X_y=False,
    as_frame=False,
    key=None,
    split_X_y=False,
    test_ratio=0.3,
    tag=None,
    target_names=None,
    data_names=None,
    **kws
):
    drop_remark = kws.pop("drop_remark", False)
    key, available_sets = _setup_key(key)
    data_file = 'h.h5'
    
    _ensure_data_file(data_file)
    data = _load_data(data_file, key, available_sets)
    if drop_remark:
        data = data.drop(columns="remark")
    
    frame, data, target, target_names = _prepare_data(data, target_names)
    
    if split_X_y:
        return _handle_split_X_y(frame, target_names, test_ratio, as_frame)
    
    if return_X_y:
        return (data, target) if as_frame else (data.values, target.values)
    
    if as_frame:
        return frame
    return _finalize_return(data, target_names, frame, data_file)

def _setup_key(key):
    """
    Ensures the key is valid and returns the appropriate key for data loading.
    """
    default_key = 'h502'
    available_sets = {"h502", "h2601", 'h1102', 'h1104', 'h1405', 'h1602',
                      'h2003', 'h2602', 'h604', 'h803', 'h805'}
    is_keys = set(list(available_sets) + ["*"])
    
    return key_checker(key or default_key, is_keys), available_sets

def _ensure_data_file(data_file):
    """
    Checks if data file exists locally, downloads if not.
    """
    if not check_file_exists(DMODULE, data_file):
        package_path = str(files(DMODULE).joinpath(data_file))
        URL = os.path.join(RemoteDataURL, data_file)
        fancier_downloader(URL, data_file, dstpath=os.path.dirname(package_path))

def _load_data(data_file, key, available_sets):
    """
    Loads data from file based on provided key.
    """
    with resources.path(DMODULE, data_file) as p:
        data_file = str(p)
    if key == '*':
        key = available_sets
    return pd.read_hdf(data_file, key=key) if isinstance(
        key, str) else pd.concat([pd.read_hdf(data_file, key=k) for k in key])

def _prepare_data(data, target_names):
    """
    Prepares the data for return based on parameters.
    """
    feature_names = list(data.columns[:12])
    target_columns = list(data.columns[12:])
    target_names = target_names or target_columns
    validate_feature(data[target_columns], target_names)
    frame, data, target = _to_dataframe(data, target_names, feature_names, )
    frame = to_numeric_dtypes(frame, drop_nan_columns= False )
    # update frame columns if target is specified 
    return frame, data, target, target_names 

def _handle_split_X_y(frame, target_names, test_ratio, as_frame):
    """
    Splits data into training and test sets and formats for return.
    """
    X, Xt = split_train_test_by_id(data=frame, test_ratio=test_ratio,
                                   keep_colindex=False)
    y, yt = X[target_names], Xt[target_names]
    Xt.drop(columns=target_names, inplace=True)
    X.drop(columns=target_names, inplace=True)
    return (X, Xt, y, yt) if as_frame else (
        X.values, Xt.values, y.values, yt.values)

def _finalize_return(data, target_names, frame, data_file):
    """
    Finalizes the return object based on loading preferences.
    """
    fdescr = description_loader(descr_module=DESCR, descr_file="hlogs.rst")
    return Boxspace(
        data=np.array(data),
        target=np.array(frame[target_names]),
        frame=frame,
        target_classes=_get_target_classes(
            frame[target_names], target_names),
        target_names=target_names,
        DESCR=fdescr,
        feature_names=list(data.columns),
        filename=data_file,
        data_module=DMODULE,
    )
load_hlogs.__doc__ =r"""\
Load hydro-logging dataset for hydrogeophysical analysis.

This dataset contains multi-target data suitable for both classification and 
regression tasks in the context of groundwater studies.

Parameters
----------
return_X_y : bool, default=False
    If True, returns `(data, target)` instead of a Bowlspace object. 
    `data` and `target` are described in more detail below.

as_frame : bool, default=False
    If True, data is a pandas DataFrame with appropriate dtypes (numeric).
    The target is a pandas DataFrame or Series, depending on the number of 
    target columns. If `return_X_y` is True, then both `data` and `target` will 
    be pandas DataFrames or Series.

split_X_y: bool, default=False
    If True, splits the data into training and testing sets based on the 
    `test_size` ratio. Returns the sets as `(X, y)` for training and `(Xt, yt)` 
    for testing.

test_ratio: float, default=0.3
    Ratio for splitting data into training and testing sets. A value of 0.3 
    implies a 30% allocation for testing.

target_names: str, optional
    The name of the target column(s) to retrieve. If None, all target columns 
    are collected for multioutput `y`. For specific classification or regression 
    tasks, it's advisable to specify the relevant target name.

key: str, default='h502'
    Identifier for the specific logging data to fetch. Accepts borehole IDs 
    (e.g., "h2601", "*"). If `key='*'`, aggregates all data into a single frame.

drop_observations: bool, default=False
    If True, drops the `remark` column from the logging data.

Returns
-------
data : :class:`~gofast.tools.Boxspace`
    Dictionary-like object with attributes:
    - data: {ndarray, DataFrame} 
      Data matrix; pandas DataFrame if `as_frame=True`.
    - target: {ndarray, Series}
      Classification target; pandas Series if `as_frame=True`.
    - feature_names: list
      Names of dataset columns.
    - target_names: list
      Names of target classes.
    - frame: DataFrame
      DataFrame with `data` and `target` if `as_frame=True`.
    - DESCR: str
      Full description of the dataset.
    - filename: str
      Path to data location.

data, target : tuple
    Returned if `return_X_y` is True. Tuple of ndarray: data matrix (n_samples, n_features)
    and target samples (n_samples,).

X, Xt, y, yt : tuple
    Returned if `split_X_y` is True. Tuple of ndarrays for training (X, y) 
    and testing (Xt, yt) sets, split according to `test_ratio`. The shapes are determined as follows:
    \[
    \text{{shape}}(X, y) = \left(1 - \text{{test\_ratio}}\right) \times (n_{\text{{samples}}}, n_{\text{{features}}}) \times 100
    \]
    \[
    \text{{shape}}(Xt, yt) = \text{{test\_ratio}} \times (n_{\text{{samples}}}, n_{\text{{features}}}) \times 100
    \]

Examples
--------
To explore available target columns without specifying any parameters:

>>> from gofast.datasets.dload import load_hlogs
>>> b = load_hlogs()
>>> b.target_names 
['aquifer_group', 'pumping_level', 'aquifer_thickness', ...]

To focus on specific targets 'pumping_level' and 'aquifer_thickness':

>>> b= load_hlogs(as_frame=True, target_names=['pumping_level', 'aquifer_thickness'])
>>> list(b.columns[-2:])
['pumping_level', 'aquifer_thickness']
"""
def load_nansha (
    *,  return_X_y=False, 
    as_frame =False, 
    key =None, 
    years=None, 
    split_X_y=False, 
    test_ratio=.3 , 
    tag=None, 
    target_names=None, 
    data_names=None, 
    samples=None, 
    seed =None, 
    shuffle =False, 
    **kws
    ): 

    drop_display_rate = kws.pop("drop_display_rate", True)
    key = key or 'b0' 
    # assertion error if key does not exist. 
    key = get_valid_key(key, "hydrology", {
        "hydrology": ("b0",  "hydrogeological", "hydrowork", "hydro"), 
        "geotechnical": ("ns",  "engineering", "engwork", "eng", "geotechnic"), 
        "subsidence": ("ls", "subsidence", "subs", "land_subsidence", "envwork") 
        }, deep_search= False 
        )
    
    # read datafile
    if key in ("hydrology", "geotechnical"):  
        data_file =f"nlogs{'+' if key=='geotechnical' else ''}.csv" 
    else: data_file = "n.npz"
    
    #-----------------------------------------------------------
    if not check_file_exists(DMODULE, data_file): 
        # If file does not exist download it from the remote and 
        # save it to the path 
        package_path = str(files(DMODULE).joinpath(data_file))
        URL= os.path.join( RemoteDataURL, data_file) 
        fancier_downloader (URL,data_file, dstpath = os.path.dirname (package_path)
                       )
    #-------------------------------------------------------------- 
    with resources.path (DMODULE, data_file) as p : 
        data_file = str(p)
     
    if key=='subsidence': 
        # use target_names and years 
        # interchangeability 
        years = target_names or years 
        data , feature_names, target_columns= _get_subsidence_data(
            data_file, years = years or "2022",
            drop_display_rate= drop_display_rate )
        # reset target_names to fit the target columns
        target_names=target_columns 
    else: 
        data = pd.read_csv( data_file )
        # since target and columns are alread set 
        # for land subsidence data, then 
        # go for "ns" and "b0" to
        # set up features and target names
        feature_names = (list( data.columns [:21 ])  + [
            'filter_pipe_diameter']) if key=='hydrology' else list(
                filter(lambda item: item!='ground_height_distance',
                       data.columns)
                ) 
        target_columns = ['static_water_level', 'drawdown', 'water_inflow', 
                          'unit_water_inflow', 'water_inflow_in_m3_d'
                          ] if key=='hydrology' else  ['ground_height_distance']
    # cast values to numeric 
    data = to_numeric_dtypes( data) 
    samples = samples or "*"
    data = random_sampling(data, samples = samples, random_state= seed, 
                            shuffle= shuffle) 
    # reverify the target_names if given 
    # target columns 
    target_names = target_names or target_columns
    # control the existence of the target_names to retreive
    try : 
        validate_feature(data[target_columns], target_names)
    except Exception as error: 
        # get valueError message and replace Feature by target
        verb ="s are" if len(target_columns) > 2 else " is"
        msg = (f" Valid target{verb}: {smart_format(target_columns, 'or')}")
        raise ValueError(str(error).replace(
            'Features'.replace('s', ''), 'Target(s)')+msg)
        
    # read dataframe and separate data to target. 
    frame, data, target = _to_dataframe(
        data, feature_names = feature_names, target_names = target_names, 
        target=data[target_names].values 
        )
    # for consistency, re-cast values to numeric 
    frame = to_numeric_dtypes(frame)
        
    if split_X_y: 
        return _split_X_y(frame,target_columns, as_frame=as_frame, 
                          test_ratio=test_ratio)
    if return_X_y : 
        return _return_X_y(data, target, as_frame)
    
    # return frame if as_frame simply 
    if as_frame: 
        return frame 
    # upload the description file.
    descr_suffix= {"hydrology": '', "geotechnical":"+", "subsidence":"++"}
    fdescr = description_loader(
        descr_module=DESCR,descr_file=f"nansha{descr_suffix.get(key)}.rst")

    return Boxspace(
        data=data.values,
        target=target.values,
        frame=frame,
        target_names=target_names,
        DESCR= fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )
 
load_nansha.__doc__ = """\
Load the Nansha Engineering and Hydrogeological Drilling Dataset.

This dataset contains multi-target information suitable for classification or 
regression problems in hydrogeological and geotechnical contexts.
See more in [1]_. 

Parameters
----------
return_X_y : bool, default=False
    If True, returns (data, target) as separate objects instead of a 
    Bowlspace object.

as_frame : bool, default=False
    If True, data is returned as a pandas DataFrame with appropriate dtypes. 
    The target is also returned as a DataFrame or Series, based on the number 
    of target columns.

split_X_y: bool, default=False
    If True, data is split into a training set (X, y) and a testing set (Xt, yt) 
    according to the test ratio specified.

test_ratio: float, default=0.3
    Ratio for splitting data into training and testing sets (default is 30%).

target_names: str, optional 
    Name(s) of the target column(s) to retrieve. If None, all target columns 
    are included.

key: str, default='b0'
    Identifier for the type of drilling data to fetch. Options include 
    engineering drilling ('ns') and hydrogeological drilling ('b0').

years: str, default="2022"
    Specific year(s) for land subsidence data, ranging from 2015 to 2022.

samples: int, optional 
    Number of samples to fetch from the dataset. Fetches all data if None.

seed: int, optional
    Seed for the random number generator, used when shuffling data.

shuffle: bool, default=False
    If True, shuffles data before sampling.

drop_display_rate: bool, default=True
    If True, removes the display rate column used for image visualization.

Returns
-------
data : :class:`~gofast.tools.box.Boxspace`
    Dictionary-like object with attributes:
    - data: {ndarray, DataFrame} 
      Data matrix; pandas DataFrame if `as_frame=True`.
    - target: {ndarray, Series}
      Classification target; pandas Series if `as_frame=True`.
    - feature_names: list
      Names of dataset columns.
    - target_names: list
      Names of target classes.
    - frame: DataFrame
      DataFrame with `data` and `target` if `as_frame=True`.
    - DESCR: str
      Full description of the dataset.
    - filename: str
      Path to data location.

data, target : tuple
    Returned if `return_X_y` is True. Tuple of ndarray: data matrix 
    (n_samples, n_features) and target samples (n_samples,).

X, Xt, y, yt : tuple
    Returned if `split_X_y` is True. Tuple of ndarrays for training (X, y) 
    and testing (Xt, yt) sets, split according to `test_ratio`.

Examples
--------
To explore available target columns without specifying any parameters:

>>> from gofast.datasets.load import load_nlogs
>>> b = load_nlogs()
>>> b.target_names
['static_water_level', 'drawdown', 'water_inflow', ...]

To focus on specific targets 'drawdown' and 'static_water_level':

>>> b= load_nlogs(as_frame=True, target_names=['drawdown', 'static_water_level'])
>>> list(b.columns[-2:])
['drawdown', 'static_water_level']

To retrieve land subsidence data for specific years with display rate:

>>> n = load_nlogs(key='ls', samples=3, years="2015 2018", drop_display_rate=False)
>>> list(n.frame.columns)
[easting, northing, longitude, 2015, 2018, disp_rate]

References 
-----------
.. [1] Liu, J., Liu, W., Allechy, F.B., Zheng, Z., Liu, R., Kouadio, K.L., 2024.
       Machine learning-based techniques for land subsidence simulation in an 
       urban area. J. Environ. Manage. 352, 17.
       https://doi.org/https://doi.org/10.1016/j.jenvman.2024.120078

"""

def load_bagoue(
        *, return_X_y=False, 
        as_frame=False, 
        split_X_y=False, 
        test_ratio ="30%" , 
        tag=None , 
        data_names=None, 
        **kws
 ):

    data_file = "bagoue.csv"
    data, target, target_classes, feature_names, fdescr = csv_data_loader(
        data_file=data_file, descr_file="bagoue.rst", 
        include_headline= True, 
    )
    frame = None
    target_column = [
        "flow",
    ]
    frame, data, target = _to_dataframe(
        data, feature_names = feature_names, target_names = target_column, 
        target=target)
    frame = to_numeric_dtypes(frame)

    if split_X_y: 
        return _split_X_y(frame,target_column, as_frame=as_frame, 
                          test_ratio=test_ratio)
    if return_X_y : 
        return _return_X_y(data, target, as_frame)
    
    if as_frame: 
        return to_numeric_dtypes(frame)
    
    return Boxspace(
        data=np.array(data),
        target=np.array(target),
        frame=frame,
        target_classes=target_classes, 
        target_name=target_column,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )

load_bagoue.__doc__="""\
Load the Bagoue dataset. 

The Bagoue dataset is a classic and a multi-class classification
dataset. Refer to the description for more details. 

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a 
    :class:`~gofast.tools.box.Boxspace` object. See below for more information 
    about the `data` and `target` object.
as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.
split_X_y: bool, default=False,
    If True, the data is splitted to hold the training set (X, y)  and the 
    testing set (Xt, yt) with the according to the test size ratio.  
test_size: float, default is {{.3}} i.e. 30% (X, y)
    The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
    respectively.
tag, data_names: None
    `tag` and `data_names` do nothing. just for API purpose. They allow 
    to fetch the same data uing the func:`~gofast.datasets.fetch_data` since the 
    latter already holds `tag` and `data_names` as parameters. 

Returns
-------
data: :class:`~gofast.tools.box.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} of shape (150, 4)
        The data matrix. If `as_frame=True`, `data` will be a pandas DataFrame.
    target: {ndarray, Series} of shape (150,)
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    frame: DataFrame of shape (150, 5)
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.
    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.
data, target: tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.
X, Xt, y, yt: Tuple if ``split_X_y`` is True 
    A tuple of two ndarray (X, Xt). The first containing a 2D array of:
        
    .. math:: 
        
        \\text{shape}(X, y) =  1-  \\text{test_ratio} * (n_{samples}, n_{features}) *100
        
        \\text{shape}(Xt, yt)= \\text{test_ratio} * (n_{samples}, n_{features}) *100
        
    where each row representing one sample and each column representing the 
    features. The second ndarray of shape(n_samples,) containing the target 
    samples.
     
Examples
--------
Let's say you are interested in the samples 10, 25, and 50, and want to
know their class name::

>>> from gofast.datasets import load_bagoue
>>> d = load_bagoue () 
>>> d.target[[10, 25, 50]]
array([0, 2, 0])
>>> list(d.target_name)
['flow']   
"""

def load_iris(
        *, return_X_y=False, as_frame=False, tag=None, data_names=None, **kws
        ):
    data_file = "iris.csv"
    data, target, target_classes, feature_names, fdescr = csv_data_loader(
        data_file=data_file, descr_file="iris.rst"
    )
    feature_names = ["sepal length (cm)","sepal width (cm)",
        "petal length (cm)","petal width (cm)",
    ]

    target_column = [
        "target",
    ]
    frame, data, target = _to_dataframe(
        data, feature_names = feature_names, target_names = target_column, 
        target = target)
    
    if kws.get("split_X_y", False): 
        return _split_X_y(frame,target_column, as_frame=as_frame, 
                          test_ratio=kws.pop("test_ratio", None))
    if return_X_y : 
        return _return_X_y(data, target, as_frame)
    
    if as_frame: 
        return to_numeric_dtypes(frame)

    return Boxspace(
        data=data.values,
        target=target,
        frame=frame,
        target_classes=target_classes, 
        target_names=target_column,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
        )


load_iris.__doc__="""\
Load and return the iris dataset (classification).
The iris dataset is a classic and very easy multi-class classification
dataset.

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a BowlSpace object. See
    below for more information about the `data` and `target` object.

as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.
(tag, data_names): None
    `tag` and `data_names` do nothing. just for API purpose and to allow 
    fetching the same data uing the func:`~gofast.data.fetch_data` since the 
    latter already holds `tag` and `data_names` as parameters. 
Returns
-------
data : :class:`~gofast.tools.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} of shape (150, 4)
        The data matrix. If `as_frame=True`, `data` will be a pandas
        DataFrame.
    target: {ndarray, Series} of shape (150,)
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    frame: DataFrame of shape (150, 5)
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.

    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.

(data, target) : tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.

    
Notes
-----
Fixed two wrong data points according to Fisher's paper.
The new version is the same as in R, but not as in the UCI
Machine Learning Repository.

Examples
--------
Let's say you are interested in the samples 10, 25, and 50, and want to
know their class name.
>>> from gofast.datasets import load_iris
>>> data = load_iris()
>>> data.target[[10, 25, 50]]
array([0, 0, 1])
>>> list(data.target_classes)
['setosa', 'versicolor', 'virginica']
"""    
   
def load_mxs (
    *,  return_X_y=False, 
    as_frame =False, 
    key =None,  
    tag =None, 
    samples=None, 
    target_names = None , 
    data_names=None, 
    split_X_y=False, 
    seed = None, 
    shuffle=False,
    test_ratio=.2,  
    **kws):
    """
    Load the dataset after implementing the mixture learning strategy (MXS).
    
    Dataset is composed of 11 boreholes merged with multiple-target that can be 
    used for a classification problem collected as Niming county. 
    
    See more details in :ref:`Documentation User Guide <user_guide>`. 
    
    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bowlspace object. See
        below for more information about the `data` and `target` object.
        
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.
    
    split_X_y: bool, default=False,
        If True, the data is splitted to hold the training set (X, y)  and the 
        testing set (Xt, yt) based on to the `test_ratio` value.
    
    target_names: str, optional 
        the name of the target to retrieve. If ``None`` the full target columns 
        are collected and compose a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
    (tag, data_names): None
        `tag` and `data_names` do nothing. just for API purpose and to allow 
        fetching the same data uing the func:`~gofast.data.fetch_data` since the 
        latter already holds `tag` and `data_names` as parameters. 
        
    samples: int,optional 
       Ratio or number of items from axis to fetch in the data. 
       Default = .5 if `samples` is ``None``.
    
    key: str, default='data'
        Kind of MXS data to fetch. Can also be: 
            
            - "sparse": for a compressed sparsed row matrix format of train set X. 
            - "scale": returns a scaled X using the standardization strategy 
            - "num": Exclusive numerical data and exclude the 'strata' feature.
            - "test": test data `X` and `y` 
            - "train": train data `X` and  `y` with preprocessing already performed
            - "raw": for original dataset X and y  with no preprocessing 
            - "data": Default when key is not supplied. It returns 
              the :class:`Bowlspace` objects.
            
        When k is not supplied, "data" is used instead and return a 
        :class:`Bowlspace` objects. where: 
            - target_map: is the mapping of MXS labels in the target y. 
            - nga_labels: is the y predicted for Naive Group of Aquifer. 
    
    drop_observations: bool, default='False'
        Drop the ``remark`` column in the logging data if set to ``True``. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    shuffle: bool, default =False, 
       If ``True``, borehole data should be shuffling before sampling. 
       
    test_ratio: float, default is 0.2 i.e. 20% (X, y)
        The ratio to split the data into training (X, y) and testing (Xt, yt) set 
        respectively.
        
    Returns
    ---------
    data : :class:`~gofast.tools.Boxspace`
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
        target_map: dict, 
           is the mapping of MXS labels in the target y. 
        nga_labels: arryalike 1D, 
           is the y predicted for Naive Group of Aquifer. 
        frame: DataFrame 
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.
    data, target: tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of shape
        (n_samples, n_features) with each row representing one sample and
        each column representing the features. The second ndarray of shape
        (n_samples,) containing the target samples.
    
    X, Xt, y, yt: Tuple if ``split_X_y`` is True 
        A tuple of two ndarray (X, Xt). The first containing a 2D array of
        training and test data whereas `y` and `yt` are training and test labels.
        The number of samples are based on the `test_ratio`. 
     
    Examples
    --------
    >>> from gofast.datasets.dload import load_mxs  
    >>> load_mxs (return_X_y= True, key ='sparse', samples ='*')
    (<1038x21 sparse matrix of type '<class 'numpy.float64'>'
     	with 8298 stored elements in Compressed Sparse Row format>,
     array([1, 1, 1, ..., 5, 5, 5], dtype=int64))
    
    Load the entire dataset as a pandas DataFrame:
    >>> df = load_mxs_dataset(as_frame=True)
    
    Load the dataset, split into training and testing sets:
    >>> X_train, X_test, y_train, y_test = load_mxs_dataset(split_X_y=True, 
                                                            return_X_y=True)
    """
  
    drop_observations = kws.pop("drop_observations", False)
    target_map = {0: '1', 1: '11*', 2: '2', 3: '2*', 4: '3', 5: '33*'}
    
    # Handling the 'key' parameter and validating against available options
    key = key or 'data'
    key = 'pp' if key.lower() in ('pp', 'preprocessed') else key.lower()
    available_dict = _validate_key(key)

    data_file = 'mxs.joblib'
    with resources.path(DMODULE, data_file) as p:
        data_file = str(p)
    data_dict = joblib.load(data_file)
    samples = ( None if samples =="*" else samples) or .5 # 50%
    data, frame,feature_names, target_names = _prepare_common_dataset(
        data_dict["data"], drop_observations, target_names,  samples, seed, shuffle)
    # Processing based on the specified 'key'
    if key in available_dict:
        Xy = _get_mxs_X_y(available_dict[key], data_dict)
        if Xy:
            Xy = resample_data(*Xy, samples=samples, random_state=seed, shuffle=shuffle)
            if split_X_y and key == 'pp':
                return Xy
            elif split_X_y:
                return _split_and_convert(Xy, test_ratio, seed, shuffle, as_frame)
    elif split_X_y:
        return _split_X_y(frame, target_names, test_ratio, as_frame)
    
    return (data, frame[target_names]) if return_X_y else frame if as_frame else\
        Boxspace(
            data=np.array(data),
            target=np.array(frame[target_names]),
            frame=data,
            target_names=target_names,
            target_map = target_map, 
            nga_labels = data_dict.get('nga_labels'), 
            DESCR= 'Not uploaded yet: Authors are waiting for a publication first.',
            feature_names=feature_names,
            filename=data_file,
            data_module=DMODULE,
            )
def _get_mxs_X_y(key_Xy: tuple, data_dict: dict):
    """
    Retrieve the data and target arrays from a dictionary given specific keys.

    This function extracts the features (X) and target (y) arrays from the provided
    data dictionary using keys specified in `key_Xy`. It is designed to fetch data
    for machine learning models where the data is organized in a dictionary format,
    facilitating easy access to the inputs (X) and outputs (y) for training or testing.

    Parameters
    ----------
    key_Xy : tuple
        A tuple containing two keys. The first key should correspond to the features
        (X) and the second key to the target (y) within the `data_dict`.
    data_dict : dict
        A dictionary where each key corresponds to a specific dataset component. This
        dictionary should contain at least the keys specified in `key_Xy` pointing to
        the features and target data.

    Returns
    -------
    list
        A list containing two elements: the features (X) and the target (y) arrays
        retrieved from the `data_dict` using the keys provided in `key_Xy`. The order
        in the list corresponds to the order of keys in `key_Xy`.

    Example
    -------
    >>> data_dict = {'X_train': array([...]), 'y_train': array([...])}
    >>> key_Xy = ('X_train', 'y_train')
    >>> Xy = _get_mxs_X_y(key_Xy, data_dict)
    >>> Xy[0]  # This will return the features array corresponding to 'X_train'
    >>> Xy[1]  # This will return the target array corresponding to 'y_train'
    """
    if key_Xy is None: 
        return 
    Xy = list(data_dict.get(k) for k in key_Xy)
    return Xy
       
def _validate_key(key: str):
    # Validate the provided 'key' against known dataset variants
    available_data = { 
        "sparse": ('X_csr', 'ymxs_transf'), "scale": ('Xsc', 'ymxs_transf'), 
        "pp": ( 'X_train', 'X_test', 'y_train', 'y_test'),
        'numeric': ( 'Xnm', 'ymxs_transf'), 'raw': ('X', 'y')
        }
    available_keys = list(available_data.keys())  + ['data']
    if key not in available_keys:
        raise ValueError(f"Invalid 'key': {key}. Expected one of {available_keys}.")
        
    return available_data

def _prepare_common_dataset(data, drop_observations, target_names, samples, seed, shuffle):
    # Process the common dataset: handling dropping columns, sampling,
    # and converting to DataFrame
    if drop_observations:
        data = data.drop(columns="remark")
    feature_names = list(data.columns [:13] ) 
    target_columns = list(data.columns [13:])
    target_names = target_names or target_columns

    sampled_data = random_sampling(
        data, samples=samples, random_state=seed, shuffle=shuffle)
    
    frame, processed_data, target = _to_dataframe(
        sampled_data, target_names, list(data.columns [:13] ),
        target=sampled_data[target_names].values)
    return processed_data, to_numeric_dtypes(frame), feature_names, target_names

def _split_and_convert(Xy, test_ratio, seed, shuffle, as_frame):
    # Split data into training and testing sets and optionally convert
    # to structured pandas format
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        *Xy, test_size=test_ratio, random_state=seed, shuffle=shuffle)
    if as_frame:
        return convert_to_structured_format(X_train, X_test, y_train, y_test,
                                            as_frame=True)
    return X_train, X_test, y_train, y_test

def _get_subsidence_data (
        data_file, /, 
        years: str="2022", 
        drop_display_rate: bool=... 
        ): 
    """Read, parse features and target for Nanshan land subsidence data
    
    Parameters 
    ------------
    data_file: str, Pathlike object 
       Full path to the object to read.
    years: str, default=2022 
        year of subsidence data collected. To collect the value of subsidence 
        of all years, set ``years="*"``
        
    drop_display_rate: bool, default=False, 
       Rate of display for visualisation in Goldern software. 
       
    Returns 
    --------
    data, feature_names, target_columns: pd.DataFrame, list
      DataFrame and list of features and targets. 
   
    """
    from ..tools.dataops import read_data
    columns =['easting',
             'northing',
             'longitude',
             'latitude',
             '2015',
             '2016',
             '2017',
             '2018',
             '2019',
             '2020',
             '2021',
             '2022',
             'disp_rate'
             ]
    data = read_data ( data_file, columns = columns )
    if drop_display_rate: 
        data.pop('disp_rate')
        columns =columns [: -1]
        # remove display rate if exists while 
        # it is set to True
        if isinstance ( years, str): 
           years = years.replace ("disp", "").replace (
               "_", ' ').replace ("rate", "")
        elif hasattr ( years, '__iter__'): 
            # maybe list etc 
            years = [str(i) for i in years if not (str(i).find(
                "disp") >= 0 or str(i).find("rate")>= 0) ] 

    if years !="*": 
        years = key_checker (years, valid_keys= data.columns ,
                     pattern =r'[#&*@!,;\s-]\s*', deep_search=True)
        years = [years] if isinstance ( years, str) else years 
    else : 
        years = columns[4:] # target columns 
    # recheck duplicates and remove it  
    years = sorted(set ((str(y) for y in years)))
    feature_names = columns[: 4 ]
    target_columns = years 
    data = data [ feature_names + target_columns ]
    
    return  data,  feature_names, target_columns     
    

def load_forensic( *, 
    return_X_y=False, 
    as_frame=False, 
    key=None, 
    split_X_y=False, 
    test_ratio =.3 ,  
    tag=None , 
    data_names=None, 
    exclude_message_column=True,  
    exclude_vectorized_features=True,  
    **kws
):
    from ._globals import FORENSIC_BF_DICT, FORENSIC_LABELS_DESCR 
    
    key = get_valid_key(key, "pp", {
        "pp": ("pp", 'preprocessed', 'cleaned', 'transformed', 'structured'), 
        "raw": ["raw", "unprocessed", "source", "original"]
        })
    data_file = f"forensic_bf{'+'if key=='pp' else ''}.csv"
    
    with resources.path (DMODULE, data_file) as p : 
        file = str(p)
    data = pd.read_csv ( file )
    
    if exclude_message_column: 
        data = data.drop ( ['message_to_investigators'], axis =1)
    if exclude_vectorized_features and key=='pp': 
        tfidf_columns = [ c for c in data.columns if c.find ('tfid')>=0] 
        data.drop ( columns =tfidf_columns, inplace=True)
    
    frame = None
    target_columns = [
        "dna_use_terrorism_fight",
    ]
    feature_names= is_in_if(data, items=target_columns, return_diff=True)
    
    frame, data, target = _to_dataframe(
         data, feature_names = feature_names, target_names = target_columns, 
         )
    frame = format_to_datetime(to_numeric_dtypes(frame), date_col ='date')
    
    if split_X_y: 
        return _split_X_y(frame, target_columns, test_ratio, as_frame) 
    
    if return_X_y:
        return _return_X_y(data, target, as_frame)
    if as_frame: 
        return to_numeric_dtypes(frame )
    
    fdescr = description_loader(
        descr_module=DESCR,descr_file=data_file.replace (".csv", ".rst"))
    
    return Boxspace(
        data=data,
        target=target,
        frame=frame,
        target_names=target_columns,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
        labels_descr=FORENSIC_LABELS_DESCR,
        colums_descr= FORENSIC_BF_DICT
    )
load_forensic.__doc__="""\
Load and return the forensic dataset for criminal investigation studies.

This function provides access to a forensic dataset, which includes public 
opinion and knowledge regarding DNA databases, their potential use in criminal
investigations, and concerns related to privacy and misuse. The dataset is 
derived from a study on the need for a forensic DNA database in the Sahel region. 
It comes in two forms: raw and preprocessed. The raw data includes the original 
responses, while the preprocessed data contains encoded and cleaned information.

Parameters
----------
return_X_y : bool, default=False
    If True, returns `(data, target)` as separate objects. Otherwise, returns 
    a Bunch object.
as_frame : bool, default=False
    If True, the data and target are returned as a pandas DataFrame and Series, 
    respectively. This is useful for further analysis and visualization in a 
    tabular format.
key : {'raw', 'preprocessed'}, default='preprocessed'
    Specifies which version of the dataset to load. 'raw' for the original 
    dataset and 'preprocessed' for the processed dataset with encoded 
    categorical variables.
split_X_y : bool, default=False
    If True, splits the dataset into training and testing sets based on the 
    `test_size` parameter. This is useful for model training and evaluation 
    purposes.
test_size : float, default=0.3
    The proportion of the dataset to include in the test split. It should be 
    between 0.0 and 1.0 and represent the size of the test dataset relative to 
    the original dataset.
tag : str or None, optional
    An optional tag to filter or categorize the data. Useful for specific 
    analyses within the dataset.
data_names : list of str or None, optional
    Specific names of datasets to be loaded. Does nothing, just for API 
    consistency.  
exclude_message_column : bool, default=True
    If True, the column 'message_to_investigators', which contains textual opinions,
    is excluded from the dataset. Set to False to include this column.
exclude_vectorized_features : bool, default=False
    If True, excludes vectorized features derived from the 'message_to_investigators' 
    column. This is applicable only when `key` is set to 'preprocessed'. Set to False
    to include these vectorized features.
**kws : dict, optional
    Additional keyword arguments to pass to the data loading function. Allows 
    customization of the data loading process.

Returns
-------
boxspace : Bunch object or tuple
    The function returns a Bunch object when `return_X_y` is False. When True,
    returns `(data, target)` 
    as separate objects. The Bunch object has the following attributes:
    - data : ndarray, shape (n_samples, n_features)
      The data matrix with features.
    - target : ndarray, shape (n_samples,)
      The classification targets.
    - frame : DataFrame
      A DataFrame with `data` and `target` when `as_frame=True`.
    - DESCR : str
      The full description of the dataset.
    - feature_names : list
      Names of the feature columns.
    - target_names : list
      Names of the target columns.
    - target_columns : list
      Column names in the dataset used as targets.
    - labels_descr, columns_descr : dict
      Formatted dictionaries providing descriptions for categorical labels.

Examples
--------
>>> from gofast.datasets import load_forensic
>>> forensic_data = load_forensic(key='raw', as_frame=True)
>>> print(forensic_data.frame.head())
>>> forensic_data = load_forensic(key='preprocessed', as_frame=True, 
...                                  exclude_message_column=False, 
...                                  exclude_vectorized_features=True)
>>> print(forensic_data.frame.head())

References
----------
Detailed dataset descriptions can be found in the forensic_bf.rst and 
forensic_bf+.rst files. These documents provide insights into the dataset's 
creation, structure, and attributes.
"""

def load_jrs_bet(
    *, 
    return_X_y=False, 
    as_frame=False,
    key=None, 
    split_X_y=False, 
    test_ratio=0.2, 
    tag=None, 
    data_names=None, 
    N=5, 
    seed=None, 
    **kws
 ):
    # Validating the key
    key = get_valid_key(key, "classic", {
    "raw": ("raw", "unprocessed", "source", "original"), 
    "classic": ("classic_learned", 'traditional_learning', 'machine_learning',
                'classical_learning'), 
    "neural":  ("deep_learned", "deep_processed", "neural_learning",
               "deep_learning", "lstm")
    }, 
        deep_search=False 
    )
    # Loading the dataset
    data_file = "jrs_bet.csv"
    with resources.path(DMODULE, data_file) as p: 
        file = str(p)
    data = pd.read_csv(file)

    # Preparing the dataset
    results = _prepare_jrs_bet_data(
        data, key=key, N=N, split_X_y=split_X_y, return_X_y=return_X_y,
        test_size=test_ratio, seed=seed
        )
    if key in ("neural", "classic"): 
        if split_X_y or return_X_y : 
            return results

    # Processing the data for return
    # target_columns=[] 
    target_columns = ["target" if key == 'classic' else 'winning_numbers']
    if key != 'neural': 
       feature_names = is_in_if(results.columns, items=target_columns,
                                return_diff=True)
    else:
        feature_names= list(results.columns )
        
    frame, data, target = _to_dataframe(results, feature_names=feature_names,
                                        target_names=target_columns)

    if key in ("neural", "raw"):
        frame = format_to_datetime(to_numeric_dtypes(frame), date_col='date')
        
    # Prepare data based on the specified model type
    if split_X_y: 
        return _split_X_y(
            frame, target_columns, test_ratio=test_ratio,as_frame=as_frame )
    
    if return_X_y : 
        return _return_X_y(data, target, as_frame)
        
    if as_frame:
        return frame

    # Loading description
    fdescr = description_loader(descr_module=DESCR, descr_file=data_file.replace(
        ".csv", f"{'+' if key in ('classic', 'neural') else ''}.rst"))

    # Returning a Boxspace object
    return Boxspace(
        data=data,
        target=target,
        frame=frame,
        target_names=target_columns,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE
    )

load_jrs_bet.__doc__="""\
Load and prepare the jrs_bet dataset for machine learning applications.

This function allows for the loading and processing of the `jrs_bet` dataset, 
a rich collection of betting data. The dataset, in its raw form, includes 
information such as dates, locations, and winning numbers. Depending on the 
specified `key`, it can be prepared for classical machine learning models 
or neural network models, with each approach entailing specific preprocessing 
steps to optimize the dataset for the chosen model type.

Parameters
----------
return_X_y : bool, optional
    If True, the function returns the features (X) and target (y) as separate
    objects, rather than a combined dataset. Useful for model training and 
    evaluation. By default, this is set to False.
as_frame : bool, optional
    If True, the data is returned as a pandas DataFrame. This is beneficial for 
    exploratory data analysis and when working with data manipulation libraries. 
    By default, this is set to False.
key : str, optional
    Specifies the type of data preparation to apply. Options include 'raw' for 
    no preprocessing, 'classic' for preparation suitable for classical machine 
    learning models, and 'neural' for preparation tailored to neural network models.
    Defaults to None, which treats the data as 'classic'.
split_X_y : bool, optional
    If True, the dataset is split into features (X) and target (y) during the 
    preparation process. This is especially useful when the dataset needs to be 
    directly fed into a machine learning algorithm. By default, this is set to False.
test_ratio : float, optional
    Defines the proportion of the dataset to be used as the test set. This is 
    important for evaluating the performance of machine learning models. 
    By default, it is set to 0.2.
tag : str, optional
    A custom tag that can be used for dataset processing. This allows for 
    additional customization or identification during the data preparation process.
    By default, this is set to None.
data_names : list, optional
    Custom names for the dataset. This allows for personalization of the 
    dataset name for clarity or specific requirements. By default, it does
    noes nothing, just for API consistency.  
N : int, optional
    Specifies the number of past draws to consider for LSTM models, useful 
    in the context of the 'neural' key. This parameter is crucial for time-series 
    prediction tasks. By default, it is set to 5.
seed : int, optional
    The seed for the random number generator, which is important for reproducibility 
    in tasks like dataset splitting. By default, this is set to None.
kws : dict
    Additional keyword arguments for more advanced or specific dataset 
    preparation needs.

Returns
-------
boxspace : Bunch object or tuple
    Depending on the `return_X_y` parameter, the function returns either a 
    Bunch object (when False) or a tuple of `(data, target)` (when True). 
    The Bunch object contains several attributes providing insights into the 
    dataset, including the data matrix (`data`), classification targets (`target`), 
    a combined DataFrame (`frame` if `as_frame` is True), a full description 
    (`DESCR`), feature names (`feature_names`), and target names (`target_names`).

Examples
--------
>>> from gofast.datasets import load_jrs_bet 
>>> data_box = load_jrs_bet(key="raw")
>>> print(data_box.frame.head())

>>> df = load_jrs_bet(as_frame=True, key='neural')
>>> print(df.head())

>>> X, y = load_jrs_bet(return_X_y=True)
>>> print(X.head(), y.head())

Notes
-----
The jrs_bet dataset is a versatile collection of data suitable for various 
machine learning applications. The raw dataset serves as a foundational bedrock 
for betting analysis. When preprocessed for classical machine learning, it 
undergoes transformation to facilitate models like logistic regression, decision 
trees, or random forests. For neural network applications, especially RNNs like 
LSTM, the dataset is structured to capture temporal patterns and dependencies, 
crucial for deep learning techniques.
"""
def _prepare_jrs_bet_data(data, /, key='classic', N=5, split_X_y=False,
                         return_X_y=False, test_size=0.2, seed=None,
                         ):
    """
    Prepare the jrs_bet dataset for machine learning models.

    This function preprocesses the jrs_bet dataset for use in classical machine
    learning or deep neural network models, specifically LSTM.

    Parameters
    ----------
    data : DataFrame
        The jrs_bet dataset.
    key : str, optional
        The type of model to prepare the data for ('classic' or 'neural'),
        by default 'classic'.
    N : int, optional
        The number of past draws to consider for LSTM, by default 5.
    split_X_y : bool, optional
        If True, split the dataset into features (X) and target (y),
        by default False.
    return_X_y : bool, optional
        If True, return features (X) and target (y) instead of the whole 
        dataset, by default False.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2.
    seed : int, optional
        The seed for the random number generator, by default None.

    Returns
    -------
    DataFrame or tuple of DataFrames
        Processed dataset suitable for the specified model type.
    """
    test_size = assert_ratio(test_size)
    # Preprocessing steps
    data.replace('HOLIDAY', pd.NA, inplace=True)
    data.dropna(inplace=True)
    data['date'] = pd.to_datetime(data['date'])

    # Split and convert winning numbers
    winning_numbers = data['winning_numbers'].str.split('-', expand=True)
    winning_numbers = winning_numbers.apply(pd.to_numeric)

    # Prepare data based on the specified model type
    if key in 'classical':
        return _prepare_for_classical_ml(
            winning_numbers, split_X_y=split_X_y, return_X_y=return_X_y, 
            seed=seed, test_size=test_size
        )
    if key =="neural":
        return _prepare_for_neural_networks(
            winning_numbers, data=data, N=N, split_X_y=split_X_y, 
            return_X_y=return_X_y, test_size=test_size
        )
    
    return data 

def _prepare_for_neural_networks(
        winning_numbers, /, data, N=5, split_X_y=False, 
        return_X_y=False, test_size=0.2
   ):
    """
    Prepare the jrs_bet dataset specifically for recurrent neural network models.

    This function processes the winning_numbers from the jrs_bet dataset 
    for application in recurrent neural network models such as LSTM.

    Parameters
    ----------
    winning_numbers : DataFrame
        Winning numbers from the jrs_bet dataset.
    data : DataFrame
        The original jrs_bet dataset with additional columns such as 'date'.
    N : int, optional
        The number of past draws to use for sequence generation, by default 5.
    split_X_y : bool, optional
        If True, split the dataset into features (X) and target (y), 
        by default False.
    return_X_y : bool, optional
        If True, return features (X) and target (y) instead of the whole 
        dataset, by default False.
    test_size : float, optional
        The proportion of the dataset to include in the test split,
        by default 0.2.

    Returns
    -------
    DataFrame or tuple of arrays
        Prepared sequences and corresponding targets suitable for RNN models.
    """
    from sklearn.preprocessing import MinMaxScaler

    # Concatenate date and winning numbers
    historical_data = pd.concat([data['date'], winning_numbers], axis=1)
    
    if not split_X_y and not return_X_y: 
        return historical_data

    # Generating sequences for LSTM
    sequences = [winning_numbers.iloc[i:i+N].values.flatten()
                 for i in range(len(winning_numbers) - N)]

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    sequences_normalized = scaler.fit_transform(sequences)

    # Preparing features and targets
    X = np.array(sequences_normalized)
    y = np.array([sequences_normalized[i+1] for i in range(
        len(sequences_normalized)-1)])

    # Reshaping for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Data splitting
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (X, y) if return_X_y else (X_train, X_test, y_train, y_test)
 
def _prepare_for_classical_ml(
        winning_numbers, /, split_X_y=False, return_X_y=False,
        seed=None, test_size=0.2):
    """
    Prepare the jrs_bet dataset for classical machine learning models.

    This function processes the winning_numbers from the jrs_bet dataset for 
    application in classical machine learning models.

    Parameters
    ----------
    winning_numbers : DataFrame
        Winning numbers from the jrs_bet dataset.
    split_X_y : bool, optional
        If True, split the dataset into features (X) and target (y), 
        by default False.
    return_X_y : bool, optional
        If True, return features (X) and target (y) instead of the whole 
        dataset, by default False.
    seed : int, optional
        The seed for the random number generator, by default None.
    test_size : float, optional
        The proportion of the dataset to include in the test split,
        by default 0.2.

    Returns
    -------
    DataFrame or tuple of DataFrames
        Prepared dataset suitable for classical machine learning models.
    """
    from sklearn.model_selection import train_test_split 
    # Frequency and recency analysis
    frequency_analysis = winning_numbers.apply(pd.Series.value_counts
                                               ).fillna(0).sum(axis=1)
    recency = {number: index for index, row in winning_numbers.iterrows() 
               for number in row}

    # Converting recency to DataFrame and merging with frequency data
    recency_df = pd.DataFrame.from_dict(
        recency, orient='index', columns=['most_recent_draw']
        ).reset_index().rename(columns={'index': 'number'})
    features_df = pd.merge(frequency_analysis.reset_index().rename(
        columns={'index': 'number', 0: 'frequency'}), recency_df, on='number')

    # Target data preparation
    all_numbers = set(range(1, 100))
    target_data = winning_numbers.isin(all_numbers).any(
        axis=1).astype(int).to_frame('target')

    # Merging features with target data
    ml_dataset = pd.merge(features_df, target_data,
                          left_on='number', right_index=True)
    ml_dataset['target'] = ml_dataset['target'].shift(-1)

    # Final dataset preparation
    final_ml_dataset = ml_dataset.dropna().drop(columns=['number'])

    if not split_X_y and not return_X_y: 
        return final_ml_dataset

    # Dataset splitting
    X = final_ml_dataset.drop('target', axis=1)
    y = final_ml_dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    
    return (X, y) if return_X_y else (X_train, X_test, y_train, y_test)
    
def _select_dyspnea_target_variable(
        objective, data=None, n_labels=1, raise_exception=False, verbose=0):
    """
    Selects and returns target variable(s) based on a specified objective from 
    a pandas DataFrame, accommodating both single-label and multi-label 
    scenarios and allowing for flexible objective specification.

    Parameters
    ----------
    objective : str
        The analysis objective indicating the type of target variable(s) 
        to select. Supported objectives include ``"Diagnosis Prediction"``, 
        and ``"Severity Prediction"``, ``"Hospitalization Outcome Prediction"``, 
        ``"Symptom-based Prediction"``, with flexibility in matching these
        objectives even without the exact phrase "Prediction".
    data : pandas.DataFrame, optional
        The dataset from which to select the target variable(s). If provided, 
        the function returns the selected target column(s) as a DataFrame or 
        Series. If not provided, returns the name(s) 
        of the target column(s).
    n_labels : int, optional
        The number of target labels to select. Defaults to 1 for single-label 
        selection. If greater than 1, a multi-label selection is performed, 
        returning multiple target columns based on the specified objective.
    raise_exception : bool, optional
        If True, raises an exception when `n_labels` exceeds the number of 
        available target columns for the specified objective. If False, a 
        warning is issued instead, and `n_labels` is set to the maximum available. 
        Defaults to False.
    verbose : int, optional
        If greater than 0, enables the function to print warnings when 
        applicable. Defaults to 0.

    Returns
    -------
    pandas.DataFrame, pandas.Series, str, or list of str
        Depending on the inputs, returns either the name(s) of the target 
        column(s) (str or list of str) or the target column(s) themselves as a 
        pandas DataFrame or Series.

    Raises
    ------
    ValueError
        If an invalid objective is provided or if `n_labels` exceeds the number
        of available target columns for the specified objective and 
        `raise_exception` is True.

    Examples
    --------
    >>> dataset = pd.DataFrame({
    ...     "diagnosis_covid_19": [0, 1],
    ...     "nyha_intensity": [2, 3],
    ...     "outcome_of_hospitalization": [1, 0]
    ... })
    >>> select_target_variable("Diagnosis Prediction", data=dataset)
    diagnosis_covid_19    0
    1
    Name: diagnosis_covid_19, dtype: int64

    >>> select_target_variable("Severity Prediction", n_labels=2)
    ['nyha_intensity', 'respiratory_distress']
    """
    # Adjusted for flexible objective matching 
    # Normalize whitespace and case
    adjusted_objective = " ".join(objective.lower().split())  
    # Allow omission of 'prediction'
    adjusted_objective = adjusted_objective.replace(" prediction", "")  

    # Define the mapping of adjusted objectives to potential target columns
    DYSPNEA_TARGETS = {
        "diagnosis": [
            "diagnosis_pneumonitis", "diagnosis_asthma_attack",
            "diagnosis_pulmonary_tuberculosis", "diagnosis_covid_19",
            "diagnosis_heart_failure", "diagnosis_copd",
            "diagnosis_bronchial_cancer", "diagnosis_pulmonary_fibrosis",
            "diagnostic_other"
        ],
        "severity": ["nyha_intensity", "glasgow_score", "respiratory_distress"],
        "symptom-based": ["cough", "fever", "asthenia", "heart_failure"],
        "hospitalization outcome": ["outcome_of_hospitalization"]
    }

    # Attempt to find matching targets for the objective
    potential_columns = []
    for key, values in DYSPNEA_TARGETS.items():
        if key in adjusted_objective:
            potential_columns = values
            break

    if not potential_columns:
        raise ValueError(f"Invalid objective provided: '{objective}'. Supported "
                         f"objectives are: {', '.join(DYSPNEA_TARGETS.keys())}.")

    if n_labels > len(potential_columns):
        if raise_exception:
            raise ValueError(f"Requested number of labels ({n_labels}) exceeds "
                             f"available targets for '{objective}'.")
        else:
            if verbose > 0:
                warnings.warn(
                    f"Requested number of labels ({n_labels}) exceeds available "
                    f"targets. Adjusting to {len(potential_columns)}.")
            n_labels = len(potential_columns)

    selected_columns = random.sample(potential_columns, k=n_labels)

    if data is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The `data` parameter must be a pandas DataFrame.")
        return data[selected_columns] if n_labels > 1 else data[selected_columns[0]]

    return selected_columns if n_labels > 1 else selected_columns[0]
       
def _get_target_classes(target, target_columns):
    """
    Determines the classification type of the target variable(s) based
    on the unique values present. Identifies if the target is suited for
    classification or regression tasks, based on unique class counts or
    target type.

    Parameters
    ----------
    target : array-like
        The target variable(s) from the dataset, used to determine the
        prediction task type (classification or regression).
    target_columns : list of str
        Names of the target columns, used as fallback for regression tasks
        or when class count exceeds threshold.

    Returns
    -------
    classification_info : list, str, or 'undetermined'
        - Returns a list of unique classes for binary or multiclass with
          10 or fewer classes.
        - Returns target column names if class count is over 10, assuming
          regression.
        - Returns a string indicating non-binary/multiclass target type.
        - Returns 'undetermined' if target type cannot be established.

    Notes
    -----
    Utilizes `type_of_target` from core utilities to assess target variable
    type. Designed to distinguish between classification and regression
    tasks for appropriate model selection and preprocessing.
    """
    from ..tools.coreutils import type_of_target
    try:
        target_type = type_of_target(target)
        unique_classes = np.unique(target)

        if target_type in ("binary", "multiclass"):
            if len(unique_classes) <= 10:
                return list(unique_classes)
            else:
                return target_columns
        else:
            return f"is {target_type}"
    except Exception as e: # noqa
        # print(f"Error determining target classification: {e}")
        return 'undetermined'

def _split_X_y(frame, target_columns, test_ratio=None, as_frame=False):
    """
    Splits the provided dataset into training and testing subsets, further 
    separating features from target variables. 
    
    This operation facilitates machine learn training and testing sets, along 
    with clear separation of input features and target outcomes.

    Parameters
    ----------
    frame : pandas.DataFrame
        The dataset to be split, containing both the features and the target
        variables. It is assumed that this dataframe is preprocessed and ready
        for splitting.
    target_columns : list of str
        Names of the columns in `frame` that should be treated as the target 
        variables. These columns are separated from the features during the
        split process.
    test_ratio : float,default='20%'
        The proportion of the dataset to be allocated to the test subset, 
        expressed as a float between 0.0 and 1.0.
    as_frame : bool, optional, default=False
        Flag indicating the format of the returned datasets. If True, the 
        datasets are returned as pandas DataFrame and Series objects. If False, 
        they are converted to numpy arrays, suitable for use in scenarios where 
        DataFrame structures are not required.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        A tuple containing the split datasets. `X_train` and `X_test` are the 
        subsets of features for training and testing, respectively. 
        `y_train` and `y_test` represent the corresponding target variables 
        for each subset. The types of these objects are determined by the 
        `return_as_frame` parameter.

    Raises
    ------
    ValueError
        If `perform_split` is False, indicating an incorrect use of the 
        function for its intended purpose.

    Examples
    --------
    >>> df = pd.DataFrame({...})
    >>> target_cols = ['target']
    >>> X_train, X_test, y_train, y_test = _split_X_y(
        df, target_cols, 0.25, return_as_frame=True)
    >>> print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    Note
    ----
    This function is designed to be flexible, accommodating various types of 
    data analysis and machine learning workflows. By providing control over 
    the split ratio and the data format, it enables seamless integration into 
    different 
    data processing pipelines.
    """
    test_ratio= assert_ratio(test_ratio) if test_ratio is not None else 0.2
    # Execute the dataset splitting based on ID, preserving the order or 
    # structure as required
    X, Xt = split_train_test_by_id(
        data=frame, test_ratio=test_ratio, keep_colindex=False)
    # Separate the target variables from the features in both training
    # and testing sets
    y = X [target_columns]
    yt = Xt[target_columns]
    X= X.drop (columns =target_columns )
    Xt=Xt.drop (columns =target_columns )
    # Return the split datasets in the specified format
    
    return convert_to_structured_format(
        X, Xt, y, yt, as_frame=as_frame, skip_sparse= True )

def _return_X_y (data, target, as_frame):
    """
    Formats the data and target variables for return, based on the specified
    format preference. This function allows for the flexible return of data
    either as pandas DataFrame/Series or as numpy arrays, facilitating
    subsequent analysis or model feeding processes.

    Parameters
    ----------
    data : pandas.DataFrame or array-like
        The input features of the dataset, either as a DataFrame or any
        structure convertible to a numpy array.
    target : pandas.Series or array-like
        The target variables of the dataset, similarly as a Series or
        convertible structure.
    as_frame : bool
        Flag indicating the desired return format. If True, returns the
        data and target as provided (suitable for DataFrame/Series inputs).
        If False, converts and returns them as numpy arrays.

    Returns
    -------
    tuple
        A tuple containing the data and target in the requested format.
        - If `return_as_frame` is True, the original input formats of `data`
          and `target` are returned.
        - If False, `data` and `target` are returned as numpy arrays, using
          np.array conversion.

    Example
    -------
    >>> df, target = _return_X_y(df, series, True)
    >>> X, y = _return_X_y(list_of_lists, list_of_targets, False)
    """
    if as_frame:
        if isinstance (target, pd.DataFrame ) and len(target.columns)==1: 
             try : target= pd.Series (data = np.squeeze(target), 
                                      name =target.columns[0])
             except: pass # do nothing
             
        return data, target
    else:
        return np.array(data), np.array (np.squeeze(target))


    
    
    
    
    
    
    
    
    
