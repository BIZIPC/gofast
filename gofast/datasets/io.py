# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio
"""
Base IO functions for managing all the datasets.
Provides functions for loading, retrieving, and managing dataset files, 
including CSV and text files, as well as dataset descriptions.
"""

import os 
import csv 
import shutil 
import numpy as np 
import pandas as pd 
from pathlib import Path 
from importlib import resources
from collections import namedtuple
from ..api.structures import Boxspace  
from ..tools.baseutils import is_readable 
from ..tools.coreutils import random_state_validator, is_iterable
from ..tools.coreutils import exist_features

__all__=[
    'csv_data_loader',
    'description_loader',
    'get_data',
    'remove_data',
    'text_files_loader'
 ]

DMODULE = "gofast.datasets.data" ; DESCR = "gofast.datasets.descr"

# create a namedtuple for remote data and url 
RemoteMetadata = namedtuple("RemoteMetadata", ["file", "url", "checksum"])
RemoteDataURL ='https://raw.githubusercontent.com/earthai-tech/gofast/master/gofast/datasets/data/'

def get_data(data =None) -> str: 
    if data is None:
        data = os.environ.get("GOFAST_DATA", os.path.join("~", "gofast_data"))
    data = os.path.expanduser(data)
    os.makedirs(data, exist_ok=True)
    return data

get_data.__doc__ ="""\
Get the data from home directory  and return gofast data directory 

By default the data directory is set to a folder named 'gofast_data' in the
user home folder. Alternatively, it can be set by the 'gofast_DATA' environment
variable or programmatically by giving an explicit folder path. The '~'
symbol is expanded to the user home folder.
If the folder does not already exist, it is automatically created.

Parameters
----------
data : str, default=None
    The path to gofast data directory. If `None`, the default path
    is `~/gofast_data`.
Returns
-------
data: str
    The path to gofast data directory.

"""
def remove_data(data=None): #clear 
    """Delete all the content of the data home cache.
    
    Parameters
    ----------
    data : str, default=None
        The path to gofast data directory. If `None`, the default path
        is `~/gofast_data`.
    """
    data = get_data(data)
    shutil.rmtree(data)


def _to_dataframe(data, target_names=None, feature_names=None, target=None):
    """
    Refines input data into a structured DataFrame, distinguishing between
    features and targets based on specified column names or a separate target array.

    Parameters
    ----------
    data : str, path-like object, DataFrame, or array-like
        Source data, which can be a file path, a DataFrame, or an array-like object.
    tnames : list of str, optional
        Column names in `data` designated as targets. If specified, these columns
        are separated into the target DataFrame `y`.
    feature_names : list of str, optional
        Column names to retain as features in the output DataFrame `X`. If not provided,
        all columns except those specified in `tnames` are used.
    target : ndarray, pd.Series, pd.DataFrame, or None, optional
        Explicit target data. If provided, this will be used as the target DataFrame `y`,
        potentially in addition to any targets specified in `tnames`.

    Returns
    -------
    DataFrame
        The combined DataFrame including both features and target(s),
        if applicable.
    DataFrame
        The features DataFrame `X`.
    DataFrame or None
        The target DataFrame `y`, if targets are specified either through
        `tnames` or `target`.

    Examples
    --------
    >>> from gofast.dataset.io import _to_dataframe
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'feature1': [1, 2, 3],
    ...     'feature2': [4, 5, 6],
    ...     'target': [0, 1, 0]
    ... })
    >>> feature_names = ['feature1', 'feature2']
    >>> tnames = ['target']
    >>> combined, X, y = _to_dataframe(data, tnames, feature_names)
    >>> X
       feature1  feature2
    0         1         4
    1         2         5
    2         3         6
    >>> y
       target
    0      0
    1      1
    2      0
    """
    
    if isinstance(data, (str, bytes)):
        data = is_readable(data)  # Assumes CSV; adjust as necessary.
    elif isinstance(data, np.ndarray) or isinstance(data, list):
        try:
            data = pd.DataFrame(data, columns=feature_names)
        except: data =pd.DataFrame(data )
        
    if target_names is not None: 
        target_names = is_iterable (target_names, exclude_string=True, transform=True)
        
    feature_data = data.drop(columns=target_names, errors='ignore') if target_names else data
    try:
        exist_features(data, target_names) # check whether tnames is on data 
    except: target_from_data= pd.DataFrame()
    else: 
        target_from_data = data[target_names] if target_names else pd.DataFrame()

    if target is not None:
        if isinstance(target, (pd.Series, pd.DataFrame, np.ndarray)):
            if isinstance(target, np.ndarray):
                target = pd.DataFrame(target, columns=target_names
                                      ) if target_names else pd.DataFrame(target)
            elif isinstance(target, pd.Series):
                target = pd.DataFrame(target)
            # No need to filter out duplicates here, as we've preemptively handled them.
            # Concatenate target_from_data and new target, handling duplicate columns
            # target = pd.concat([target_from_data, target], axis=1
            #                     ).loc[:, ~target.columns.duplicated()]
            
            # Before concatenating, ensure that the columns from 
            #  'target_from_data' and 'target' do not overlap.
            # This avoids the boolean indexing issue by never creating 
            # duplicate columns in the first place.
            # First, identify the duplicate columns that would result 
            # from concatenation.
            if not target_from_data.empty:
                duplicate_columns = target_from_data.columns.intersection(
                    target.columns)
                # Drop duplicate columns from one of the DataFrames
                # before concatenation.
                target_from_data = target_from_data.drop(
                    columns=duplicate_columns, errors='ignore')
            # Now, concatenate 'target_from_data' and 'target' without 
            # risking duplicate columns.
            target = pd.concat([target_from_data, target], axis=1)
        else:
            raise TypeError("The 'target' parameter must be an ndarray,"
                            " pd.Series, pd.DataFrame, or None.")
    else:
        target = target_from_data
        
    # for consistency, recheck the feature data. 
    feature_data = feature_data [
        feature_names] if feature_names is not None else feature_data 
    # Ensure that `target` is either a DataFrame or None for 
    # consistent return types
    target = target if not target.empty else None
    combined = pd.concat([feature_data, target], axis=1
                         ) if target is not None else feature_data

    return combined, feature_data, target


def __to_dataframe(data, tnames=None , feature_names =None, target =None ): 
    """ Validate that data is readable by pandas rearder and parse the data.
     then separate data from training to target. Be sure that the target 
     must be included to the dataframe columns 
     
    :param data: str, or path-like object 
        data file to read or dataframe
    :param tnames: list of str 
        name of target that might be a column of the target frame. 
        
    :param feature_names: List of features to selects. Preferably 
        should be include in the dataframe columns 
    :params target: Ndarray or array-like, 
        A target for supervised learning . Can be ndarray -for muliclass 
        output and array-like for singular label 
    
    :returns: 
        dataframe combined and X, y (X= features frames, y = target frame)
    """
    # if a ndarray is given ,then convert to dataframe 
    # by adding feature names. 
    d0, y = None , None 
    if isinstance (data, np.ndarray ):
        d0 = pd.DataFrame(data = data, columns = feature_names)
    else : 
        # read with pandas config parsers including the target 
        df = is_readable(data)  
    # if tnames are given convert the array 
    # of target  to a target frame 
    if  ( 
        d0 is not None 
        and tnames is not None 
        and target is not None
            ) : 
        
        tnames = is_iterable(tnames, exclude_string= True, transform =True)
        target = pd.DataFrame(data =target , columns =tnames ) 
        # if target is not None: 
        df = pd.concat ([d0, target], axis =1 )# stack to columns 

    X = df[feature_names ]  
    
    if tnames is not None :
        y = df [tnames[0]] if len(tnames)==1 else df [tnames]
    
    return df, X, y 

def csv_data_loader(
    data_file,*, data_module=DMODULE, descr_file=None, descr_module=DESCR,
    include_headline =False, 
):
    feature_names= None # expect to extract features as the head columns 
    cat_feature_exist = False # for homogeneous feature control
    #with resources.files(data_module).joinpath(
    #        data_file).open('r', encoding='utf-8') as csv_file: #python >3.8
    with resources.open_text(data_module, data_file, 
                             encoding='utf-8') as csv_file: # Python 3.8
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0]) ; n_features = int(temp[1])
        # remove empty if exist in the list 
        tnames = np.array(list(filter(None,temp[2:])))
        # to prevent an error change the datatype to string 
        data = np.empty((n_samples, n_features)).astype('<U99')
        target = np.empty((n_samples,), dtype=float)
        if include_headline: 
            # remove target  expected to be located to the last columns
            feature_names = list(next(data_file)[:-1])  
            
        #XXX TODO: move the target[i] to try exception if target is str 
        for i, ir in enumerate(data_file):
            try : 
                data[i] = np.asarray(ir[:-1], dtype=float)
            except ValueError: 
                data[i] = np.asarray(ir[:-1], dtype ='<U99' ) # dont convert anything 
                cat_feature_exist = True # mean cat feature exists
            target[i] = np.asarray(ir[-1], dtype=float)
            
    if not cat_feature_exist: 
        # reconvert the datatype to float 
        data = data.astype (float)
    # reconvert target if problem is classification rather than regression 
    try : 
        target =target.astype(int )
    except : pass  
    
    if descr_file is None:
        return data, target, tnames, feature_names
    else:
        assert descr_module is not None
        descr = description_loader(descr_module=descr_module, descr_file=descr_file)
        return data, target, tnames, feature_names,  descr

csv_data_loader.__doc__="""\
Loads `data_file` from `data_module with `importlib.resources`.

Parameters
----------
data_file: str
    Name of csv file to be loaded from `data_module/data_file`.
    For example `'bagoue.csv'`.
data_module : str or module, default='gofast.datasets.data'
    Module where data lives. The default is `'gofast.datasets.data'`.
descr_file_name : str, default=None
    Name of rst file to be loaded from `descr_module/descr_file`.
    For example `'bagoue.rst'`. See also :func:`description_loader`.
    If not None, also returns the corresponding description of
    the dataset.
descr_module : str or module, default='gofast.datasets.descr'
    Module where `descr_file` lives. See also :func:`description_loader`.
    The default is `'gofast.datasets.descr'`.
Returns
-------
data : ndarray of shape (n_samples, n_features)
    A 2D array with each row representing one sample and each column
    representing the features of a given sample.
target : ndarry of shape (n_samples,)
    A 1D array holding target variables for all the samples in `data`.
    For example target[0] is the target variable for data[0].
target_names : ndarry of shape (n_samples,)
    A 1D array containing the names of the classifications. For example
    target_names[0] is the name of the target[0] class.
descr : str, optional
    Description of the dataset (the content of `descr_file_name`).
    Only returned if `descr_file` is not None.

"""

def description_loader(descr_file, *, descr_module=DESCR, encoding ='utf8'):
    # fdescr=resources.files(descr_module).joinpath(descr_file).read_text(
    #     encoding=encoding)
    fdescr = resources.read_text(descr_module, descr_file, encoding= 'utf8')
    return fdescr

description_loader.__doc__ ="""\
Load `descr_file` from `descr_module` with `importlib.resources`.
 
Parameters
----------
descr_file_name : str, default=None
    Name of rst file to be loaded from `descr_module/descr_file`.
    For example `'bagoue.rst'`. See also :func:`description_loader`.
    If not None, also returns the corresponding description of
    the dataset.
descr_module : str or module, default='gofast.datasets.descr'
    Module where `descr_file` lives. See also :func:`description_loader`.
    The default  is `'gofast.datasets.descr'`.
     
Returns
-------
fdescr : str
    Content of `descr_file_name`.

"""

def text_files_loader(
    container_path,*,description=None,categories=None,
    load_content=True,shuffle=True,encoding=None,decode_error="strict",
    random_state=42, allowed_extensions=None,
):
    target = []
    target_names = []
    filenames = []

    folders = [
        f for f in sorted(os.listdir(container_path)) if os.path.isdir(
            os.path.join(container_path, f))
    ]

    if categories is not None:
        folders = [f for f in folders if f in categories]

    if allowed_extensions is not None:
        allowed_extensions = frozenset(allowed_extensions)

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = os.path.join(container_path, folder)
        files = sorted(os.listdir(folder_path))
        if allowed_extensions is not None:
            documents = [
                os.path.join(folder_path, file)
                for file in files
                if os.path.splitext(file)[1] in allowed_extensions
            ]
        else:
            documents = [os.path.join(folder_path, file) for file in files]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = random_state_validator(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    if load_content:
        data = []
        for filename in filenames:
            data.append(Path(filename).read_bytes())
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        return Boxspace(
            data=data,
            filenames=filenames,
            target_names=target_names,
            target=target,
            DESCR=description,
        )

    return Boxspace(
        filenames=filenames, target_names=target_names, target=target,
        DESCR=description
    )
      
text_files_loader.__doc__ ="""\
Load text files with categories as subfolder names.

Individual samples are assumed to be files stored a two levels folder
structure such as the following::

    container_folder/
        category_1_folder/
            file1.txt
            file2.txt
            ...
            file30.txt
        category_2_folder/
            file31.txt
            file32.txt
            ...
            
The folder names are used as supervised signal label names. The individual
file names are not important.

In addition, if load_content is false it does not try to load the files in memory.
If you set load_content=True, you should also specify the encoding of the
text using the 'encoding' parameter. For many modern text files, 'utf-8'
will be the correct encoding. If you want files with a specific file extension 
(e.g. `.txt`) then you can pass a list of those file extensions to 
`allowed_extensions`.

Parameters
----------
container_path : str
    Path to the main folder holding one subfolder per category.
description : str, default=None
    A paragraph describing the characteristic of the dataset: its source,
    reference, etc.
categories : list of str, default=None
    If None (default), load all the categories. If not None, list of
    category names to load (other categories ignored).
load_content : bool, default=True
    Whether to load or not the content of the different files. If true a
    'data' attribute containing the text information is present in the data
    structure returned. If not, a filenames attribute gives the path to the
    files.
shuffle : bool, default=True
    Whether or not to shuffle the data: might be important for models that
    make the assumption that the samples are independent and identically
    distributed (i.i.d.), such as stochastic gradient descent.
encoding : str, default=None
    If None, do not try to decode the content of the files (e.g. for images
    or other non-text content). If not None, encoding to use to decode text
    files to Unicode if load_content is True.
decode_error : {'strict', 'ignore', 'replace'}, default='strict'
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. Passed as keyword
    argument 'errors' to bytes.decode.
random_state : int, RandomState instance or None, default=42
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
allowed_extensions : list of str, default=None
    List of desired file extensions to filter the files to be loaded.

Returns
-------
data : :class:`~gofast.tools.Boxspace`
    Dictionary-like object, with the following attributes.
    data : list of str
      Only present when `load_content=True`.
      The raw text data to learn.
    target : ndarray
      The target labels (integer index).
    target_names : list
      The names of target classes.
    DESCR : str
      The full description of the dataset.
    filenames: ndarray
      The filenames holding the dataset.
"""

























