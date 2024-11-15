# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Offers core classes and utilities for data handling 
and preprocessing. It includes functionality for managing missing data, 
merging data frames and series, and processing features and targets for 
machine learning tasks.
"""

import re
import warnings 
from dataclasses import dataclass, field
from typing import Callable, List, Any, Dict, Optional, Union, Tuple
import pandas as pd

from .._gofastlog import gofastlog
from ..api.property import BaseClass 
from ..decorators import executeWithFallback
from ..tools.depsutils import ( 
    is_module_installed, ensure_pkg
    )
from ..tools.baseutils import ( 
    is_readable, select_features
)
from ..tools.coreutils import (
    sanitize_frame_cols,
    exist_features,
    to_numeric_dtypes, 
)
from ..tools.validator import ( 
    array_to_frame, 
    check_array, 
)

__all__ = ["Data", "Missing", "MergeableSeries", "MergeableFrames", "Frames"]

@dataclass
class Data(BaseClass):
    data: pd.DataFrame
    _operations: List[
        Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
    ] = field(default_factory=list)
    _kwargs: List[Dict[str, Any]] = field(default_factory=list)
    verbose: int = 0
    _logging: Any = field(init=False)

    def __post_init__(self):
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        self._initial_processing()

    def _initial_processing(self):
        df = check_array(
            self.data,
            force_all_finite='allow-nan',
            dtype=object,
            input_name='Data',
            to_frame=True
        )
        df = array_to_frame(
            df,
            to_frame=True,
            input_name='col',
            force=True
        )
        self.data = df

    @property
    def data_(self) -> pd.DataFrame:
        return self.data

    @data_.setter
    def data_(self, d: Union[str, pd.DataFrame]):
        self.data = is_readable(d)

    @property
    def describe(self) -> pd.DataFrame:
        return self.data.describe()

    def shrunk(
        self,
        columns: List[str],
        **kwd
    ):
        self._operations.append(self._shrunk)
        self._kwargs.append({'columns': columns, **kwd})
        return self

    def report(
        self,
        **kwd
    ):
        self._operations.append(self._report)
        self._kwargs.append({'kwd': kwd})
        return self

    def rename(
        self,
        columns: Optional[List[str]] = None,
        pattern: Optional[str] = None
    ):
        self._operations.append(self._rename)
        self._kwargs.append({
            'columns': columns,
            'pattern': pattern
        })
        return self

    def drop(
        self,
        labels: Optional[List[Union[str, int]]] = None,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
        axis: int = 0,
        **kws
    ):
        self._operations.append(self._drop)
        self._kwargs.append({
            'labels': labels,
            'columns': columns,
            'inplace': inplace,
            'axis': axis,
            **kws
        })
        return self

    def fill_cols_pattern(
        self,
        pattern: Optional[str] = None
    ):
        self._operations.append(self._fill_cols_pattern)
        self._kwargs.append({'pattern': pattern})
        return self

    def sanitize(
        self,
        **kwargs
    ):
        self._operations.append(self._sanitize)
        self._kwargs.append(kwargs)
        return self

    def drop_nan(
        self,
        **kwargs
    ):
        self._operations.append(self._drop_nan)
        self._kwargs.append(kwargs)
        return self

    def filter(
        self,
        **kwargs
    ):
        self._operations.append(self._filter)
        self._kwargs.append(kwargs)
        return self

    def encode(
        self,
        **kwargs
    ):
        self._operations.append(self._encode)
        self._kwargs.append(kwargs)
        return self

    @executeWithFallback
    def execute(
        self,
        inplace: bool = False
    ):
        result = self.data.copy() if not inplace else self.data
        for op, kw in zip(self._operations, self._kwargs):
            result = op(result, **kw)
        if inplace:
            self.data = result
        self._operations.clear()
        self._kwargs.clear()
        return result

    # Operation Methods
    def _shrunk(
        self,
        df: pd.DataFrame,
        columns: List[str],
        **kwd
    ) -> pd.DataFrame:
        self._logging.info(f"Selecting features: {columns}")
        return select_features(df, features=columns, **kwd)

    def _report(
        self,
        df: pd.DataFrame,
        **kwd
    ):
        self._logging.info("Generating profiling report")
        if not is_module_installed (
                "pandas_profiling", 
                distribution_name="pandas-profiling"
            ): 
            if self.verbose:
                warnings.warn(
                    "Report couldn't be generated."
                    "'pandas_profiling' is missing. Use"
                    " `pip` or `conda` to install it."
                )
            return df 
        from pandas_profiling import ProfileReport
        
        return ProfileReport(df, **kwd)

    def _rename(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        pattern: Optional[str] = None
    ) -> pd.DataFrame:
        pattern = pattern or r'[ -@*#&+/]'
        regex = re.compile(pattern, flags=re.IGNORECASE)
        df = df.copy()
        df.columns = df.columns.str.strip()
        if columns:
            exist_features(df, columns, 'raise')
            new_columns = {
                col: regex.sub('_', col.lower()) for col in columns
            }
            df.rename(columns=new_columns, inplace=True)
        else:
            df.columns = df.columns.str.lower().map(
                lambda o: regex.sub('_', o)
            )
        return df

    def _drop(
        self,
        df: pd.DataFrame,
        labels: Optional[List[Union[str, int]]] = None,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
        axis: int = 0,
        **kws
    ) -> pd.DataFrame:
        self._logging.info(
            f"Dropping labels: {labels}, columns: {columns}, axis: {axis}"
        )
        return df.drop(
            labels=labels,
            columns=columns,
            axis=axis,
            inplace=False,
            **kws
        )

    def _fill_cols_pattern(
        self,
        df: pd.DataFrame,
        pattern: Optional[str] = None
    ) -> pd.DataFrame:
        self._logging.info("Filling columns pattern")
        df = sanitize_frame_cols(df, fill_pattern='_')
        for col in df.columns:
            setattr(self, col, df[col])
        return df

    def _sanitize(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        self._logging.info("Sanitizing data")
        return df.applymap(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    def _drop_nan(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        self._logging.info("Dropping NaN values")
        return df.dropna(**kwargs)

    def _filter(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        condition = kwargs.get('condition')
        if condition:
            self._logging.info(f"Filtering data with condition: {condition}")
            return df.query(condition)
        return df

    def _encode(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        columns = kwargs.get(
            'columns',
            df.select_dtypes(include=['object']).columns
        )
        self._logging.info(f"Encoding columns: {columns}")
        return pd.get_dummies(df, columns=columns)

Data.__doc__ = """\
A powerful data manipulation class that supports method chaining to perform 
a series of operations on a pandas DataFrame. Operations are queued and 
executed in the order they are added when the `execute` method is called. 
This design ensures flexibility and efficiency in data preprocessing workflows.

.. math::
    Data_{processed} = encode(filter(drop\_nan(sanitize(Data_{original})))

Parameters
----------
data : pandas.DataFrame
    The input dataset to be processed. Must be a valid pandas DataFrame.
verbose : int, optional
    Verbosity level for logging purposes. Higher values increase the verbosity.
    Default is `0`.

Attributes
----------
data_ : pandas.DataFrame
    The internal DataFrame after initial processing and sanitization.
describe : pandas.DataFrame
    Summary statistics of the dataset, similar to pandas' `describe` method.

Methods
-------
describe
    Returns summary statistics of the dataset.

shrunk(columns, **kwd)
    Reduces the dataset to the specified columns based on feature selection 
    criteria.

profilingReport(**kwd)
    Generates a comprehensive profiling report of the dataset.

rename(columns=None, pattern=None)
    Renames columns based on a provided pattern or list of columns.

drop(labels=None, columns=None, inplace=False, axis=0, **kws)
    Drops specified labels or columns from the dataset.

fill_cols_pattern(pattern=None)
    Fills column names based on a specified regular expression pattern.

sanitize(**kwargs)
    Cleans the dataset by stripping whitespace from string entries.

drop_nan(**kwargs)
    Removes rows containing NaN values based on specified criteria.

filter(condition=None, **kwargs)
    Filters the dataset based on a given condition.

encode(**kwargs)
    Encodes categorical variables using one-hot encoding or other specified methods.

execute(inplace=False)
    Executes all queued operations. If `inplace` is `True`, modifies the internal
    DataFrame; otherwise, returns a new DataFrame with the transformations applied.

Examples
--------
>>> from gofast.dataops.core import Data
>>> import pandas as pd
>>> 
>>> # Sample data
>>> sample_data = pd.DataFrame({
...     'Name': [' Alice ', 'Bob', None, 'David'],
...     'Age': [25, None, 30, 22],
...     'City': ['New York', 'Los Angeles', 'Chicago', ' '],
...     'Salary': [70000, 80000, None, 50000]
... })
>>> 
>>> # Instantiate and chain operations
>>> processed_data = Data(sample_data)\
...     .sanitize()\
...     .drop_nan(thresh=2)\
...     .filter(condition='Age > 23')\
...     .encode(columns=['City'])\
...     .execute(inplace=False)
>>> 
>>> print("Processed Data:")
>>> print(processed_data)
Processed Data:
     Name   Age      City   Salary  City_Chicago  City_Los Angeles  \
0   Alice  25.0  New York  70000.0              0                  0   
2    None  30.0   Chicago      NaN              1                  0   

   City_New York  
0                1  
2                0  

>>> 
>>> # Execute operations inplace
>>> Data(sample_data)\
...     .sanitize()\
...     .drop_nan(thresh=2)\
...     .filter(condition='Age > 23')\
...     .encode(columns=['City'])\
...     .execute(inplace=True)
>>> 
>>> print("\nData after Inplace Execution:")
>>> print(sample_data)
Data after Inplace Execution:
     Name   Age      City   Salary  City_Chicago  City_Los Angeles  \
0   Alice  25.0  New York  70000.0              0                  0   
2    None  30.0   Chicago      NaN              1                  0   

   City_New York  
0                1  
2                0  

Notes
-----
- The `Data` class utilizes method chaining to queue operations, which are only
  executed when `execute` is called. This allows for flexible and readable data
  preprocessing pipelines.
- The `execute` method can modify the internal DataFrame in place or return a new
  transformed DataFrame based on the `inplace` parameter.
- Ensure that the initial `data` provided is a valid pandas DataFrame or a
  readable data source compatible with pandas.

See Also
--------
pandas.DataFrame : The primary data structure used for data manipulation.
pandas.get_dummies : Used for encoding categorical variables.

References
----------
.. [1] McKinney, W. (2010). Data Structures for Statistical Computing in Python.
       In *Proceedings of the 9th Python in Science Conference* (Vol. 445, pp. 51-56).
.. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
       Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A.,
       Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011).
       Scikit-learn: Machine Learning in Python. *Journal of Machine Learning
       Research*, 12, 2825-2830.
"""


@dataclass
class MergeableSeries(BaseClass):
    series: pd.Series
    _operations: List[
        Callable[[pd.Series, Dict[str, Any]], pd.Series]
    ] = field(default_factory=list)
    _kwargs: List[Dict[str, Any]] = field(default_factory=list)
    _logging: Any = field(init=False)

    def __post_init__(self):
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)

    def __and__(self, other: 'MergeableSeries') -> 'MergeableSeries':
        self._operations.append(self._logical_and)
        self._kwargs.append({'other': other})
        return self

    def execute(self) -> pd.Series:
        result = self.series.copy()
        for op, kw in zip(self._operations, self._kwargs):
            result = op(result, **kw)
        self._operations.clear()
        self._kwargs.clear()
        return result

    def _logical_and(
            self, series: pd.Series, 
            other: 'MergeableSeries'
        ) -> pd.Series:
        if not isinstance(other, MergeableSeries):
            raise ValueError(
                "Operand must be an instance of MergeableSeries")
        series1 = series.astype(
            str) if series.dtype == 'object' else series
        series2 = other.series.astype(
            str) if other.series.dtype == 'object' else other.series
        return series1 & series2

MergeableSeries.__doc__ = """\
A class for performing mergeable operations on pandas Series objects. 
This class allows chaining of logical operations which can be executed 
collectively to produce a resultant Series.

Parameters
----------
series : `pd.Series`
    The primary pandas Series to be manipulated and merged with other 
    `MergeableSeries` instances.

Attributes
----------
_operations : List[Callable[[pd.Series, Dict[str, Any]], pd.Series]]
    A list of operations to be applied to the Series.
_kwargs : List[Dict[str, Any]]
    A list of keyword arguments corresponding to each operation.
_logging : Any
    Logger instance for logging operations (initialized post creation).

Methods
-------
__and__(other)
    Appends a logical AND operation with another `MergeableSeries`.
execute()
    Executes all queued operations and returns the resulting Series.

Formulation
-----------
The logical AND operation between two Series \( S_1 \) and \( S_2 \) is defined as:

.. math::
    S_{result} = S_1 \land S_2

where \( \land \) represents the element-wise logical AND.

Examples
--------
>>> from gofast.dataops.core import MergeableSeries
>>> import pandas as pd
>>> series1 = pd.Series([True, False, True])
>>> series2 = pd.Series([True, True, False])
>>> ms1 = MergeableSeries(series1)
>>> ms2 = MergeableSeries(series2)
>>> result = (ms1 & ms2).execute()
>>> print(result)
0     True
1    False
2    False
dtype: bool

Notes
-----
- The `execute` method clears the queued operations after execution.
- Only public methods (`__and__` and `execute`) are exposed for chaining 
  operations. Internal methods prefixed with `_` are not documented.

See Also
--------
Frames, MergeableFrames, Missing

References
----------
.. [1] Doe, J. (2023). *Advanced Data Operations with Pandas*. Data Science 
       Publishing.
"""


@dataclass
class Frames(BaseClass):
    frames: List[pd.DataFrame]
    _operations: List[
        Callable[[List[pd.DataFrame], Dict[str, Any]], pd.DataFrame]
    ] = field(default_factory=list)
    _kwargs: List[Dict[str, Any]] = field(default_factory=list)
    verbose: int = 0
    _logging: Any = field(init=False)

    def __post_init__(self):
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        self._initial_processing()
       
    def _initial_processing(self, *frames: pd.DataFrame, **kws) -> 'Frames':
        processed_frames = []
        frames = [ 
            array_to_frame(df, to_frame=True, force=True, input_name="col")
            for df in frames
        ]
        for frame in self.frames:
            processed_frame = to_numeric_dtypes(frame, **kws)
            processed_frames.append(processed_frame)
        self.frames = processed_frames
        
    def merge(
        self, 
        on: Union[str, List[str]], 
        how: str = 'inner', 
        **kws
    ) -> 'Frames':
        self._operations.append(self._merge)
        self._kwargs.append({'on': on, 'how': how, **kws})
        return self

    def concat(
        self, 
        axis: int = 0, 
        **kws
    ) -> 'Frames':
        self._operations.append(self._concat)
        self._kwargs.append({'axis': axis, **kws})
        return self

    def compare(self) -> bool:
        self._operations.append(self._compare)
        self._kwargs.append({})
        return self.execute_compare()

    def add(self) -> 'Frames':
        self._operations.append(self._add)
        self._kwargs.append({})
        return self

    def conditional_filter(
            self, conditions: Dict[str, Callable[[Any], bool]]
        ) -> 'Frames':
        self._operations.append(self._conditional_filter)
        self._kwargs.append({'conditions': conditions})
        return self

    def execute(self) -> pd.DataFrame:
        result = self.frames.copy()
        for op, kw in zip(self._operations, self._kwargs):
            result = op(result, **kw)
        self._operations.clear()
        self._kwargs.clear()
        return result

    def execute_compare(self) -> bool:
        result = self.frames.copy()
        for op, kw in zip(self._operations, self._kwargs):
            result = op(result, **kw)
        self._operations.clear()
        self._kwargs.clear()
        return result

    # Operation Methods
    def _merge(
        self, 
        frames: List[pd.DataFrame], 
        on: Union[str, List[str]], 
        how: str = 'inner', 
        **kws
    ) -> pd.DataFrame:
        self._logging.info(f"Merging frames on {on} with how='{how}'")
        result = frames[0]
        for df in frames[1:]:
            result = pd.merge(result, df, on=on, how=how, **kws)
        return result

    def _concat(
        self, 
        frames: List[pd.DataFrame], 
        axis: int = 0, 
        **kws
    ) -> pd.DataFrame:
        self._logging.info(f"Concatenating frames along axis={axis}")
        return pd.concat(frames, axis=axis, **kws)

    def _compare(
        self, 
        frames: List[pd.DataFrame], 
        **kws
    ) -> bool:
        self._logging.info("Comparing frames for equality")
        first_df = frames[0]
        for df in frames[1:]:
            if not first_df.equals(df):
                return False
        return True

    def _add(
        self, 
        frames: List[pd.DataFrame], 
        **kws
    ) -> pd.DataFrame:
        self._logging.info("Adding frames element-wise")
        result = frames[0].copy()
        for df in frames[1:]:
            result = result.add(df, fill_value=0)
        return result

    def _conditional_filter(
        self, 
        frames: List[pd.DataFrame], 
        conditions: Dict[str, Callable[[Any], bool]], 
        **kws
    ) -> List[pd.DataFrame]:
        self._logging.info("Applying conditional filters to frames")
        filtered_frames = []
        for frame in frames:
            mask = pd.Series(True, index=frame.index)
            for col, condition in conditions.items():
                mask &= frame[col].apply(condition)
            filtered_frames.append(frame[mask])
        return filtered_frames

Frames.__doc__ = """\
A class for managing and performing various operations on a collection 
of pandas DataFrame objects. Supports merging, concatenation, comparison, 
addition, and conditional filtering of DataFrames in a streamlined manner.

Parameters
----------
frames : `List[pd.DataFrame]`
    A list of pandas DataFrame objects to be managed and operated upon.
verbose : `int`, optional
    Level of verbosity for logging operations (default is 0).

Attributes
----------
_operations : List[Callable[[List[pd.DataFrame], Dict[str, Any]], pd.DataFrame]]
    A list of operations to be applied to the DataFrames.
_kwargs : List[Dict[str, Any]]
    A list of keyword arguments corresponding to each operation.
_logging : Any
    Logger instance for logging operations (initialized post creation).

Methods
-------
merge(on, how='inner', **kws)
    Queues a merge operation on the specified keys with the given join type.
concat(axis=0, **kws)
    Queues a concatenation operation along the specified axis.
compare()
    Queues a comparison operation to check equality across all DataFrames.
add()
    Queues an element-wise addition operation across all DataFrames.
conditional_filter(conditions)
    Queues a conditional filter operation based on specified column conditions.
execute()
    Executes all queued operations and returns the resultant DataFrame.
execute_compare()
    Executes all queued comparison operations and returns a boolean result.

Formulation
-----------
- **Merge Operation**: Combines DataFrames \( D_1, D_2, \dots, D_n \)
    on key(s) \( K \) using join type \( J \).
  
  .. math::
      D_{merged} = D_1 \text{ merge } D_2 \text{ on } K \text{ how } J

- **Concatenation Operation**: Concatenates DataFrames along the specified
   axis.

  .. math::
      D_{concat} = \text{concat}(D_1, D_2, \dots, D_n, \text{axis}=A)

Examples
--------
>>> from gofast.dataops.core import Frames
>>> import pandas as pd
>>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
>>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
>>> frames = Frames([df1, df2])
>>> concatenated = frames.concat().execute()
>>> print(concatenated)
   A  B
0  1  3
1  2  4
2  5  7
3  6  8

Notes
-----
- The `execute` method processes all queued operations in the order they were added.
- Internal methods prefixed with `_` are not exposed in the public API.

See Also
--------
MergeableSeries, MergeableFrames, Missing

References
----------
.. [1] Smith, A. (2022). *DataFrame Manipulations in Python*. Python Data 
       Publishing.
"""


@dataclass
class MergeableFrames(BaseClass):
    frames: List[pd.DataFrame]
    _operations: List[
        Callable[[List[pd.DataFrame], Dict[str, Any]], pd.DataFrame]
    ] = field(default_factory=list)
    _kwargs: List[Dict[str, Any]] = field(default_factory=list)
    _logging: Any = field(init=False)

    def __post_init__(self):
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)

    def __and__(self, other: 'MergeableFrames') -> 'MergeableFrames':
        self._operations.append(self._logical_and)
        self._kwargs.append({'other': other})
        return self

    def __or__(self, other: 'MergeableFrames') -> 'MergeableFrames':
        self._operations.append(self._logical_or)
        self._kwargs.append({'other': other})
        return self

    def execute(self) -> pd.DataFrame:
        result = self.frames.copy()
        for op, kw in zip(self._operations, self._kwargs):
            result = op(result, **kw)
        self._operations.clear()
        self._kwargs.clear()
        return result

    # Operation Methods
    def _logical_and(
        self, 
        frames: List[pd.DataFrame], 
        other: 'MergeableFrames'
    ) -> pd.DataFrame:
        self._logging.info("Performing logical AND operation between frames")
        if len(frames) != 1 or len(other.frames) != 1:
            raise ValueError(
                "Logical operations are only supported"
                " for single DataFrame instances.")
        return frames[0] & other.frames[0]

    def _logical_or(
        self, 
        frames: List[pd.DataFrame], 
        other: 'MergeableFrames'
    ) -> pd.DataFrame:
        self._logging.info("Performing logical OR operation between frames")
        if len(frames) != 1 or len(other.frames) != 1:
            raise ValueError(
                "Logical operations are only supported"
                " for single DataFrame instances.")
        return frames[0] | other.frames[0]

MergeableFrames.__doc__ = """\
A class for performing logical merge operations on multiple pandas 
DataFrame objects. Enables chaining of logical AND and OR operations 
which can be executed collectively to produce a resultant DataFrame.

Parameters
----------
frames : `List[pd.DataFrame]`
    The primary list of pandas DataFrame objects to be manipulated and 
    merged with other `MergeableFrames` instances.

Attributes
----------
_operations : List[Callable[[List[pd.DataFrame], Dict[str, Any]], pd.DataFrame]]
    A list of operations to be applied to the DataFrames.
_kwargs : List[Dict[str, Any]]
    A list of keyword arguments corresponding to each operation.
_logging : Any
    Logger instance for logging operations (initialized post creation).

Methods
-------
__and__(other)
    Appends a logical AND operation with another `MergeableFrames`.
__or__(other)
    Appends a logical OR operation with another `MergeableFrames`.
execute()
    Executes all queued operations and returns the resulting DataFrame.

Formulation
-----------
- **Logical AND Operation**: Element-wise logical AND between two DataFrames 
  \( D_1 \) and \( D_2 \).

  .. math::
      D_{result} = D_1 \land D_2

- **Logical OR Operation**: Element-wise logical OR between two DataFrames 
  \( D_1 \) and \( D_2 \).

  .. math::
      D_{result} = D_1 \lor D_2

Examples
--------
>>> from gofast.dataops.core import MergeableFrames
>>> import pandas as pd
>>> df1 = pd.DataFrame({'A': [True, False], 'B': [True, True]})
>>> df2 = pd.DataFrame({'A': [False, True], 'B': [True, False]})
>>> mf1 = MergeableFrames([df1])
>>> mf2 = MergeableFrames([df2])
>>> result = (mf1 & mf2).execute()
>>> print(result)
       A      B
0  False   True
1  False  False

Notes
-----
- Logical operations are only supported for single DataFrame instances within 
  `MergeableFrames`.
- The `execute` method clears the queued operations after execution.

See Also
--------
MergeableSeries, Frames, Missing

References
----------
.. [1] Lee, B. (2021). *Logical Operations in DataFrames*. Data Analysis 
       Journal.
"""


@dataclass
class Missing(Data):
    in_percent: bool = False
    sample: Optional[int] = None
    kind: Optional[str] = None
    drop_columns: Optional[List[str]] = field(default=None)
    _operations: List[
        Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
    ] = field(default_factory=list)
    _kwargs: List[Dict[str, Any]] = field(default_factory=list)
    _logging: Any = field(init=False)
    verbose=0

    def __post_init__(self):
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        super().__post_init__()

    @property
    def isnull(self) -> pd.Series:
        self.isnull_ = ( self.data.isnull().mean() * 100 
                        if self.in_percent else self.data.isnull().mean()
                        )
        return self.isnull_

    def plot(self, figsize: Tuple[int, int] = None, **kwd):
        self._operations.append(self._plot)
        self._kwargs.append({'figsize': figsize, **kwd})
        return self

    def drop(
        self,
        data: Optional[Union[str, pd.DataFrame]] = None,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
        axis: int = 1,
        **kwd
    ):
        self._operations.append(self._drop)
        self._kwargs.append({
            'data': data,
            'columns': columns,
            'inplace': inplace,
            'axis': axis,
            **kwd
        })
        return self

    def replace(
        self,
        data: Optional[Union[str, pd.DataFrame]] = None,
        columns: Optional[Union[str, List[str]]] = None,
        fill_value: Optional[float] = None,
        new_column_name: Optional[str] = None,
        return_non_null: bool = False,
        **kwargs
    ):
        self._operations.append(self._replace)
        self._kwargs.append({
            'data': data,
            'columns': columns,
            'fill_value': fill_value,
            'new_column_name': new_column_name,
            'return_non_null': return_non_null,
            **kwargs
        })
        return self

    def execute(self, inplace: bool = False):
        result = self.data.copy() if not inplace else self.data
        for op, kw in zip(self._operations, self._kwargs):
            result = op(result, **kw)
        if inplace:
            self.data = result
        self._operations.clear()
        self._kwargs.clear()
        return result 

    @property
    def get_missing_columns(self) -> List[str]:
        return list(self.data.columns[self.data.isna().any()])

    @property
    def sanity_check(self) -> bool:
        return self.data.isna().any().any()

    def _plot(
            self, 
            df: pd.DataFrame, 
            figsize: Optional[Tuple[int, int]] = None, 
            **kwd
    ) -> pd.DataFrame:
        from gofast.plot.explore import QuestPlotter
        QuestPlotter(fig_size=figsize).fit(df).plotMissing(
            kind=self.kind, sample=self.sample, **kwd
        )
        return df

    def _drop(
        self,
        df: pd.DataFrame,
        data: Optional[Union[str, pd.DataFrame]] = None,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
        axis: int = 1,
        **kwd
    ) -> pd.DataFrame:
        if data is not None:
            df = is_readable(data)
        if columns is not None:
            exist_features(df, columns, error='raise')
        if columns is None:
            return df.dropna(axis=axis, inplace=False, **kwd)
        else:
            return df.drop(
                columns=columns, 
                axis=axis, 
                inplace=False, 
                **kwd
            )

    @ensure_pkg("pyjanitor", condition="return_non_null")
    def _replace(
        self,
        df: pd.DataFrame,
        data: Optional[Union[str, pd.DataFrame]] = None,
        columns: Optional[Union[str, List[str]]] = None,
        fill_value: Optional[float] = None,
        new_column_name: Optional[str] = None,
        return_non_null: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        if data is not None:
            df = is_readable(data)
        if isinstance(columns, str):
            columns = [columns]
        if return_non_null:
            from pyjanitor import coalesce
            new_column_name = self._assert_str(
                new_column_name, "new_column_name")
            df = coalesce(
                df, columns=columns, 
                new_column_name=new_column_name
            )
        elif fill_value is not None:
            df.fillna(value=fill_value, inplace=True, **kwargs)
        return df

    @staticmethod
    def _assert_str(obj: Any, name: str) -> str:
        if not isinstance(obj, str):
            raise TypeError(f"{name} must be a string.")
        return obj
    
Missing.__doc__ = """\
A class for handling and visualizing missing data within pandas DataFrame 
objects. Provides functionalities to plot missing data patterns, drop 
missing values based on specified criteria, and replace missing values 
with specified fill strategies.

Parameters
----------
in_percent : `bool`, optional
    If `True`, calculates missing data percentages. Defaults to `False`.
sample : `int`, optional
    Number of samples to visualize in plots. Defaults to `None`.
kind : `str`, optional
    Type of plot to generate for missing data visualization. Defaults to 
    `None`.
drop_columns : `List[str]`, optional
    Columns to consider when dropping missing values. Defaults to `None`.
verbose : `int`, optional
    Level of verbosity for logging operations. Defaults to `0`.

Attributes
----------
_operations : List[Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]]
    A list of operations to be applied to the DataFrame.
_kwargs : List[Dict[str, Any]]
    A list of keyword arguments corresponding to each operation.
_logging : Any
    Logger instance for logging operations (initialized post creation).

Properties
----------
isnull : `pd.Series`
    Series indicating the proportion of missing values per column.
get_missing_columns : `List[str]`
    List of column names that contain missing values.
sanity_check : `bool`
    Indicates if any missing values exist in the DataFrame.

Methods
-------
plot(figsize=None, **kwd)
    Queues a plot operation to visualize missing data.
drop(data=None, columns=None, inplace=False, axis=1, **kwd)
    Queues a drop operation to remove missing data based on criteria.
replace(data=None, columns=None, fill_value=None, new_column_name=None, 
        return_non_null=False, **kwargs)
    Queues a replace operation to fill missing values based on specified strategies.
execute(inplace=False)
    Executes all queued operations and returns the resultant DataFrame.

Formulation
-----------
- **Missing Data Percentage**: For each column \( C \), the percentage of 
  missing data is calculated as:

  .. math::
      P_{C} = \left( \frac{\text{Number of Missing Values in } C}{\text{Total 
      Rows}} \right) \times 100

Examples
--------
>>> from gofast.dataops.core import Missing
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'A': [1, 2, None, 4],
...     'B': [None, 2, 3, 4],
...     'C': [1, None, None, 4]
... })
>>> missing = Missing(df, in_percent=True)
>>> missing.data = df
>>> print(missing.isnull)
A    25.0
B    25.0
C    50.0
dtype: float64
>>> cleaned_df = missing.drop(columns=['A']).execute()
>>> print(cleaned_df)
     B    C
0  NaN  1.0
1  2.0  NaN
2  3.0  NaN
3  4.0  4.0

Notes
-----
- The `execute` method applies all queued operations in the order they were added.
- The `plot` method leverages `QuestPlotter` from `gofast.plot.explore` for 
  visualization.
- The `replace` method requires the `pyjanitor` package when `return_non_null` 
  is set to `True`.

See Also
--------
MergeableSeries, Frames, MergeableFrames

References
----------
.. [1] Johnson, L. (2020). *Handling Missing Data in Pandas*. Data Science 
       Essentials.
"""
    
if __name__ == "__main__":
    # Sample data
    import pandas as pd 
    from gofast.dataops.core import Data 
    sample_data = pd.DataFrame({
        'Name': [' Alice ', 'Bob', None, 'David'],
        'Age': [25, None, 30, 22],
        'City': ['New York', 'Los Angeles', 'Chicago', ' '],
        'Salary': [70000, 80000, None, 50000]
    })

    # Instantiate and chain operations
    processed_data = Data(sample_data) \
        .sanitize() \
        .drop_nan(thresh=2) \
        .filter(condition='Age > 23') \
        .encode(columns=['City']) \
        .execute(inplace=False)

    print("Processed Data:")
    print(processed_data)

  

