# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 

import gofast.stats.descriptive # noqa 
from gofast.api.formatter import MultiFrameFormatter 
from gofast.api.util import escape_dataframe_elements 
from gofast.decorators import DynamicMethod 

__all__= ["summary" ]

@DynamicMethod (expected_type="both", prefixer ="exclude" )
def summary(
    df, 
    include_correlation=False, 
    numeric_only=True, 
   ):
    """
    Generate a customizable summary for a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame for which the summary is generated.
    include_correlation : bool, optional
        If True, include the correlation matrix for numerical features in the
        summary. Default is False.
    numeric_only : bool, optional
        If True, make a summary only for  numeric values. If False, include 
        all data types. 
    Returns
    -------
    summary : MultiFrameFormatter istance that operate like bunch object 
        tht expect print for visualization. 
        
    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast._public import summary 
    >>> data = {
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 6, None, 8, 9],
    ...     'C': ['foo', 'bar', 'baz', 'qux', 'quux']
    ... }
    >>> df = pd.DataFrame(data)
    >>> summary(df)
                 Infos             
    ===============================
                         A        B
    -------------------------------
    Data Types       int64  float64
    Missing Values       0        1
    Byte Size           40       40
    Unique Counts        5        4
    ===============================
            Basic Statistics       
    -------------------------------
    count           5.0000   4.0000
    mean            3.0000   7.0000
    std             1.5811   1.8257
    min             1.0000   5.0000
    25%             2.0000   5.7500
    50%             3.0000   7.0000
    75%             4.0000   8.2500
    max             5.0000   9.0000
    ===============================
    """
    def _encode_categorical_data(df):
        original_columns = df.columns.tolist()
        from sklearn.preprocessing import LabelEncoder
        
        encoded_df = df.copy()
        for column in df.columns:
            if df[column].dtype == 'object':
                encoder = LabelEncoder()
                encoded_df[column] = encoder.fit_transform(df[column].astype(str))
        return encoded_df[original_columns]

    dfs = [] 
    titles = ["Infos"]
    if numeric_only:
        df = df.select_dtypes([np.number])
    
    infos = {#π' is used for frame formatter
        "Data Types": df.dtypes.to_dict(), 
        "Missing Values": df.isnull().sum().to_dict(),
        "Byte Size": {col: df[col].nbytes if df[col].dtype != 'O' 
                      else "<N/A>" for col in df.columns}, 
        "Unique Counts": {col: df[col].nunique() for col in df.columns}
    }
    df_infos = pd.DataFrame(infos.values(), index=infos.keys())
    dfs.append(df_infos)

    titles.append("Basic Statistics")
    df_descr = df.describe(None, 'all' if not numeric_only else None,
                           datetime_is_numeric=True )
    dfs.append(df_descr)

    if include_correlation: 
        if numeric_only is False: 
            df = _encode_categorical_data(df)
        df_corr = df.corr()
        #df_corr.index = [ index.replace (" ", "π") for index in df_corr.index]
        titles.extend(["Correlation Matrix"])
        dfs.append(df_corr)
        
    dfs = [escape_dataframe_elements(df) for df in dfs]
    summary = MultiFrameFormatter(
        titles=titles, max_rows =11, max_cols ="auto")
    summary.add_dfs(*dfs)
    print(summary)