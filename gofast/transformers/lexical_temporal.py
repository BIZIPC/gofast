# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import division, annotations
import warnings
import numpy as np   
import pandas as pd 

from sklearn.base import BaseEstimator,TransformerMixin 
from sklearn.feature_extraction.text import TfidfVectorizer

from ..tools.validator import build_data_if 

__all__=[ "TextToVectorTransformer", "TextFeatureExtractor",
         "DateFeatureExtractor" ]

class TextToVectorTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that converts text columns in a DataFrame or a numpy ndarray 
    to TF-IDF vectorized features.

    Parameters
    ----------
    columns : str, list of str, or 'auto', default='auto'
        Specifies the columns to be vectorized. If set to 'auto', the transformer
        will automatically detect and vectorize all columns with object dtype.
        If a single string is provided, it will be wrapped into a list.
    append_transformed : bool, default=True
        Determines whether to append the transformed columns to the original 
        DataFrame. If set to False, only the transformed columns are returned.
    keep_original : bool, default=False
        Whether to retain the original text columns in the output. Effective 
        only if `append_transformed` is True.

    Attributes
    ----------
    vectorizers_ : dict of {str: TfidfVectorizer}
        A dictionary mapping column names to their fitted TfidfVectorizer 
        instances.

    Examples
    --------
    >>> from gofast.transformers.lexical_temporal import TextToVectorTransformer
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'text1': ['hello world', 'example text'],
    ...     'text2': ['another column', 'with text']
    ... })
    >>> transformer = TextToVectorTransformer()
    >>> transformed_data = transformer.fit_transform(df)
        text1_tfidf_0  text1_tfidf_1  ...  text2_tfidf_2  text2_tfidf_3
     0       0.000000       0.707107  ...       0.000000       0.000000
     1       0.707107       0.000000  ...       0.707107       0.707107
    >>> print(transformed_data.shape)  # Output depends on the input data
    (2, 8)
    
    Notes
    -----
    If `columns` is set to 'auto' and no object dtype columns are found, a 
    warning is issued and no transformation is applied. If `columns` contains 
    names not present in the DataFrame, a ValueError is raised. When input is 
    an ndarray, it is assumed that all data are text, and they are vectorized 
    accordingly. Care should be taken as passing non-text data can lead to 
    unexpected results.
    """
    def __init__(self, columns='auto', append_transformed=True, 
                 keep_original=False):
        self.columns = columns
        self.append_transformed = append_transformed
        self.keep_original = keep_original
        self.vectorizers_ = {}
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the text data in the specified columns or ndarray.
    
        This method identifies and fits a TF-IDF vectorizer to each text column
        specified, or to the entire ndarray assuming all data are text. It prepares
        the transformer to convert these text entries into TF-IDF numerical vectors
        on transformation.
    
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Input data to fit the transformer. Can be a pandas DataFrame or a 
            numpy ndarray. If a DataFrame is used, text columns are identified 
            based on the `columns` parameter. If an ndarray is used, all data are
            assumed to be text.
        y : None
            Unused parameter. Included for compatibility with sklearn's fit method
            signature.
    
        Raises
        ------
        ValueError
            If specified columns in `columns` are not found in the DataFrame.
    
        Returns
        -------
        self : object
            Returns the instance itself, fitted to the text data.
    
        Notes
        -----
        The method uses a TfidfVectorizer for each column and fits it to the 
        text data present in that column. If `columns` is set to 'auto', it 
        automatically detects text columns based on the object dtype. It will
        raise a warning if no text columns are detected when 'auto' is used.
        """
 
        X = build_data_if(
            X , input_name="col", force=True, raise_exception=True ) 
 
        if isinstance(X, pd.DataFrame):
            if self.columns == 'auto':
                # Automatically determine text columns
                self.columns = [col for col in X.columns if X[col].dtype == object]
            if not self.columns:
                warnings.warn(
                    "No text columns found. No transformation will be applied.")
                return self
            if isinstance(self.columns, str):
                self.columns = [self.columns]
            # Validate columns
            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
            for col in self.columns:
                self.vectorizers_[col] = TfidfVectorizer()
                self.vectorizers_[col].fit(X[col])
        elif isinstance(X, np.ndarray):
            # Assume all columns are text
            self.vectorizers_['array'] = TfidfVectorizer()
            self.vectorizers_['array'].fit(X.flatten())
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input data into a vectorized format using the fitted 
        TF-IDF vectorizers.
    
        This method converts text data into TF-IDF vectors using the fitted 
        vectorizers.
        If the input is a DataFrame and `append_transformed` is True, it 
        appends the transformed data as new columns to the original DataFrame
        unless `keep_original` is False, in which case it replaces the original
        text columns.
    
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The data to transform. Can be a pandas DataFrame or a numpy ndarray.
            The type and structure should match the data type used during fitting.
    
        Returns
        -------
        transformed_data : pandas.DataFrame or numpy.ndarray
            The transformed data. If `X` is a DataFrame, the transformed data
            will either replace the original text columns or be appended as 
            new columns depending on
            the `append_transformed` and `keep_original` parameters. If `X` 
            is an ndarray, the result is a 2D array of the TF-IDF features.
    
        Notes
        -----
        If both `append_transformed` and `keep_original` are False, only the
        new transformed columns are returned, without any of the original 
        columns from the DataFrame.
        """
        X = build_data_if(
            X , input_name="col", force=True, raise_exception=True ) 
        if isinstance(X, pd.DataFrame):
            if self.keep_original: 
                self.append_transformed =True 
                
            if not self.append_transformed:
                transformed_data = pd.DataFrame(index=X.index)
            else:
                transformed_data = X.copy() 
            for col in self.columns:
                transformed = self.vectorizers_[col].transform(X[col])
                for i in range(transformed.shape[1]):
                    new_col_name = f"{col}_tfidf_{i}"
                    transformed_data[new_col_name] = ( 
                        transformed[:, i].toarray().ravel()
                        )
            if not self.keep_original: 
                try: 
                    transformed_data = transformed_data.drop(columns = self.columns )
                except: pass 
            
            return transformed_data
        elif isinstance(X, np.ndarray):
            transformed = self.vectorizers_['array'].transform(X.flatten())
            return transformed.toarray()

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transform text data into TF-IDF features.

    Parameters
    ----------
    max_features : int, default=1000
        Maximum number of features to extract with TF-IDF.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Vectorizer used for converting text data into TF-IDF features.

    Examples
    --------
    >>> text_data = ['sample text data', 'another sample text']
    >>> extractor = TextFeatureExtractor(max_features=500)
    >>> features = extractor.fit_transform(text_data)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the text data using TfidfVectorizer.

    transform(X, y=None)
        Transform the input text data into TF-IDF features.

    """
    def __init__(self, max_features=1000):
        """
        Initialize the TextFeatureExtractor.

        Parameters
        ----------
        max_features : int, default=1000
            Maximum number of features to extract with TF-IDF.

        """
        self.max_features=max_features 
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the text data using TfidfVectorizer.

        Parameters
        ----------
        X : list or array-like, shape (n_samples,)
            Text data to be transformed.

        y : array-like, shape (n_samples,), optional, default=None
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features)
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input text data into TF-IDF features.

        Parameters
        ----------
        X : list or array-like, shape (n_samples,)
            Text data to be transformed.

        y : array-like, shape (n_samples,), optional, default=None
            Target values. Not used in this transformer.

        Returns
        -------
        X_tfidf : sparse matrix, shape (n_samples, n_features)
            Transformed data with TF-IDF features.

        """
        return self.vectorizer.transform(X)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract year, month, and day features from date columns.

    Parameters
    ----------
    date_format : str, default="%Y-%m-%d"
        The date format to use for parsing date columns.

    Examples
    --------
    >>> date_data = pd.DataFrame({'date': ['2021-01-01', '2021-02-01']})
    >>> extractor = DateFeatureExtractor()
    >>> features = extractor.fit_transform(date_data)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the date data.

    transform(X, y=None)
        Transform the input date data into year, month, and day features.
    """
    def __init__(self, date_format="%Y-%m-%d"):
        """
        Initialize the DateFeatureExtractor.

        Parameters
        ----------
        date_format : str, default="%Y-%m-%d"
            The date format to use for parsing date columns.

        """
        self.date_format = date_format
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the date data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Date data to be transformed.

        y : array-like, shape (n_samples,), optional, default=None
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input date data into year, month, and day features.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Date data to be transformed.

        y : array-like, shape (n_samples,), optional, default=None
            Target values. Not used in this transformer.

        Returns
        -------
        new_X : DataFrame, shape (n_samples, n_features * 3)
            Transformed data with year, month, and day features for each date column.

        """
        new_X = X.copy()
        for col in X.columns:
            new_X[col + '_year'] = pd.to_datetime(X[col], format=self.date_format).dt.year
            new_X[col + '_month'] = pd.to_datetime(X[col], format=self.date_format).dt.month
            new_X[col + '_day'] = pd.to_datetime(X[col], format=self.date_format).dt.day
        return new_X

