# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import division, annotations  
import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator,TransformerMixin 

from ..tools._dependency import import_optional_dependency 
from ..tools.validator import build_data_if

__all__=[ 
   "TimeSeriesFeatureExtractor",
   "DateTimeCyclicalEncoder",
   "LagFeatureGenerator",
   "DifferencingTransformer",
   "MovingAverageTransformer",
   "CumulativeSumTransformer",
   "SeasonalDecomposeTransformer",
   "FourierFeaturesTransformer",
   "TrendFeatureExtractor",   
    ]

class FourierFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generate Fourier series terms as features for capturing cyclical 
    patterns in time series data.

    Parameters
    ----------
    periods : list of int
        List of periods to generate Fourier features for.

    Examples
    --------
    >>> from gofast.transformers.ts import FourierFeaturesTransformer
    >>> transformer = FourierFeaturesTransformer(periods=[12, 24])
    >>> X = pd.DataFrame({'time': np.arange(100)})
    >>> fourier_features = transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and does nothing during fitting.

    transform(X, y=None)
        Generate Fourier series terms as features for the input time series data.

    Notes
    -----
    FourierFeaturesTransformer is a transformer that generates Fourier series 
    terms as features for capturing cyclical patterns in time series data. It 
    computes sine and cosine terms for the specified periods using the 
    following formula:

    .. math::

        \text{{sin}}(2\pi ft) \text{{ and }} \text{{cos}}(2\pi ft)

    where:
        - \(f\) is the frequency corresponding to the period \(T\), 
        calculated as \(f = \frac{1}{T}\).
        - \(t\) is the time index of the data.

    The transformer creates two features for each specified period: one for 
    the sine term and one for the cosine term.

    """
    def __init__(self, periods):
        """
        Initialize the FourierFeaturesTransformer transformer.

        Parameters
        ----------
        periods : list of int
            List of periods to generate Fourier features for.

        Returns
        -------
        None

        """
        self.periods = periods
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Generate Fourier series terms as features for the input time series data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input time series data with a 'time' column.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        fourier_features : DataFrame, shape (n_samples, 2 * n_periods)
            Dataframe containing the generated Fourier features.

        """
        X = build_data_if (X, to_frame =True, force=True, input_name='ft', 
                           raise_warning ='mute')
        X_transformed = pd.DataFrame(index=X.index)
        for period in self.periods:
            frequency = 1 / period
            X_transformed[f'sin_{period}'] = np.sin(
                2 * np.pi * frequency * X.index)
            X_transformed[f'cos_{period}'] = np.cos(
                2 * np.pi * frequency * X.index)
        return X_transformed


class TrendFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract linear trend features from time series data by fitting a polynomial
    of a specified order. This transformer allows for flexible handling of time
    series data where the time column and value column can be specified, making
    it suitable for datasets where the date or time information is not set as the
    DataFrame index.

    Parameters
    ----------
    order : int, default=1
        The order of the polynomial trend to fit. For example, `order=1` fits
        a linear trend, `order=2` fits a quadratic trend, and so on.
    time_col : str, optional
        The name of the column in the DataFrame that represents the time variable.
        If None (default), the DataFrame's index is used as the time variable.
    value_col : str, optional
        The name of the column in the DataFrame that represents the value to which
        the trend is to be fitted. If None (default), the first column of the DataFrame
        is used.

    Attributes
    ----------
    No public attributes.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from gofast.transformers import TrendFeatureExtractor
    >>> np.random.seed(0)  # For reproducible output
    >>> X = pd.DataFrame({
    ...     'time': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    ...     'value': np.random.randn(100).cumsum()
    ... })
    >>> transformer = TrendFeatureExtractor(order=1, time_col='time', value_col='value')
    >>> trend_features = transformer.fit_transform(X)
    >>> print(trend_features.head())

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer does not have any
        trainable parameters and, therefore, the fitting process does nothing
        except verifying the correctness of the input data.

    transform(X, y=None)
        Extract linear trend features from the provided time series data. The
        method fits a polynomial of the specified order to the time series data
        and returns the trend component as a new DataFrame.

    Notes
    -----
    The TrendFeatureExtractor is designed to work with time series data where the
    temporal ordering is crucial for analysis. By fitting a trend line to the data,
    this transformer helps in analyzing the underlying trend of the time series,
    which can be particularly useful for feature engineering in predictive modeling
    tasks.
    """
    def __init__(self, order=1, time_col=None, value_col=None):
        self.order = order
        self.time_col = time_col
        self.value_col = value_col
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

        Parameters
        ----------
        X : DataFrame
            Training data. Not used in this transformer.
        y : Not used.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Extract linear trend features from time series data.

        Parameters
        ----------
        X : DataFrame
            Input time series data.
        y : Not used.

        Returns
        -------
        trend_features : DataFrame
            DataFrame containing the extracted trend features.

        """
        if self.time_col is None:
            time_data = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Time column '{self.time_col}' not found in input data.")
            time_data = X[self.time_col]
            
        if self.value_col is None:
            value_data = X.iloc[:, 0]
        else:
            if self.value_col not in X.columns:
                raise ValueError(f"Value column '{self.value_col}' not found in input data.")
            value_data = X[self.value_col]

        # Fit and evaluate the polynomial
        trends = np.polyfit(time_data, value_data, deg=self.order)
        trend_poly = np.poly1d(trends)
        trend_features = pd.DataFrame(trend_poly(time_data),
                                       index=X.index if self.time_col is None else None,
                                       columns=[f'trend_{self.order}'])
        
        return trend_features


class CumulativeSumTransformer(BaseEstimator, TransformerMixin):
    """
    Compute the cumulative sum for each column in the data.

    Parameters
    ----------
    None

    Examples
    --------
    >>> transformer = CumulativeSumTransformer()
    >>> X = pd.DataFrame({'value': np.random.randn(100)})
    >>> cum_sum = transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X, y=None)
        Compute the cumulative sum for each column in the input DataFrame.

    Notes
    -----
    CumulativeSumTransformer is a transformer that computes the cumulative 
    sum of each column in the input DataFrame. It is useful for creating 
    cumulative sums of time series or accumulating data over time.

    The `fit` method is a no-op, as this transformer does not have any 
    trainable parameters. The `transform` method calculates the cumulative 
    sum for each column in the DataFrame.

    """
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Compute the cumulative sum for each column in the input DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing columns for which to compute the 
            cumulative sum.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        cum_sum : DataFrame, shape (n_samples, n_features)
            DataFrame with the cumulative sum of each column.

        """
        X = build_data_if(X, to_frame =True, input_name ='cs', force=True, 
                          raise_warning='silence')
        return X.cumsum()

class SeasonalDecomposeTransformer(BaseEstimator, TransformerMixin):
    """
    Decompose time series data into seasonal, trend, and residual components.

    Parameters
    ----------
    model : str, default='additive'
        Type of seasonal component. Can be 'additive' or 'multiplicative'.

    freq : int, default=1
        Frequency of the time series.

    Examples
    --------
    >>> from gofast.transformer import SeasonalDecomposeTransformer
    >>> transformer = SeasonalDecomposeTransformer(model='additive', freq=12)
    >>> X = pd.DataFrame({'value': np.random.randn(100)})
    >>> decomposed = transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X, y=None)
        Decompose time series data into seasonal, trend, and residual components.

    Notes
    -----
    SeasonalDecomposeTransformer is a transformer that decomposes time series
    data into three components: seasonal, trend, and residual. It uses the 
    seasonal decomposition of time series (STL) method to extract these components.

    The decomposition model can be either 'additive' or 'multiplicative', 
    specified using the 'model' parameter. The 'freq' parameter defines the 
    frequency of the time series, which is used to identify the seasonal 
    component.

    The transformed data will have three columns: 'seasonal', 'trend', and 
    'residual', containing the respective components.

    """
    def __init__(self, model='additive', freq=1):
        """
        Initialize the SeasonalDecomposeTransformer transformer.

        Parameters
        ----------
        model : str, default='additive'
            Type of seasonal component. Can be 'additive' or 'multiplicative'.

        freq : int, default=1
            Frequency of the time series.

        Returns
        -------
        None

        """
        self.model = model
        self.freq = freq
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        
        return self
    
    def transform(self, X, y=None):
        """
        Decompose time series data into seasonal, trend, and residual
        components.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input time series data with a single column.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        decomposed : DataFrame, shape (n_samples, 3)
            Dataframe containing the seasonal, trend, and residual 
            components.

        """
        import_optional_dependency("statsmodels")
        from statsmodels.tsa.seasonal import seasonal_decompose
        result = seasonal_decompose(X, model=self.model, period=self.freq)
        return pd.concat([result.seasonal, result.trend, result.resid], axis=1)



class MovingAverageTransformer(BaseEstimator, TransformerMixin):
    """
    Compute moving average for time series data.

    Parameters
    ----------
    window : int
        Size of the moving window.

    Examples
    --------
    >>> transformer = MovingAverageTransformer(window=5)
    >>> X = pd.DataFrame({'value': np.random.randn(100)})
    >>> moving_avg = transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X, y=None)
        Compute the moving average for each column in the input DataFrame.

    Notes
    -----
    MovingAverageTransformer is a transformer that calculates the moving 
    average of each column in the input DataFrame. It is useful for smoothing
    time series data to identify trends and patterns.

    The `fit` method is a no-op, as this transformer does not have any 
    trainable parameters. The `transform` method calculates the moving 
    average for each column in the DataFrame using a specified moving 
    window size.

    """
    def __init__(self, window):
        """
        Initialize the MovingAverageTransformer.

        Parameters
        ----------
        window : int
            Size of the moving window.

        """
        self.window = window
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Compute the moving average for each column in the input DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing columns for which to compute
            the moving average.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        moving_avg : DataFrame, shape (n_samples, n_features)
            DataFrame with the moving average of each column.

        """
        X = build_data_if(X, to_frame =True, input_name ='mav', force=True, 
                          raise_warning='silence')
        return X.rolling(window=self.window).mean()
class DifferencingTransformer(BaseEstimator, TransformerMixin):
    """
    Initializes the DifferencingTransformer, a transformer designed to
    apply differencing to time series data, making it stationary by removing
    trends and seasonality. Differencing involves subtracting the current value
    from a previous value at a specified lag.

    Parameters
    ----------
    periods : int, default=1
        Specifies the lag, i.e., the number of periods to shift for calculating
        the difference. A period of 1 means subtracting the current value from
        the immediately preceding value, and so on.
    
    zero_first : bool, default=False
        If True, the first value after differencing, which is NaN due to the
        shifting, is replaced with zero. This option is useful for models that
        cannot handle NaN values and ensures the output series has the same
        length as the input.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from gofast.transformers import DifferencingTransformer
    >>> np.random.seed(42)  # For reproducible output
    >>> X = pd.DataFrame({'value': np.random.randn(100).cumsum()})
    >>> transformer = DifferencingTransformer(periods=1, zero_first=True)
    >>> stationary_data = transformer.fit_transform(X)
    >>> print(stationary_data.head())

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and thus the fitting process does nothing except for input
        validation.
    
    transform(X, y=None)
        Applies differencing to the input DataFrame to produce a stationary
        series by removing trends and seasonality.

    Notes
    -----
    Differencing is a common preprocessing step for time series forecasting,
    where data must be stationary to meet the assumptions of various
    forecasting models. The DifferencingTransformer simplifies this process,
    making it easy to integrate into a preprocessing pipeline.
    """
    def __init__(self, periods=1, zero_first=False):
        """
        Initialize the DifferencingTransformer.

        Parameters
        ----------
        periods : int, default=1
            The number of periods to shift for calculating the difference.
        replace_first_with_zero : bool, default=False
            If True, replaces the first differenced value with zero to avoid 
            NaN values.
            This can be useful when the differenced series is used as input to 
            models that do not support NaN values.

        """
        self.periods = periods
        self.zero_first = zero_first
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Apply differencing to the input DataFrame to make it stationary.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            The input DataFrame containing columns for which to apply differencing.
            
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        DataFrame, shape (n_samples, n_features)
            The DataFrame with differenced data to make it stationary. If
            `replace_first_with_zero` is True, the first row of the differenced
            data will be set to zero instead of NaN.

        """
        # Ensure X is a DataFrame
        X = build_data_if (
            X, to_frame =True, input_name ='dt', force=True, 
            raise_warning='mute'
        )
        
        # Apply differencing
        differenced_data = X.diff(periods=self.periods)
        
        # Optionally replace the first value with zero
        if self.zero_first:
            differenced_data.iloc[0] = 0.
        
        return differenced_data
    
  
class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates lag features for time series data to help capture temporal dependencies,
    enhancing model performance by providing historical context.

    This transformer is designed to work with time series data where capturing
    patterns based on past values is crucial. It can create one or multiple lag
    features based on the specified lag periods.

    Parameters
    ----------
    lags : int or list of ints
        Specifies the lag periods for which features should be generated.
        For example, `lags=1` generates a single lag feature (t-1) for each
        data point. A list, e.g., `lags=[1, 2, 3]`, generates multiple lag
        features (t-1, t-2, t-3).
    time_col : str, optional
        The name of the column representing time in the input DataFrame.
        This parameter allows specifying which column to treat as the temporal
        dimension. If None (default), the DataFrame's index is used.
    value_col : str, optional
        Specifies the column from which to generate lag features. Useful for
        DataFrames with multiple columns when only one contains the time series
        data. If None (default), the first column of the DataFrame is used.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.pipeline import make_pipeline
    >>> from gofast.transformers import LagFeatureGenerator
    >>> np.random.seed(0)  # For reproducible output
    >>> X = pd.DataFrame({
    ...     'time': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    ...     'value': np.random.randn(100).cumsum()
    ... })
    >>> generator = LagFeatureGenerator(lags=[1, 2, 3], time_col='time', value_col='value')
    >>> lag_features = generator.fit_transform(X)
    >>> print(lag_features.head())

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and does nothing during fitting.

    transform(X, y=None)
        Generates lag features for the specified value column in the input
        DataFrame, creating new columns for each specified lag period.

    Notes
    -----
    The LagFeatureGenerator is a transformer that can be an essential part of
    the preprocessing pipeline for time series forecasting models. By introducing
    lag features, it allows models to leverage historical data, potentially
    improving forecast accuracy by capturing seasonal trends, cycles, and other
    temporal patterns.
    """
    def __init__(self, lags, time_col=None, value_col=None):
        self.lags = np.atleast_1d(lags).astype(int)
        self.time_col = time_col
        self.value_col = value_col

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self

    def transform(self, X, y=None):
        """
        Generate lag features for the input DataFrame to capture temporal 
        dependencies.

        Parameters
        ----------
        X : DataFrame
            Input DataFrame containing columns for which to generate lag
            features.
        y : Not used.

        Returns
        -------
        DataFrame
            DataFrame with lag features added to capture temporal dependencies.
        """
        X = build_data_if (
            X, 
            to_frame =True, 
            raise_warning='silence', 
            force=True, 
            input_name='lf' 
        )
        
        if self.time_col and self.time_col not in X.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in input data.")
        if self.value_col and self.value_col not in X.columns:
            self.value_col = X.columns[0]  # Default to first column if not specified
        
        # Use specified value column or default to first column
        value_data = X[self.value_col] if self.value_col else X.iloc[:, 0]

        # Generate lag features
        for lag in self.lags:
            X[f'lag_{lag}'] = value_data.shift(lag)

        return X
    
   
class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract common statistical features from time series data for each column.

    Parameters
    ----------
    rolling_window : int
        The size of the moving window to compute the rolling statistics.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> extractor = TimeSeriesFeatureExtractor(rolling_window=5)
    >>> X = pd.DataFrame({'time_series': np.random.rand(100)})
    >>> features = extractor.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. No actual computation is needed for 
        this transformer.

    transform(X, y=None)
        Extract common statistical features from time series data for each column.

    Notes
    -----
    TimeSeriesFeatureExtractor is a transformer that extracts common statistical 
    features from time series data for each column. These features include the
    rolling mean, rolling standard deviation, rolling minimum, rolling maximum,
    and rolling median.

    The `rolling_window` parameter specifies the size of the moving window used
    to compute the rolling statistics. Larger window sizes result in smoother
    statistical features.

    """
    def __init__(self, rolling_window):
        """
        Initialize the TimeSeriesFeatureExtractor.

        Parameters
        ----------
        rolling_window : int
            The size of the moving window to compute the rolling statistics.

        """
        self.rolling_window = rolling_window
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. No actual computation is needed
        for this transformer.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the time series data.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Extract common statistical features from time series data for each column.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the time series data to extract features from.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        features : DataFrame, shape (n_samples, n_features * 5)
            DataFrame with extracted statistical features for each column.
            Features include rolling mean, rolling standard deviation, 
            rolling minimum, rolling maximum, and rolling median.

        """
        X = build_data_if (
            X, to_frame=True, force=True, 
           raise_warning='mute', input_name='tsfe')
        # Rolling statistical features
        return X.rolling(window=self.rolling_window).agg(
            ['mean', 'std', 'min', 'max', 'median'])

class DateTimeCyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode datetime columns as cyclical features using sine and cosine
    transformations.

    Parameters
    ----------
    datetime_features : list of str
        List of datetime column names to be encoded as cyclical.

    Examples
    --------
    >>> encoder = DateTimeCyclicalEncoder(datetime_features=['timestamp'])
    >>> X = pd.DataFrame({'timestamp': pd.date_range(start='1/1/2018', 
                                                     periods=24, freq='H')})
    >>> encoded_features = encoder.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X, y=None)
        Encode datetime columns as cyclical features using sine and cosine 
        transformations.

    Notes
    -----
    DateTimeCyclicalEncoder is a transformer that encodes datetime columns as 
    cyclical features using sine and cosine transformations. This encoding 
    is useful for capturing cyclical patterns in time-based data.

    The `fit` method is a no-op, as this transformer does not have any 
    trainable parameters. The `transform` method encodes the specified 
    datetime columns as cyclical features, adding sine and cosine components
    for the hour of the day.

    """
    def __init__(self, datetime_features):
        """
        Initialize the DateTimeCyclicalEncoder.

        Parameters
        ----------
        datetime_features : list of str
            List of datetime column names to be encoded as cyclical.

        """
        self.datetime_features = datetime_features
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Encode datetime columns as cyclical features using sine and 
        cosine transformations.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing datetime columns to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        encoded_data : DataFrame, shape (n_samples, n_features + 2 * num_datetime_features)
            DataFrame with datetime columns encoded as cyclical features using 
            sine and cosine transformations.

        """
        X = build_data_if ( X, raise_warning ='mute', to_frame=True,
                           force=True, input_name ='dtc')
        X_transformed = X.copy()
        for feature in self.datetime_features:
            dt_col = pd.to_datetime(X_transformed[feature])
            X_transformed[feature + '_sin_hour'] = np.sin(
                2 * np.pi * dt_col.dt.hour / 24)
            X_transformed[feature + '_cos_hour'] = np.cos(
                2 * np.pi * dt_col.dt.hour / 24)
        return X_transformed
   

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   