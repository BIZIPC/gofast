# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Encompasses functions that add value to the data through enrichment 
and summarization.
"""

from __future__ import annotations, print_function 
import warnings 
import numpy as np 
import pandas as pd 

from ..api.summary import ReportFactory
from ..api.types import List,  DataFrame, Optional, Series, Array1D 
from ..api.types import Union, TypeGuard, Tuple, ArrayLike
from ..api.util import get_table_size 
from ..decorators import isdf
from ..tools.coreutils import is_iterable
from ..tools.coreutils import to_numeric_dtypes, exist_features
from ..tools.validator import  build_data_if, is_frame
from ..tools.validator import check_consistent_length 
from ..tools.validator import is_valid_policies

TW = get_table_size() 

__all__=[
     "enrich_data_spectrum", "outlier_performance_impact",
    "prepare_data", "simple_extractive_summary",  
    ]

@isdf 
def prepare_data(
    data: DataFrame, 
    target_column: Optional [Union[str, List[str], Series, Array1D]]=None, 
    encode_categories: bool = False, 
    nan_policy: str='propagate', 
    verbose: bool = False, 
) -> Tuple[DataFrame, Union[Series, DataFrame]]:
    
    """
    Prepares the feature matrix X and target vector y from the provided DataFrame,
    optionally handling categorical encoding and NaN values according to specified
    policies.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing both features and potentially the target
        variables. This DataFrame should be pre-cleaned to some extent, as the
        function assumes initial data cleaning has been done, though additional
        preprocessing options are available through other parameters.

    target_column : Optional[Union[str, List[str], pd.Series, np.ndarray]], default=None
        Specifies the target variable(s) within the provided DataFrame. This can be
        provided as a column name(s) within the DataFrame or as a separate data
        structure that matches the length of `data`. If `None`, only the feature
        matrix X is prepared, facilitating use cases like unsupervised learning.

    encode_categories : bool, default=False
        If True, converts categorical variables within `data` to numeric codes.
        This is accomplished via LabelEncoder, suitable for handling nominal
        categorical data without an inherent ordering. For more complex categorical
        data types or encoding needs, manual preprocessing is recommended.

    nan_policy : str, default='propagate'
        Controls how NaN values are handled in the input data. Options include:
        - 'propagate': Retains NaN values, requiring handling in subsequent steps.
        - 'omit': Removes rows or columns with NaNs, potentially reducing data size
          but ensuring compatibility with models that cannot handle NaNs.
        - 'raise': Raises an error upon detecting NaNs, useful for workflows where
          NaN values indicate critical issues needing resolution.

    verbose : bool, default=False
        When True, outputs detailed information about the processing steps,
        such as the status of categorical encoding and how NaNs are handled.
        This is particularly useful for debugging or detailed analysis phases.

    Returns
    -------
    X : pd.DataFrame  
        A DataFrame containing the feature matrix with the target column(s) 
        removed if they were specified by name within the `data` DataFrame. 
        This matrix is ready for use in machine learning models as input variables.
    
    y : Union[pd.Series, pd.DataFrame]  
        The target data extracted based on `target_column`. It is returned as 
        a pd.Series if a single target is specified or as a pd.DataFrame if 
        multiple targets are identified. For tasks where the target variable 
        is not defined (`target_column` is None), `y` is not returned.
    
    Raises
    ------
    ValueError  
        Raised in several scenarios:
        - If `target_column` is provided and its length does not match the 
          number of samples in `data`, ensuring data integrity and alignment 
          between features and targets.
        - If an invalid `nan_policy` is specified, ensuring that only the 
          predefined policies are applied to handle NaN values in the dataset.
        
    KeyError  
        Triggered if `target_column` is specified by name(s) and any of the 
        names are not found within the `data` DataFrame columns. This ensures 
        that the specified target variable(s) is actually present in the dataset.
    
    Examples
    --------
    Simple usage with automatic encoding of categorical features
    >>> import pandas as pd 
    >>> from gofast.dataops.enrichment import prepare_data 
    >>> df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'C'],
        'target': [0, 1, 0]
    })
    >>> X, y = prepare_data(df, 'target', encode_categories=True)
    >>> print(X)
       feature1  feature2
    0         1         0
    1         2         1
    2         3         2
    >>> print(y)
       target
    0       0
    1       1
    2       0
       
    Handling NaN values by omitting them
    >>> df_with_nan = pd.DataFrame({
        'feature1': [np.nan, 2, 3],
        'feature2': ['A', np.nan, 'C'],
        'target': [1, np.nan, 0]
    })
    >>> X, y = prepare_data(df_with_nan, 'target', nan_policy='omit', verbose=True)
    "NaN values have been omitted from the dataset."
    
    Raising an error when NaN values are found
    >>> try:
    ...    prepare_data(df_with_nan, 'target', nan_policy='raise')
    ... except ValueError as e:
    ...    print(e)
       
    NaN values found in the dataset. Consider using 'omit' to drop them ...
  

    These examples illustrate the function's versatility in handling different 
    data preparation scenarios, from encoding categorical features for machine
    learning models to managing datasets with missing values according to 
    user-defined policies.
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Handle NaN values according to the specified policy
    texts = {}
    nan_policy= is_valid_policies(nan_policy)

    if nan_policy == 'raise' and data.isnull().values.any():
        raise ValueError("NaN values found in the dataset. Consider using 'omit' "
                         "to drop them or 'propagate' to ignore.")
    if nan_policy == 'omit':
        data = data.dropna().reset_index(drop=True)
        texts ['NaN status']= "NaN values have been omitted from the dataset." 
    
    if target_column is not None: 
        if isinstance(target_column, (str, list)):
            target_column= list(is_iterable(
                target_column, exclude_string= True, transform =True )
                )
            exist_features(data, features= target_column )
            y = data[target_column]
            X = data.drop(target_column, axis=1)
                
            y = data[target_column]
            X = data.drop(target_column, axis=1)
            
        elif isinstance(target_column, (pd.Series, np.ndarray)):
            if len(target_column) != len(data):
                raise ValueError("Length of the target array/Series must match "
                                 "the number of samples in 'data'.")
            y = pd.Series(target_column, index=data.index) if isinstance(
                target_column, np.ndarray) else target_column
            X = data
        else:
            raise ValueError("Invalid type for 'target_column'. Must be str, list,"
                             " pd.Series, or one-dimensional array.")
        check_consistent_length(X, y)
        
    else:
        X = data.copy () 
    # Check if there are categorical data columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
   
    if categorical_columns.any():
        # If encode_categories is False, raise a warning about unencoded categories
        msg= "Categorical features detected."
        
        texts ['categories_status']= msg 
        if not encode_categories:
            msg += (
                " It's recommended to encode categorical variables. Set"
                "`encode_categories=True` to automatically encode them."
                )
            texts ['categories_status']= msg # update 
            if not verbose: 
                warnings.warn(msg, UserWarning)
        else:
            texts ['categories_status']= msg 
            # If encode_categories is True, encode the categorical features
            # label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                # Use LabelEncoder to transform the categorical columns
                X[col] = le.fit_transform(X[col])
                # Optionally, keep track of the label encoders for each column
                # label_encoders[col] = le

            texts['encoded_features']= ( 
                f"Encoded categorical feature(s): {list(categorical_columns)}.")
            
    if verbose and texts: 
        print(ReportFactory().add_recommendations(
            texts,  max_char_text= TW ))
        
    return ( X, y ) if target_column is not None else X 

def simple_extractive_summary(
    texts: List[str], raise_exception: bool = True,
    encode: bool = False
    ) -> Union[str, Tuple[str, ArrayLike]]:
    """
    Generates a simple extractive summary from a list of texts. 
    
    Function selects the sentence with the highest term frequency-inverse 
    document frequency (TF-IDF) score and optionally returns its TF-IDF 
    encoding.

    Parameters
    ----------
    texts : List[str]
        A list where each element is a string representing a sentence or 
        passage of text.
    raise_exception : bool, default True
        Raise ValueError if the input list contains only one sentence.
    encode : bool, default False
        If True, returns the TF-IDF encoding of the most representative sentence 
        along with the sentence.

    Returns
    -------
    Union[str, Tuple[str, np.ndarray]]
        The sentence from the input list that has the highest TF-IDF score.
        If 'encode' is True, also returns the sentence's TF-IDF encoded vector.

    Raises
    ------
    ValueError
        If the input list contains less than two sentences and raise_exception is True.

    Examples
    --------
    >>> from gofast.dataops.enrichment import simple_extractive_summary
    >>> messages = [
    ...     "Further explain the background and rationale for the study. "
    ...     "Explain DNA in simple terms for non-scientists. "
    ...     "Explain the objectives of the study which do not seem perceptible. THANKS",
    ...     "We think this investigation is a good thing. In our opinion, it already allows the "
    ...     "initiators to have an idea of what the populations think of the use of DNA in forensic "
    ...     "investigations in Burkina Faso. And above all, know, through this survey, if these "
    ...     "populations approve of the establishment of a possible genetic database in our country."
    ... ]
    >>> summary, encoding = simple_extractive_summary(messages, encode=True)
    >>> print(summary)
    >>> print(encoding) # encoding is the TF-IDF vector of the summary
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if len(texts) < 2:
        if not raise_exception:
            return (texts[0], None) if encode else texts[0]
        raise ValueError("The input list must contain at least two "
                         "sentences for summarization.")

    vectorizer = TfidfVectorizer(stop_words='english')
    # Create a TF-IDF Vectorizer, excluding common English stop words
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Calculate similarity with the entire set of documents
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix).flatten()
    # Exclude the last entry (similarity of the document with itself)
    sorted_idx = np.argsort(similarity_matrix, axis=0)[:-1]
    # The index of the most similar sentence
    most_similar_idx = sorted_idx[-1]
    
    if encode:
        return texts[most_similar_idx], tfidf_matrix[most_similar_idx].todense()

    return texts[most_similar_idx]

@isdf
def outlier_performance_impact(
    data: DataFrame, /, 
    target_column: Union[str, List[str], Series, Array1D],
    test_size: float = 0.2, 
    random_state: int = 42, 
    encode_categories: bool=False, 
    verbose: bool = False) -> Tuple[float, float]:
    """
    Assess the impact of outliers on the predictive performance of a model. 
    
    Applicable for both regression and classification tasks, including 
    multi-label targets.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing features and a target variable.
        
    target_column : Union[str, List[str], pd.Series, np.ndarray]
        The name of the target variable column(s) in the DataFrame, or the target 
        variable array/Series itself. If a string or list of strings is provided, 
        the column(s) will be used as the target variable and removed from the data.
        
    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test split.
        
    random_state : int, optional (default=42)
        The random state to use for reproducible train-test splits.
        
    encode_categories : bool, default False
        If True, categorical features in `data` are automatically encoded using 
        LabelEncoder. This is useful for models that require numerical input for
        categorical data.
    
    verbose : bool
        If True, prints the evaluation metric with and without outliers, 
        and the impact message.
        
    Returns
    -------
    Tuple[float, float]
        A tuple containing the evaluation metric (MSE for regression or 
                                                  accuracy for classification)
        of the model's predictions on the test set with outliers present and 
        with outliers removed.
        
     Raises:
     -------
     KeyError
         If the target column is not present in the DataFrame.
         
     ValueError
         If the test size is not between 0 and 1.
         
    Examples:
    ---------
    >>> import pandas as pd 
    >>> from gofast.dataops.enrichment import outlier_performance_impact
    >>> df = pd.DataFrame({
    ...     'feature1': np.random.rand(100),
    ...     'feature2': np.random.rand(100),
    ...     'target': np.random.rand(100)
    ... })
    >>> mse_with_outliers, mse_without_outliers = outlier_performance_impact(df, 'target')

    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    is_frame (data, df_only= True, raise_exception= True )
    X, y = prepare_data(
        data, target_column, encode_categories=encode_categories,
        nan_policy= 'omit' )
    # Determine if the task is regression or classification
    is_categorical = _is_categorical(y)
    if is_categorical:
        model = LogisticRegression()
        metric = accuracy_score
        metric_name = "Accuracy"
        y = _encode_target(y)
    else:
        model = LinearRegression()
        metric = mean_squared_error
        metric_name = "MSE"
    # Ensure y is a np.ndarray for model fitting
    y = y if isinstance(y, np.ndarray) else y.values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Define an internal function for model evaluation to keep the main logic clean
    def _evaluate_model(X_train: pd.DataFrame, y_train: np.ndarray,
                        X_test: pd.DataFrame, y_test: np.ndarray, 
                        model, metric) -> float:
        """ Train the model and evaluate its performance on the test set."""
        model.fit(X_train, y_train.ravel() )
        predictions = model.predict(X_test)
        return metric(y_test.ravel(), predictions)

    # Evaluate the model on the original data
    original_metric = _evaluate_model(X_train, y_train, X_test, y_test, model, metric)

    # Identify and remove outliers using the IQR method
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1
    is_not_outlier = ~((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).any(axis=1)
    X_train_filtered = X_train[is_not_outlier]
    y_train_filtered = y_train[is_not_outlier]

    # Evaluate the model on the filtered data
    filtered_metric = _evaluate_model(X_train_filtered, y_train_filtered,
                                      X_test, y_test, model, metric)

    # Print results if verbose is True
    if verbose:
        texts ={'original_metric_status': (
            f'{metric_name} with outliers in the training set: {original_metric}'), 
            'filtered_metric_status': (
                f'{metric_name} without outliers in the training set: {filtered_metric}')
            }
        # Check the impact
        if not is_categorical and filtered_metric < original_metric or \
           is_categorical and filtered_metric > original_metric:
            texts['outliers_impact']= (
                'Outliers appear to have a negative impact on the model performance.')
        else:
            texts['outliers_impact']= ('Outliers do not appear to have a significant negative'
                  ' impact on the model performance.')
            
        print(ReportFactory().add_recommendations(texts,  max_char_text= TW ))

    return original_metric, filtered_metric

def _is_categorical(y: Union[Series, DataFrame]) -> bool:
    """
    Determine if the target variable(s) is categorical.

    Parameters
    ----------
    y : Union[pd.Series, pd.DataFrame]
        Target variable(s).

    Returns
    -------
    bool
        True if the target variable(s) is categorical, False otherwise.
    """
    if isinstance(y, pd.DataFrame):
        return all(y.dtypes.apply(lambda dtype: dtype.kind in 'iOc'))
    return y.dtype.kind in 'iOc'

def _encode_target(y: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
    """
    Encode the target variable(s) if it is categorical.

    Parameters
    ----------
    y : Union[pd.Series, pd.DataFrame]
        Target variable(s).

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Encoded target variable(s).
    """
    from sklearn.preprocessing import LabelEncoder
    
    if isinstance(y, pd.DataFrame):
        encoder = LabelEncoder()
        return y.apply(lambda col: encoder.fit_transform(col))
    elif y.dtype.kind == 'O':
        encoder = LabelEncoder()
        return pd.Series(encoder.fit_transform(y), index=y.index)
    return y

 

def enrich_data_spectrum(
    data:DataFrame, /, 
    noise_level:float=0.01, 
    resample_size:int=100, 
    synthetic_size:int=100, 
    bootstrap_size:int=100
    )->TypeGuard[DataFrame]:
    """
    Augment a regression dataset using various techniques. 
    
    The technique including adding noise, resampling, generating 
    synthetic data, and bootstrapping.

    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame with regression data. Assumes numerical features.
    noise_level : float, optional
        The level of Gaussian noise to add, specified as a fraction of the standard
        deviation of each numerical feature (default is 0.01).
    resample_size : int, optional
        Number of data points to generate via resampling without replacement from the
        original dataset (default is 100).
    synthetic_size : int, optional
        Number of synthetic data points to generate through interpolation between
        pairs of existing data points (default is 100).
    bootstrap_size : int, optional
        Number of data points to generate via bootstrapping (sampling with replacement)
        from the original dataset (default is 100).
    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with the original and newly generated data points.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_california_housing
    >>> from gofast.dataops.enrichment import enrich_data_spectrum
    >>> housing = fetch_california_housing()
    >>> data = pd.DataFrame(housing.data, columns=housing.feature_names)
    >>> augmented_data = enrich_data_spectrum(
        data, noise_level=0.02, resample_size=50, synthetic_size=50, bootstrap_size=50)
    >>> print(augmented_data.shape)
    
    Note 
    ------ 
    `enrich_data_spectrum` proposes several techniques that can be used to 
    augment data in regression as explained: 
        
    - Adding Noise: Adds Gaussian noise to each numerical feature based on a 
      specified noise level relative to the feature's standard deviation.
    - Data Resampling: Randomly resamples (without replacement) a specified 
      number of data points from the original dataset.
    - Synthetic Data Generation: Creates new data points by linear interpolation
      between randomly chosen pairs of existing data points.
    - Bootstrapping: Resamples (with replacement) a specified number of data 
      points, allowing for duplicates.
    
    Adjust the parameters like noise_level, resample_size, synthetic_size, 
    and bootstrap_size according to your dataset and needs.
    This function is quite general and may need to be tailored to fit the 
    specific characteristics of your dataset and regression task.
    Be cautious with the synthetic data generation step; ensure that the 
    generated points are plausible within your problem domain.
    Always validate the model performance with and without the augmented 
    data to ensure that the augmentation is beneficial.
    
    """
    from sklearn.utils import resample
    data = build_data_if(data,  to_frame=True, force=True, 
                         input_name="feature_",  raise_warning='mute')
    data = to_numeric_dtypes( data , pop_cat_features= True) 
    if len(data.columns)==0: 
        raise TypeError("Numeric features are expected.")
    augmented_df = data.copy()

    # Adding noise
    for col in data.select_dtypes(include=[np.number]):
        noise = np.random.normal(0, noise_level * data[col].std(),
                                 size=data.shape[0])
        augmented_df[col] += noise

    # Data resampling
    resampled_data = resample(data, n_samples=resample_size, replace=False)
    augmented_df = pd.concat([augmented_df, resampled_data], axis=0)

    # Synthetic data generation
    for _ in range(synthetic_size):
        idx1, idx2 = np.random.choice(data.index, 2, replace=False)
        synthetic_point = data.loc[idx1] + np.random.rand() * (
            data.loc[idx2] - data.loc[idx1])
        # Create a DataFrame from synthetic_point to use with pd.concat
        synthetic_df = pd.DataFrame([synthetic_point], columns=data.columns)
        augmented_df = pd.concat([augmented_df, synthetic_df], ignore_index=True)
        # append deprecated
        # augmented_df = augmented_df.append(synthetic_point, ignore_index=True)

    # Bootstrapping
    bootstrapped_data = resample(data, n_samples=bootstrap_size, replace=True)
    augmented_df = pd.concat([augmented_df, bootstrapped_data], axis=0)

    return augmented_df.reset_index(drop=True)















