# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"Focuses on initial data preparation tasks."

from __future__ import annotations, print_function 
from scipy import stats
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from ..api.summary import ReportFactory
from ..api.types import List,  DataFrame, Optional, Series
from ..api.types import Dict, Union, Tuple, ArrayLike
from ..api.util import get_table_size 
from ..decorators import Dataify, DynamicMethod, DataTransformer
from ..tools.coreutils import is_iterable, assert_ratio
from ..tools.funcutils import ensure_pkg
from ..tools.validator import is_frame, parameter_validator, check_consistent_length 

TW = get_table_size() 

__all__= [ 
    
    "apply_bow_vectorization",
    "apply_tfidf_vectorization",
    "apply_word_embeddings",
    "augment_data",
    "base_transform",
    "boxcox_transformation",
    "transform_dates",
    ]

@Dataify (auto_columns=True)
def transform_dates(
    data: DataFrame, /, 
    transform: bool = True, 
    fmt: Optional[str] = None, 
    return_dt_columns: bool = False,
    include_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    force: bool = False,
    errors: str = 'coerce',
    **dt_kws
) -> Union[pd.DataFrame, List[str]]:
    """
    Detects and optionally transforms columns in a DataFrame that can be 
    interpreted as dates. 
    
    Funtion uses advanced parameters for greater control over the 
    conversion process.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to inspect and process.
    transform : bool, optional
        Determines whether to perform the conversion of detected datetime 
        columns. Defaults to True.
    fmt : str or None, optional
        Specifies the datetime format string to use for conversion. If None, 
        pandas will infer the format.
    return_dt_columns : bool, optional
        If True, the function returns a list of column names detected 
        (and potentially converted) as datetime. 
        Otherwise, it returns the modified DataFrame or the original DataFrame
        if no transformation is performed.
    include_columns : List[str] or None, optional
        Specifies a list of column names to consider for datetime conversion. 
        If None, all columns are considered.
    exclude_columns : List[str] or None, optional
        Specifies a list of column names to exclude from datetime conversion. 
        This parameter is ignored if `include_columns` is provided.
    force : bool, optional
        If True, forces the conversion of columns to datetime objects, even 
        for columns with mixed or unexpected data types.
    errors : str, optional
        Determines how to handle conversion errors. Options are 'raise', 
        'coerce', and 'ignore' (default is 'coerce').
        'raise' will raise an exception for any errors, 'coerce' will convert 
        problematic data to NaT, and 'ignore' will
        return the original data without conversion.
    **dt_kws : dict
        Additional keyword arguments to be passed to `pd.to_datetime`.

    Returns
    -------
    Union[pd.DataFrame, List[str]]
        Depending on `return_dt_columns`, returns either a list of column names 
        detected as datetime or the DataFrame with the datetime conversions 
        applied.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import transform_dates
    >>> data = pd.DataFrame({
    ...     'date': ['2021-01-01', '2021-01-02'],
    ...     'value': [1, 2],
    ...     'timestamp': ['2021-01-01 12:00:00', None],
    ...     'text': ['Some text', 'More text']
    ... })
    >>> transform_dates(data, fmt='%Y-%m-%d', return_dt_columns=True)
    ['date', 'timestamp']

    >>> transform_dates(data, include_columns=['date', 'timestamp'], 
    ...                    errors='ignore').dtypes
    date          datetime64[ns]
    value                  int64
    timestamp     datetime64[ns]
    text                  object
    dtype: object
    """
    # Use the helper function to identify potential datetime columns
    potential_dt_columns = detect_datetime_columns(data)
    
    # Filter columns based on include/exclude lists if provided
    if include_columns is not None:
        datetime_columns = [col for col in include_columns 
                            if col in potential_dt_columns]
    elif exclude_columns is not None:
        datetime_columns = [col for col in potential_dt_columns 
                            if col not in exclude_columns]
    else:
        datetime_columns = potential_dt_columns

    df = data.copy()
    
    if transform:
        for col in datetime_columns:
            if force or col in datetime_columns:
                df[col] = pd.to_datetime(df[col], format=fmt,
                                         errors=errors, **dt_kws)
    
    if return_dt_columns:
        return datetime_columns
    
    return df
    
def detect_datetime_columns(data: DataFrame, / ) -> List[str]:
    """
    Detects columns in a DataFrame that can be interpreted as date and time,
    with an improved check to avoid false positives on purely numeric columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    List[str]
        A list of column names that can potentially be formatted as datetime objects.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import detect_datetime_columns
    >>> data = pd.DataFrame({
    ...     'date': ['2021-01-01', '2021-01-02'],
    ...     'value': [1, 2],
    ...     'timestamp': ['2021-01-01 12:00:00', None],
    ...     'text': ['Some text', 'More text']
    ... })
    >>> detect_datetime_columns(data)
    ['date', 'timestamp']
    """
    datetime_columns = []
    is_frame (data, df_only=True, raise_exception= True, objname='Data' )
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]) and data[col].dropna().empty:
            # Skip numeric columns with no values, as they cannot be dates
            continue
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            try:
                # Attempt conversion on columns with string-like or mixed types
                _ = pd.to_datetime(data[col], errors='raise')
                datetime_columns.append(col)
            except (ValueError, TypeError):
                # If conversion fails, skip the column.
                continue

    return datetime_columns

def boxcox_transformation(
    data: DataFrame, 
    columns: Optional[Union[str, List[str]]] = None, 
    min_value: float = 1, 
    adjust_non_positive: str = 'skip',
    verbose: int = 0, 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
) -> (DataFrame, Dict[str, Optional[float]]):
    """
    Apply Box-Cox transformation to each numeric column of a pandas DataFrame.
    
    The Box-Cox transformation can only be applied to positive data. This function
    offers the option to adjust columns with non-positive values by either 
    skipping those columns or adding a constant to make all values positive 
    before applying the transformation.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing numeric data to transform. Non-numeric columns 
        will be ignored.
    columns : str or list of str, optional
        List of column names to apply the Box-Cox transformation to. If None, 
        the transformation is applied to all numeric columns in the DataFrame.
    min_value : float or int, optional
        The minimum value to be considered positive. Default is 1. Values in 
        columns must be greater than this minimum to apply the Box-Cox 
        transformation.
    adjust_non_positive : {'skip', 'adjust'}, optional
        Determines how to handle columns with values <= min_value:
            - 'skip': Skip the transformation for these columns.
            - 'adjust': Add a constant to all elements in these columns to 
              make them > min_value before applying the transformation.
    verbose : int, optional
        Verbosity mode. 0 = silent, 1 = print messages about columns being 
        skipped or adjusted.
        
    view : bool, optional
        If True, displays visualizations of the data before and after 
        transformation.
    cmap : str, optional
        The colormap for visualizing the data distributions. Default is 
        'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the visualizations. Default is (12, 5).

    Returns
    -------
    transformed_data : pandas.DataFrame
        The DataFrame after applying the Box-Cox transformation to eligible 
        columns.
    lambda_values : dict
        A dictionary mapping column names to the lambda value used for the 
        Box-Cox transformation. Columns that were skipped or not numeric will 
        have a lambda value of None.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.dataops.preprocessing import boxcox_transformation
    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({
    ...     'A': np.random.rand(10) * 100,
    ...     'B': np.random.normal(loc=50, scale=10, size=10),
    ...     'C': np.random.randint(1, 10, size=10)  # Ensure positive values for example
    ... })
    >>> transformed_data, lambda_values = boxcox_transformation(
    ...     data, columns=['A', 'B'], adjust_non_positive='adjust', verbose=1)
    >>> print(transformed_data.head())
    >>> print(lambda_values)
    """
    is_frame (data, df_only=True, raise_exception= True,
              objname='Boxcox transformation' )
    
    transformed_data = pd.DataFrame()
    lambda_values = {}
    verbosity_texts = {}

    if columns is not None:
        columns = is_iterable(columns, exclude_string=True, transform= True )
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            verbosity_texts['missing_columns']=(
                f" Columns {missing_cols} not found in DataFrame."
                 " Skipping these columns.")
        columns = [col for col in columns if col in data.columns]

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    columns_to_transform = columns if columns is not None else numeric_columns
    verbosity_texts['columns_to_transform']=(
        f"Transformed columns: {list(columns_to_transform)}")
    
    # validate adjust_non_positive parameter
    adjust_non_positive=parameter_validator(
        "adjust_non_positive", ["adjust", "skip"], 
        error_msg= (
                "`adjust_non_positive` argument expects ['skip', 'adjust']"
               f" Got: {adjust_non_positive!r}")
        ) (adjust_non_positive)

    skipped_num_columns=[]
    skipped_non_num_columns =[]
    for column in columns_to_transform:
        if column in numeric_columns:
            col_data = data[column]
            if adjust_non_positive == 'adjust' and (col_data <= min_value).any():
                adjustment = min_value - col_data.min() + 1
                col_data += adjustment
                transformed, fitted_lambda = stats.boxcox(col_data)
            elif (col_data > min_value).all():
                transformed, fitted_lambda = stats.boxcox(col_data)
            else:
                transformed = col_data.copy()
                fitted_lambda = None
                skipped_num_columns.append (column)
                
            transformed_data[column] = transformed
            lambda_values[column] = fitted_lambda
        else:
            skipped_non_num_columns.append (column)
            
    if skipped_num_columns: 
        verbosity_texts['skipped_columns']=(
            f"Column(s) '{skipped_num_columns}' skipped: contains values <= {min_value}.")
    if skipped_non_num_columns: 
        verbosity_texts['non_numeric_columns']=(
            f"Column(s) '{skipped_non_num_columns}' is not numeric and will be skipped.")

    # Include non-transformed columns in the returned DataFrame
    for column in data.columns:
        if column not in transformed_data:
            transformed_data[column] = data[column]
            
    if view:
        # Initialize a flag to check the presence of valid data for heatmaps
        valid_data_exists = True
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(fig_size[0] * 2, fig_size[1]))
        
        # Determine columns suitable for the heatmap
        heatmap_columns = columns if columns else data.select_dtypes(
            include=[np.number]).columns
        heatmap_columns = [col for col in heatmap_columns if col in data.columns 
                           and np.issubdtype(data[col].dtype, np.number)]
        
        # Check for non-NaN data in original dataset and plot heatmap
        if heatmap_columns and not data[heatmap_columns].isnull().all().all():
            sns.heatmap(data[heatmap_columns].dropna(axis=1, how='all').corr(), ax=axs[0],
                        annot=True, cmap=cmap)
            axs[0].set_title('Correlation Matrix Before Transformation')
        else:
            valid_data_exists = False
            verbosity_texts['before_transformed_data_status'] = ( 
                'No valid data for Correlation Matrix Before Transformation') 
        
        # Verify transformed_data's structure and plot its heatmap
        if 'transformed_data' in locals() and not transformed_data[heatmap_columns].isnull(
                ).all().all():
            sns.heatmap(transformed_data[heatmap_columns].dropna(
                axis=1, how='all').corr(), ax=axs[1], annot=True, cmap=cmap)
            axs[1].set_title('Correlation Matrix After Transformation')
        else:
            valid_data_exists = False
            verbosity_texts['after_transformed_data_status'] = ( 
                'No valid data for Correlation Matrix After Transformation') 
        
        # Display the plots if valid data exists; otherwise, close the plot
        # to avoid displaying empty figures
        if valid_data_exists:
            plt.tight_layout()
            plt.show()
        else:
            verbosity_texts['matplotlib_window_status']='Closed'
            plt.close()  # Closes the matplotlib window if no valid data is present
    
    # Print verbose recommendations if any and if verbose mode is enabled
    if verbose and verbosity_texts:
        recommendations = ReportFactory('BoxCox Transformation').add_recommendations(
            verbosity_texts, max_char_text=TW)
        print(recommendations)
        
    return transformed_data, lambda_values


@DataTransformer('data', mode='lazy')
@DynamicMethod(
    'both', 
    capture_columns=True, 
    prefixer='exclude' 
   )
def base_transform(
    data: DataFrame, 
    target_columns:Optional[str|List[str]]=None, 
    columns: Optional[str|List[str]]=None, 
    noise_level: float=None, 
    seed: int=None):
    """
    Applies preprocessing transformations to the specified DataFrame, including 
    handling of missing values, feature scaling, encoding categorical variables, 
    and optionally introducing noise to numeric features. Transformations can 
    be selectively applied to specified columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to undergo preprocessing transformations.
    target_columns : str or list of str, optional
        The name(s) of the column(s) considered as target variable(s). 
        These columns are excluded from transformations to prevent leakage. 
        Default is None, indicating
        no columns are treated as targets.
    columns : str or list of str, optional
        Specific columns to which transformations should be applied. 
        If None, transformations
        are applied to all columns excluding `target_columns`. 
        Default is None.
    noise_level : float, optional
        The level of noise (as a fraction between 0 and 1) to introduce into 
        the numeric columns. If None, no noise is added. Default is None.
    seed : int, optional
        Seed for the random number generator, ensuring reproducibility of 
        the noise and other random aspects of preprocessing. Default is None.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame with numeric features scaled, missing 
        values imputed, categorical variables encoded, and optionally noise 
        added.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification

    # Generating a synthetic dataset
    >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    >>> data = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3',
    ...                                 'feature_4'])
    >>> data['target'] = y

    # Apply base_transform to preprocess features, excluding the target column
    >>> from gofast.dataops.preprocessing import base_transform 
    >>> preprocessed_data = base_transform(data, target_columns='target', 
    ...                                    noise_level=0.1, seed=42)
    >>> print(preprocessed_data.head())

    Note
    ----
    This function is designed to be flexible, allowing selective preprocessing
    on parts of the DataFrame. It leverages `ColumnTransformer` to efficiently 
    process columns based on their data type. The inclusion of `noise_level` 
    allows for simulating real-world data imperfections.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    
    np.random.seed(seed)
    target_columns = [target_columns] if isinstance(
        target_columns, str) else target_columns

    if target_columns is not None:
        data = data.drop(columns=target_columns, errors='ignore')
    # get original data columns 
    original_columns = list(data.columns  )
    # Identify numeric and categorical features for transformation
    numeric_features = data.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(
        include=['object']).columns.tolist()

    # Define transformations for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Apply ColumnTransformer to handle each feature type
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    data_processed = preprocessor.fit_transform(data)
    processed_columns = numeric_features + categorical_features 
    try: 
        data_processed = pd.DataFrame(
            data_processed, columns=processed_columns,  
            index=data.index)
    except : 
        data_processed = pd.DataFrame(
            data_processed, index=data.index)

    # Apply noise to non-target numeric columns if specified
    if noise_level is not None:
        noise_level = assert_ratio ( noise_level )
        # Add noise if specified
        # if  noise_level > 0:
        #     noise_mask = np.random.rand(*data_processed.shape) < noise_level
        #     data_processed.where(~noise_mask, other=np.nan, inplace=True)
        assert 0 <= noise_level <= 1, "noise_level must be between 0 and 1"
        for column in numeric_features:
            noise_mask = np.random.rand(data.shape[0]) < noise_level
            data_processed.loc[noise_mask, column] = np.nan
    # now rearrange the columns back 
    try:
        return  data_processed [original_columns ]
    except:
        return data_processed # if something wrong, return it

def augment_data(
    X: Union[DataFrame, ArrayLike], 
    y: Optional[Union[Series, np.ndarray]] = None, 
    augmentation_factor: int = 2, 
    shuffle: bool = True
) -> Union[Tuple[Union[DataFrame, ArrayLike], Optional[Union[
    Series, ArrayLike]]], Union[DataFrame, ArrayLike]]:
    """
    Augment a dataset by repeating it with random variations to enhance 
    training diversity.

    This function is useful in scenarios with limited data, helping improve the 
    generalization of machine learning models by creating a more diverse training set.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Input data, either as a Pandas DataFrame or a NumPy ndarray.

    y : Optional[Union[pd.Series, np.ndarray]], optional
        Target labels, either as a Pandas Series or a NumPy ndarray. If `None`, 
        the function only processes the `X` data. This is useful in unsupervised 
        learning scenarios where target labels may not be applicable.
    augmentation_factor : int, optional
        The multiplier for data augmentation. Defaults to 2 (doubling the data).
    shuffle : bool, optional
        If True, shuffle the data after augmentation. Defaults to True.

    Returns
    -------
    X_augmented : pd.DataFrame or np.ndarray
        Augmented input data in the same format as `X`.
    y_augmented : pd.Series, np.ndarray, or None
        Augmented target labels in the same format as `y`, if `y` is not None.
        Otherwise, None.

    Raises
    ------
    ValueError
        If `augmentation_factor` is less than 1 or if the lengths of `X` and `y` 
        are mismatched when `y` is not None.

    Raises
    ------
    ValueError
        If `augmentation_factor` is less than 1 or if `X` and `y` have mismatched lengths.

    Examples
    --------
    >>> from gofast.dataops.preprocessing import augment_data 
    >>> X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
    >>> X_aug, y_aug = augment_data(X, y)
    >>> X_aug.shape, y_aug.shape
    ((4, 2), (4,))
    >>> X = np.array([[1, 2], [3, 4]])
    >>> X_aug = augment_data(X, y=None)
    >>> X_aug.shape
    (4, 2)
    """
    from sklearn.utils import shuffle as shuffle_data
    
    if augmentation_factor < 1:
        raise ValueError("Augmentation factor must be at least 1.")

    is_X_df = isinstance(X, pd.DataFrame)
    is_y_series = isinstance(y, pd.Series) if y is not None else False

    if is_X_df:
        # Separating numerical and categorical columns
        num_columns = X.select_dtypes(include=['number']).columns
        cat_columns = X.select_dtypes(exclude=['number']).columns

        # Augment only numerical columns
        X_num = X[num_columns]
        X_num_augmented = np.concatenate([X_num] * augmentation_factor)
        X_num_augmented += np.random.normal(loc=0.0, scale=0.1 * X_num.std(axis=0), 
                                            size=X_num_augmented.shape)
        # Repeat categorical columns without augmentation
        X_cat_augmented = pd.concat([X[cat_columns]] * augmentation_factor
                                    ).reset_index(drop=True)

        # Combine numerical and categorical data
        X_augmented = pd.concat([pd.DataFrame(
            X_num_augmented, columns=num_columns), X_cat_augmented], axis=1)
   
    else:
        # If X is not a DataFrame, it's treated as a numerical array
        X_np = np.asarray(X, dtype= float) 
        X_augmented = np.concatenate([X_np] * augmentation_factor)
        X_augmented += np.random.normal(
            loc=0.0, scale=0.1 * X_np.std(axis=0), size=X_augmented.shape)

    y_np = np.asarray(y) if y is not None else None
    
    # Shuffle if required
    if y_np is not None:
        check_consistent_length(X, y_np )
        y_augmented = np.concatenate([y_np] * augmentation_factor)
        if shuffle:
            X_augmented, y_augmented = shuffle_data(X_augmented, y_augmented)
        if is_y_series:
            y_augmented = pd.Series(y_augmented, name=y.name)
            
        return X_augmented, y_augmented

    else:
        if shuffle:
            X_augmented = shuffle_data(X_augmented)
            
        return X_augmented

def apply_tfidf_vectorization(
    data: DataFrame,/,
    text_columns: Union[str, List[str]],
    max_features: int = 100,
    stop_words: Union[str, List[str]] = 'english',
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
  ) -> DataFrame:
    """
    Applies TF-IDF (Term Frequency-Inverse Document Frequency) vectorization 
    to one or more text columns in a pandas DataFrame. 
    
    Function concatenates the resulting features back into the original 
    DataFrame.
    
    TF-IDF method weighs the words based on their occurrence in a document 
    relative to their frequency across all documents, helping to highlight 
    words that are more interesting, i.e., frequent in a document but not 
    across documents.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to vectorize.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data.
    max_features : int, optional
        The maximum number of features to generate. Defaults to 100.
    stop_words : Union[str, List[str]], optional
        The stop words to use for the TF-IDF vectorizer. Can be 'english' or 
        a custom list of stop words. Defaults to 'english'.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. 'fill' will 
        replace them with `fill_value`, 'ignore' will keep them as is, and 
        'drop' will remove rows with missing values. Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` if 
        `missing_value_handling` is 'fill'. Defaults to an empty string.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True.

    Returns
    -------
    pd.DataFrame
        The original DataFrame concatenated with the TF-IDF features. The 
        original text column(s) can be optionally dropped.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import apply_tfidf_vectorization
    >>> data = pd.DataFrame({
    ...     'message_to_investigators': ['This is a sample message', 'Another sample message', np.nan],
    ...     'additional_notes': ['Note one', np.nan, 'Note three']
    ... })
    >>> processed_data = apply_tfidf_vectorization(
    ... data, text_columns=['message_to_investigators', 'additional_notes'])
    >>> processed_data.head()
    """
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(text_columns, exclude_string= True, transform=True)
    tfidf_features_df = pd.DataFrame()

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value
            )
        column_tfidf_df = _generate_tfidf_features(
            handled_data, max_features, stop_words
            )
        tfidf_features_df = pd.concat(
            [tfidf_features_df, column_tfidf_df], 
            axis=1
            )

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat(
        [data.reset_index(drop=True), tfidf_features_df.reset_index(drop=True)
         ], axis=1
    )

    return prepared_data


@Dataify(auto_columns=True) 
def apply_word_embeddings(
    data: DataFrame,/, 
    text_columns: Union[str, List[str]],
    embedding_file_path: str,
    n_components: int = 50,
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
  ) -> DataFrame:
    """
    Applies word embedding vectorization followed by dimensionality reduction 
    to text columns in a pandas DataFrame.
    
    This process converts text data into a numerical form that captures 
    semantic relationships between words, making it suitable for use in machine
    learning models. The function leverages pre-trained word embeddings 
    (e.g., Word2Vec, GloVe) to represent words in a high-dimensional space and 
    then applies PCA (Principal Component Analysis) to reduce the
    dimensionality of these embeddings to a specified number of components.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to be processed.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data to be 
        vectorized. Can be a single column name or a list of names for multiple
        columns.
    embedding_file_path : str
        The file path to the pre-trained word embeddings. This file should be 
        in a format compatible with Gensim's KeyedVectors, such as Word2Vec's .
        bin format or GloVe's .txt format.
    n_components : int, optional
        The number of dimensions to reduce the word embeddings to using PCA. 
        Defaults to 50, balancing between retaining
        semantic information and ensuring manageability for machine learning models.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. Options are:
        - 'fill': Replace missing values with `fill_value`.
        - 'drop': Remove rows with missing values in any of the specified text columns.
        - 'ignore': Leave missing values as is, which may affect the embedding process.
        Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` 
        if `missing_value_handling` is 'fill'. This can be an empty string 
        (default) or any placeholder text.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True, removing the text
        columns to only include the generated features and original non-text data.

    Returns
    -------
    pd.DataFrame
        A DataFrame consisting of the original DataFrame 
        (minus the text columns if `drop_text_columns` is True) concatenated
        with the dimensionality-reduced word embedding features. The new 
        features are numerical and ready for use in machine learning models.

    Examples
    --------
    Assuming we have a DataFrame `df` with a text column 'reviews' and 
    pre-trained word embeddings stored at 'path/to/embeddings.bin':

    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import apply_word_embeddings
    >>> df = pd.DataFrame({'reviews': [
    ...  'This product is great', 'Terrible customer service', 'Will buy again', 
    ... 'Not worth the price']})
    >>> processed_df = apply_word_embeddings(df,
    ...                                      text_columns='reviews',
    ...                                      embedding_file_path='path/to/embeddings.bin',
    ...                                      n_components=50,
    ...                                      missing_value_handling='fill',
    ...                                      fill_value='[UNK]',
    ...                                      drop_text_columns=True)
    >>> processed_df.head()

    This will create a DataFrame with 50 new columns, each representing a 
    component of the reduced dimensionality word embeddings, ready for further 
    analysis or machine learning.
    """
    embeddings = _load_word_embeddings(embedding_file_path)
    
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(
        text_columns, exclude_string= True, transform=True)
    
    if isinstance(text_columns, str):
        text_columns = [text_columns]

    all_reduced_embeddings = []

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value
            )
        avg_embeddings = _average_word_embeddings(handled_data, embeddings)
        reduced_embeddings = _reduce_dimensions(avg_embeddings, n_components)
        all_reduced_embeddings.append(reduced_embeddings)

    # Combine reduced embeddings into a DataFrame
    embeddings_df = pd.DataFrame(np.hstack(all_reduced_embeddings))

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat([data.reset_index(drop=True), 
                               embeddings_df.reset_index(drop=True)],
                              axis=1
                              )
    return prepared_data

def _generate_bow_features(
        text_data: pd.Series, max_features: int, 
        stop_words: Union[str, List[str]]
    ) -> pd.DataFrame:
    """
    Generates Bag of Words (BoW) features for a given text data Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    max_features : int
        The maximum number of features to generate.
    stop_words : Union[str, List[str]]
        The stop words to use for the BoW vectorizer. Can be 'english' or a 
        custom list of stop words.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the BoW features for the text data.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    bow_vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
    bow_features = bow_vectorizer.fit_transform(text_data).toarray()
    feature_names = [f'bow_{i}' for i in range(bow_features.shape[1])]
    return pd.DataFrame(bow_features, columns=feature_names, index=text_data.index)

@ensure_pkg("gensim","Word-Embeddings expect 'gensim'to be installed." )
def _load_word_embeddings(embedding_file_path: str):
    """
    Loads pre-trained word embeddings from a file.

    Parameters
    ----------
    embedding_file_path : str
        Path to the file containing the pre-trained word embeddings.

    Returns
    -------
    KeyedVectors
        The loaded word embeddings.
    """
    from gensim.models import KeyedVectors
    embeddings = KeyedVectors.load_word2vec_format(embedding_file_path, binary=True)
    
    return embeddings

def _average_word_embeddings(
        text_data: Series, embeddings
    ) -> ArrayLike:
    """
    Generates an average word embedding for each text sample in a Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    embeddings : KeyedVectors
        The pre-trained word embeddings.

    Returns
    -------
    np.ndarray
        An array of averaged word embeddings for the text data.
    """
    def get_embedding(word):
        try:
            return embeddings[word]
        except KeyError:
            return np.zeros(embeddings.vector_size)

    avg_embeddings = text_data.apply(lambda x: np.mean(
        [get_embedding(word) for word in x.split() if word in embeddings],
        axis=0)
        )
    return np.vstack(avg_embeddings)

def _reduce_dimensions(
        embeddings: ArrayLike, n_components: int = 50
        ) -> ArrayLike:
    """
    Reduces the dimensionality of word embeddings using PCA.

    Parameters
    ----------
    embeddings : np.ndarray
        The word embeddings array.
    n_components : int
        The number of dimensions to reduce to.

    Returns
    -------
    np.ndarray
        The dimensionality-reduced word embeddings.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def _handle_missing_values(
        column_data: Series, missing_value_handling: str, fill_value: str = ''
   ) -> Series:
    """
    Handles missing values in a pandas Series according to the specified method.

    Parameters
    ----------
    column_data : pd.Series
        The Series (column) from the DataFrame for which missing values 
        need to be handled.
    missing_value_handling : str
        The method for handling missing values: 'fill', 'drop', or 'ignore'.
    fill_value : str, optional
        The value to use for filling missing values if `missing_value_handling`
        is 'fill'. Defaults to an empty string.

    Returns
    -------
    pd.Series
        The Series with missing values handled according to the specified method.

    Raises
    ------
    ValueError
        If an invalid `missing_value_handling` option is provided.
    """
    if missing_value_handling == 'fill':
        return column_data.fillna(fill_value)
    elif missing_value_handling == 'drop':
        return column_data.dropna()
    elif missing_value_handling == 'ignore':
        return column_data
    else:
        raise ValueError("Invalid missing_value_handling option. Choose"
                         " 'fill', 'drop', or 'ignore'.")
        
def _generate_tfidf_features(
        text_data: pd.Series, max_features: int, 
        stop_words: Union[str, List[str]]) -> pd.DataFrame:
    """
    Generates TF-IDF features for a given text data Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    max_features : int
        The maximum number of features to generate.
    stop_words : Union[str, List[str]]
        The stop words to use for the TF-IDF vectorizer. Can be 'english' 
        or a custom list of stop words.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the TF-IDF features for the text data.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    tfidf_features = tfidf_vectorizer.fit_transform(text_data).toarray()
    feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    return pd.DataFrame(tfidf_features, columns=feature_names, index=text_data.index)

def apply_bow_vectorization(
    data: DataFrame, /, 
    text_columns: Union[str, List[str]],
    max_features: int = 100,
    stop_words: Union[str, List[str]] = 'english',
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
 ) -> pd.DataFrame:
    """
    Applies Bag of Words (BoW) vectorization to one or more text columns in 
    a pandas DataFrame. 
    
    Function concatenates the resulting features back into the original 
    DataFrame.
    
    Bow is a simpler approach that creates a vocabulary of all the unique 
    words in the dataset and then models each text as a count of the number 
    of times each word appears.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to vectorize.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data.
    max_features : int, optional
        The maximum number of features to generate. Defaults to 100.
    stop_words : Union[str, List[str]], optional
        The stop words to use for the BoW vectorizer. Can be 'english' or 
        a custom list of stop words. Defaults to 'english'.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. 'fill' will 
        replace them with `fill_value`, 'ignore' will keep them as is, and 
        'drop' will remove rows with missing values. Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` if 
        `missing_value_handling` is 'fill'. Defaults to an empty string.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True.

    Returns
    -------
    pd.DataFrame
        The original DataFrame concatenated with the BoW features. The 
        original text column(s) can be optionally dropped.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import apply_bow_vectorization
    >>> data = pd.DataFrame({
    ...     'message_to_investigators': ['This is a sample message', 'Another sample message', np.nan],
    ...     'additional_notes': ['Note one', np.nan, 'Note three']
    ... })
    >>> processed_data = apply_bow_vectorization(
    ... data, text_columns=['message_to_investigators', 'additional_notes'])
    >>> processed_data.head()
    """
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(
        text_columns, exclude_string= True, transform=True)

    bow_features_df = pd.DataFrame()

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value)
        column_bow_df = _generate_bow_features(
            handled_data, max_features, stop_words)
        bow_features_df = pd.concat([bow_features_df, column_bow_df], axis=1)

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat([data.reset_index(drop=True),
                               bow_features_df.reset_index(drop=True)],
                              axis=1
                              )

    return prepared_data



    
    
    
    






















