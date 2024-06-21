# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
import copy
import warnings 
import numpy as np
import pandas as pd
from ..compat.pandas import iteritems_compat 

from .extension import RegexMap, isinstance_, fetch_estimator_name 
from .formatter import MultiFrameFormatter, DataFrameFormatter, DescriptionFormatter 
from .box import KeyBox 
from .structures import FlexDict
from .util import to_snake_case, get_table_size 
from .util import format_value, df_to_custom_dict, format_text, to_camel_case  
from .util import find_maximum_table_width, format_df, format_correlations
from .util import beautify_dict 

class ResultSummary:
    """
    Initializes a ResultSummary object which can store, format, and display
    results in a structured format. The class allows for optional customization
    of the display settings such as padding of keys and maximum character limits
    for value display.
    
    Parameters
    ----------
    name : str, optional
        The name of the result set, which will be displayed as the title
        of the output. Defaults to "Result".
    pad_keys : str, optional
        If set to "auto", keys in the result dictionary will be left-padded
        to align with the longest key, enhancing readability.
    max_char : int, optional
        The maximum number of characters that a value can have before being
        truncated. Defaults to ``100``.
    flatten_nested_dicts : bool, optional
        Determines whether nested dictionaries within the results should be 
        displayed in a flattened, one-line format.
        When set to ``True``, nested dictionaries are presented as a compact single
        line, which might be useful for brief overviews or when space is limited.
        If set to ``False``, nested dictionaries are displayed with full 
        indentation and key alignment, which improves readability for complex 
        structures. Defaults to ``True``.
    mute_note: bool, default=False 
       Skip displaying the note after result formatage. 
       
    Examples
    --------
    >>> from gofast.api.summary import ResultSummary
    >>> summary = ResultSummary(name="Data Check", pad_keys="auto", max_char=50)
    >>> results = {
        'long_string_data': "This is a very long string that needs truncation.",
        'data_counts': {'A': 20, 'B': 15}
    }
    >>> summary.add_results(results)
    >>> print(summary)
    DataCheck(
      {
        long_string_data          : "This is a very long string that needs trunc..."
        data_counts               : {'A': 20, 'B': 15}
      }
    )
    """
    def __init__(self, name=None, pad_keys=None, max_char=None,
                 flatten_nested_dicts =True, mute_note=False):
        """
        Initialize the ResultSummary with optional customization for display.
        """
        self.name = name or "Result"
        self.pad_keys = pad_keys
        self.max_char = max_char or get_table_size()
        self.flatten_nested_dicts = flatten_nested_dicts 
        self.mute_note=mute_note
        self.results = {}
        
    def add_results(self, results):
        """
        Adds results to the summary and dynamically creates attributes for each
        key in the results dictionary, converting keys to snake_case for attribute
        access.

        Parameters
        ----------
        results : dict
            A dictionary containing the results to add to the summary. Keys should
            be strings and will be converted to snake_case as object attributes.

        Raises
        ------
        TypeError
            If the results parameter is not a dictionary.

        Examples
        --------
        >>> summary = ResultSummary()
        >>> summary.add_results({'Missing Data': {'A': 20}})
        >>> print(summary.missing_data)
        {'A': 20}
        """
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary")
    
        # Deep copy to ensure that changes to input dictionary 
        # do not affect internal state
        self.results = copy.deepcopy(results)
    
        # Apply snake_case to dictionary keys and set attributes
        for name in list(self.results.keys()):
            snake_name = to_snake_case(name)
            setattr(self, snake_name, self.results[name])
            
        return self 

    def __str__(self):
        """
        Return a formatted string representation of the results dictionary.
        """
        result_title = to_camel_case(self.name)+ '(\n  {\n'
        formatted_results = []
        
        # Determine key padding if auto pad_keys is specified
        if self.pad_keys == "auto":
            max_key_length = max(len(key) for key in self.results.keys())
            key_padding = max_key_length
        else:
            key_padding = 0  # No padding by default
    
        # Construct the formatted result string
        for key, value in self.results.items():
            if self.pad_keys == "auto":
                formatted_key = key.ljust(key_padding)
            else:
                formatted_key = key
            if isinstance(value, dict):
                if self.flatten_nested_dicts: 
                    value_str= str(value)
                else: 
                    value_str = beautify_dict(
                        value, key=f"       {formatted_key}",
                        max_char= self.max_char
                        ) 
                    formatted_results.append(value_str +',') 
                    continue 
            else:
                value_str = str(value)
            
            # Truncate values if necessary
            if len(value_str) > self.max_char:
                value_str = value_str[:self.max_char] + "..."
            
            formatted_results.append(f"       {formatted_key} : {value_str}")
    
        result_str = '\n'.join(formatted_results) + "\n\n  }\n)"
        entries_summary = f"[ {len(self.results)} entries ]"
        result_str += f"\n\n{entries_summary}"
        # If ellipsis (...) is present in the formatted result, it indicates 
        # that some data has been truncated
        # for brevity. For the complete dictionary result, please access the 
        # corresponding attribute."
        note = ( "\n\n[ Note: Data may be truncated. For the complete dictionary"
                " data, access the corresponding 'results' attribute ]"
                ) if "..." in result_str else ''
        if self.mute_note: 
            note =''
        return f"{result_title}\n{result_str}{note}"

    def __repr__(self):
        """
        Return a developer-friendly representation of the ResultSummary.
        """
        name =to_camel_case(self.name)
        return ( f"<{name} with {len(self.results)} entries."
                " Use print() to see detailed contents.>") if self.results else ( 
                    f"<Empty {name}>")

class ModelSummary(KeyBox):
    """
    A class for creating and storing a summary of machine learning model
    tuning results. Inherits from FlexDict for flexible attribute management.
    
    This class facilitates the generation and representation of model tuning
    results as a structured summary report. It leverages the
    `summarize_optimized_results` function to format the results into a
    comprehensive summary.

    Parameters
    ----------
    title : str, optional
        The title for the summary report. If provided, it will be included
        at the beginning of the summary report.
    **kwargs : dict, optional
        Additional keyword arguments that are passed to the FlexDict constructor
        for further customization of the object.
    descriptor : str, optional
        A dynamic label or descriptor that defines the identity of the output
        when the object is represented as a string. This label is used in the 
        representation of the object in outputs like print statements. If not
        provided, defaults to 'ModelSummary'. 
        
    Attributes
    ----------
    title : str
        The title of the summary report.
    summary_report : str
        The formatted summary report as a string.

    Methods
    -------
    summary(model_results, **kwargs)
        Generates and stores a summary report based on the provided model results.
    """
    
    def __init__(self, title=None, descriptor=None, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.descriptor=descriptor 
        self.summary_report = ""
        
    def summary(self, model_results, title=None, **kwargs):
        """
        Generates a summary report from the provided model tuning results and
        stores it in the object.

        Parameters
        ----------
        model_results : dict
            The results of model tuning to be summarized. Can include results
            from a single model or multiple models.
        title : str
            The title of the summary report.
        **kwargs : dict
            Additional keyword arguments passed to the summarize_optimized_results
            function for customization of the summary generation process.

        Returns
        -------
        self : ModelSummary
            The ModelSummary instance with the generated summary report stored.
        """
        self.summary_report = summarize_optimized_results(
            model_results, result_title=title or self.title, **kwargs)
        return self
    
    def add_multi_contents(
            self, *dict_contents, titles=None, headers=None, **kwargs):
        """
        Incorporates one or more dictionaries into the summary report of the 
        ModelSummary instance, formatting them into a cohesive summary report.
        This method leverages the `summarize_tables` function to generate a 
        formatted string representation of the provided table data, which is 
        then assigned to the instance's `summary_report` attribute.
    
        This method is designed to aggregate and format multiple tables of 
        data (e.g., model performance metrics) into a single summary report 
        that is easy to read and interpret. It supports the inclusion of 
        titles and headers for individual tables and allows for additional 
        formatting options through keyword arguments.
    
        Parameters
        ----------
        *dict_contents : dict or list of dict
            The table(s) to be added to the summary report. Each table can 
            be directly provided as a dictionary, a list of dictionaries for 
            multiple tables, or a nested dictionary where each key is considered 
            a table name.
        titles : list of str, optional
            The titles for each table or group of tables within the summary 
            report. If not provided, titles will be omitted.
        headers : list of str, optional
            The headers for each table within the summary report. If not 
            provided, headers will be derived from the content if possible.
        **kwargs : dict
            Additional keyword arguments to be passed to the `summarize_tables`
            function for table formatting.
    
        Returns
        -------
        ModelSummary
            The instance itself, allowing for method chaining.
    
        Examples
        --------
        >>>  >>> from gofast.api.summary import ModelSummary
        >>> summary = ModelSummary()
        >>> dict_contents = [{
        ...     "Estimator": {"Accuracy": 0.95, "Precision": 0.89, "Recall": 0.93},
        ...     "RandomForest": {"Accuracy": 0.97, "Precision": 0.91, "Recall": 0.95}
        ... }]
        >>> summary.add_multi_contents(*dict_contents, titles=["Model Performance"])
        >>> print(summary.summary_report)
        Model Performance
        ========================
                 Estimator       
        ------------------------
          Accuracy   : 0.9500
          Precision  : 0.8900
          Recall     : 0.9300
        ========================
              RandomForest       
        ------------------------
          Accuracy   : 0.9700
          Precision  : 0.9100
          Recall     : 0.9500
        ========================
    
        The `add_dict_contents` method facilitates easy aggregation and formatting 
        of model performance data or similar tabular data into a comprehensive 
        summary report, enhancing the interpretability and presentation of the 
        data.
        """
        self.summary_report= summarize_tables(
            *dict_contents, titles =titles, headers =headers, **kwargs)
        
        return self 
    
    def add_performance(self, model_results, **kwargs):
        """
        Formats and adds a summary of model performance evaluation results to 
        the instance's summary report. This method uses the `summarize_model_results` 
        function to create a formatted string representation of the model's 
        performance, including best parameters, estimator, and cross-validation results.
    
        Parameters
        ----------
        model_results : dict
            A dictionary containing the results of the model performance evaluation.
            Expected keys include 'best_parameters_', 'best_estimator_', and 
            'cv_results_', among others.
        **kwargs : dict
            Additional keyword arguments to be passed to the `summarize_model_results`
            function for further customization of the summary output.
    
        Returns
        -------
        self : object
            The instance itself, allowing for method chaining.
    
        Examples
        --------
        >>> from gofast.api.summary import ModelSummary
        >>> model_results = {
        ...    'best_parameters_': {'C': 1, 'gamma': 0.1},
        ...    'best_estimator_': "SVC",
        ...    'cv_results_': {
        ...        'split0_test_score': [0.6789, 0.8],
        ...        'split1_test_score': [0.5678, 0.9],
        ...        'split2_test_score': [0.9807, 0.95],
        ...        'split3_test_score': [0.8541, 0.85],
        ...        'params': [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01}],
        ...    },
        ...    'scoring': 'accuracy',
        ... }
        >>> summary = ModelSummary(title="SVC Performance")
        >>> summary.add_performance(model_results)
        >>> print(summary.summary_report)
                             Model Results                  
        ====================================================
        Best estimator   : SVC
        Best parameters  : {'C': 1, 'gamma': 0.1}
        CV               : 4 folds
        Scoring          : accuracy
        ====================================================
        
                             Tuning Results                 
        ====================================================
          Fold Mean score CV score std score Global mean
        --------------------------------------------------
        0  cv1     0.6233   0.6789    0.0555      0.6233
        1  cv2     0.7339   0.9000    0.1661      0.7339
        2  cv3     0.9653   0.9807    0.0154      0.9653
        3  cv4     0.8520   0.8541    0.0021      0.8520
        ====================================================
    
        The `add_performance` method simplifies the inclusion of detailed model 
        evaluation results into the summary report, providing a clear and structured 
        presentation of the model's performance.
        """
        
        self.summary_report = summarize_model_results(
            model_results, title=self.title, **kwargs)
        return self 
    
    def add_flex_summary(self, model_results=None, model=None, **kwargs):
        """
        Generates and assigns a flexible summary report for scikit-learn models, 
        especially useful for models optimized using techniques like GridSearchCV 
        or RandomizedSearchCV. This method can directly use a dictionary of model 
        results or extract necessary information from a scikit-learn model object.
    
        Parameters
        ----------
        model_results : dict, optional
            A dictionary containing the model's performance evaluation results, 
            including keys such as 'best_estimator_', 'best_params_', and 
            'cv_results_'. Directly used for generating the summary report if provided.
        model : sklearn estimator, optional
            A scikit-learn model instance, ideally optimized models with attributes 
            like 'best_estimator_', 'best_params_', and optionally 'cv_results_'. 
            Necessary attributes are extracted to generate the summary report if 
            `model_results` is not directly provided.
        **kwargs : dict
            Additional keyword arguments for customizing the summary report generation, 
            passed to the underlying summary generation function.
    
        Returns
        -------
        self : object
            The instance itself, facilitating method chaining.
    
        Raises
        ------
        ValueError
            If neither `model_results` is provided nor `model` with the required 
            attributes ('best_estimator_', 'best_params_', and optionally 'cv_results_').
    
        Examples
        --------
        >>> from sklearn.model_selection import GridSearchCV
        >>> from sklearn.svm import SVC
        >>> from sklearn.datasets import load_iris
        >>> from gofast.api.summary import ModelSummary
        >>> iris = load_iris()
        >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        >>> svc = SVC()
        >>> clf = GridSearchCV(svc, parameters)
        >>> clf.fit(iris.data, iris.target)
        >>> summary = ModelSummary(title="SVC Optimization Summary")
        >>> summary.add_flex_summary(model=clf)
        >>> print(summary.summary_report)
        SVC Optimization Summary
        ====================================================
        Best estimator   : SVC(C=1, kernel='linear')
        Best parameters  : {'C': 1, 'kernel': 'linear'}
        CV               : 5 folds
        Scoring          : accuracy (default)
        ====================================================
        
        This method streamlines the generation of a comprehensive summary report for 
        scikit-learn models, particularly those resulting from hyperparameter optimization 
        processes, enhancing the interpretability and presentation of model performance 
        and tuning results.
        """

        if model_results is None:
            if model and all(hasattr(model, attr) for attr in [
                    'best_estimator_', 'best_params_']):
                model_results = {
                    'best_estimator_': model.best_estimator_,
                    'best_params_': model.best_params_,
                }
                if hasattr (model, 'cv_results_'): 
                    model_results['cv_results_']= model.cv_results_
            else:
                raise ValueError(
                    "Either 'model_results' must be provided or 'model' must have "
                    "'best_estimator_', 'best_params_', and/or 'cv_results_' attributes.")
        
        self.summary_report = summarize_model_results(
            model_results, title=self.title, **kwargs)
        
        return self
        
    def __str__(self):
        """
        Provides the string representation of the summary report.

        Returns
        -------
        str
            The summary report as a formatted string.
        """
        return self.summary_report
    
    def __repr__(self):
        """
        Provides a formal representation indicating whether the summary
        report is populated.

        Returns
        -------
        str
            A message indicating if the ModelSummary contains a report or is empty.
        """
        name= to_camel_case(self.descriptor or "ModelSummary")
        return (f"<{name} object containing results. Use print() to see contents>"
                if self.summary_report else f"<Empty {name}>")

class Summary(FlexDict):
    """
    A utility class for generating detailed summary reports of pandas DataFrames. 
    It inherits from `FlexDict`, enabling attribute-like access to dictionary 
    keys for flexible interaction with summary data. The class provides methods 
    to compile comprehensive insights into the DataFrame, including basic statistics, 
    correlation matrices, counts of unique values in categorical columns, and 
    data samples.

    Attributes
    ----------
    title : str
        An optional title for the summary report. If specified, this title 
        will be included at the top of the generated summary.
    
    descriptor : str, optional
        A dynamic label or descriptor that defines the identity of the output
        when the object is represented as a string. This label is used in the 
        representation of the object in outputs like print statements. If not
        provided, defaults to 'Summary'. 
        
    Methods
    -------
    data_summary(df, include_correlation=False, include_uniques=False, 
                     include_sample=False, sample_size=5):
        Produces a formatted summary of the given DataFrame, optionally 
        including a correlation matrix, unique value counts for categorical 
        columns, and a data sample.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.summary import Summary 
    >>> data = {
    ...     'Age': [25, 30, 35, 40, np.nan],
    ...     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Edward'],
    ...     'Salary': [50000, 60000, 70000, 80000, 90000]
    ... }
    >>> df = pd.DataFrame(data)
    >>> summary = Summary(title="Employee Data Overview")
    >>> summary.basic_statistics(df, include_correlation=True,)

    Notes
    -----
    - The summary report is designed to provide a quick yet comprehensive 
      overview of the DataFrame's content, facilitating initial data exploration 
      and analysis tasks.
    - `FlexDict` is assumed to be a class that allows dictionary items to be 
      accessed as attributes, enhancing ease of use. This behavior enriches the 
      `Summary` class by enabling dynamic attribute assignment based 
      on the provided summary data.
    - External functions `format_dataframe` and others are leveraged for 
      formatting sections of the summary report. These functions must be defined 
      and available in the same scope or imported for the `Summary` class 
      to function correctly.
    - The class focuses on numerical and categorical data within the DataFrame. 
      Other data types (e.g., datetime) are included in the basic statistics but 
      might require specialized handling for more detailed analysis.
    """
    
    def __init__(self, title=None, descriptor =None, **kwargs):
        """
        Initializes the Summary object with an optional title for the 
        report and any additional properties via keyword arguments.

        Parameters
        ----------
        title : str, optional
            The title of the summary report to be generated.
        **kwargs : dict, optional
            Additional keyword arguments passed to the FlexDict constructor.
        """
        super().__init__(**kwargs)
        self.title = title
        self.descriptor= descriptor 
        self.summary_report = ""

    def add_basic_statistics(self, df, include_correlation=False):
        """
        Generates basic statistical measures for the provided DataFrame and,
        optionally, a correlation matrix for numeric columns.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be summarized.
        include_correlation : bool, optional
            If True, includes a correlation matrix for numeric columns in the
            summary. Defaults to False.

        Returns
        -------
        Summary
            The instance itself, allowing for method chaining.

        Raises
        ------
        ValueError
            If `df` is not a pandas DataFrame.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd 
        >>> from gofast.api.summary import Summary
        >>> df = pd.DataFrame({
        ...     'Age': [25, 30, 35, 40, np.nan],
        ...     'Salary': [50000, 60000, 70000, 80000, 90000]
        ... })
        >>> summary = Summary(title="Employee Stats")
        >>> summary.add_basic_statistics(df, include_correlation=True)
        >>> print(summary)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The provided data must be a pandas DataFrame.")
        
        if df.empty: 
            warnings.warn("Empty DataFrame detected. Summary returns No data.")
            self.summary_report="<Empty Summary: Object has No Data>"
            return self 
        
        summary_parts = []
        # Basic statistics
        stats = df.describe(include='all').T.applymap(format_value)
        summary_parts.append(format_dataframe(stats, title="Basic Statistics"))

        # Correlation matrix
        format_corr_str =''
        if include_correlation: 
            # get the maximum table width  
            max_table_width = find_maximum_table_width(summary_parts[0])
            corr_matrix = df.corr().round(2)
            dict_df = df_to_custom_dict(corr_matrix)
            
            format_corr_str= format_report(
                dict_df, report_title = "Correlation Matrix",
                max_table_width=max_table_width )
            
            # noww convert correlation df to text dict with key and 
            #summary_parts.insert (0, format_corr_str ) 
            summary_parts.append (format_corr_str)
        # Compile all parts into a single summary report
        self.summary_report = "\n\n".join(summary_parts)
        
        return self 
    
    def add_unique_counts(
        self, df, include_sample=False, sample_size=5,
        aesthetic_space_allocation=4
        ):
        """
        Generates counts of unique values for categorical columns in the provided
        DataFrame and, optionally, includes a random sample of data.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be summarized.
        include_sample : bool, optional
            If True, adds a random sample of data to the summary. Defaults to False.
        sample_size : int, optional
            The size of the sample to include if `include_sample` is True. Defaults
            to 5.

        Returns
        -------
        Summary
            The instance itself, allowing for method chaining.

        Raises
        ------
        ValueError
            If `df` is not a pandas DataFrame.

        Examples
        --------
        >>> import pandas as pd 
        >>> from gofast.api.summary import Summary
        >>> df = pd.DataFrame({
        ...     'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR'],
        ...     'Age': [28, 34, 45, 32, 41]
        ... })
        >>> summary = Summary(title="Department Overview")
        >>> summary.add_unique_counts(df, include_sample=True)
        >>> print(summary)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if df.empty: 
            warnings.warn("Empty DataFrame detected. Summary returns No data.")
            self.summary_report="<Empty Summary: Object has No Data>"
            return self 
        
        summary_parts={} 
        unique_counts = {col: df[col].nunique() for col in df.select_dtypes(
            include=['object', 'category']).columns}
        if len(unique_counts) ==0: 
            # Empy dataFrame, no categorical is found . 
            raise TypeError("'Unique Counts' expects categorical data. Not found.")
            
        unique_df = pd.DataFrame.from_dict(
            unique_counts, orient='index', columns=['Unique Counts'])
        
        summary_parts = df_to_custom_dict(unique_df, )
        
        # Sample of the data
        if include_sample:
            sample_data = df.sample(n=min(sample_size, len(df)))
            summary_parts ["Sample Data %% : Table "] = sample_data
        
        # Compile all parts into a single summary report
        self.summary_report = format_report(
            summary_parts, report_title ="Unique Counts", max_table_width="auto" )
        
        return self
    
    def summary(self, df, **kwargs):
        """
        Generates a formatted summary of the specified DataFrame and incorporates
        it into a report object held by the class instance.

        This method uses custom formatting settings provided through keyword
        arguments to create a visually appealing summary representation, which
        is then stored as part of the class's state.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to summarize.
        **kwargs : dict
            Additional keyword arguments that are passed to the formatting
            function to customize the summary's appearance.

        Returns
        -------
        Summary
            Returns the instance itself, allowing for method chaining.

        Notes
        -----
        This method is designed to be flexible, accommodating any formatting
        options supported by `format_dataframe`. This allows the user to
        specify how the DataFrame should be displayed in the summary, such as
        adjusting column widths, hiding indices, or adding a title.

        Examples
        --------
        >>> from gofast.api.summary import Summary
        >>> summary = Summary()
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> summary.summary(df, max_text_length=20)
        >>> print(summary)
        =====
          A B
        -----
        0 1 3
        1 2 4
        =====
        """
        self.summary_report = format_dataframe(df, title=self.title, **kwargs)
        return self
    
    def summary2(self, df, **kwargs):
        """
        Similar to `summary`, but uses a different formatting function to process
        the DataFrame summary. This method is intended to provide an alternative
        visual representation of the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to summarize.
        **kwargs : dict
            Additional keyword arguments that are passed to the `format_df`
            function to customize the summary's appearance.

        Returns
        -------
        Summary
            Returns the instance itself, allowing for method chaining.

        Notes
        -----
        The choice between `summary` and `summary2` depends on the specific
        formatting needs and the output style desired. While both methods
        prepare a summary of the DataFrame, `summary2` may involve different
        formatting conventions or settings specific to `format_df`.

        Examples
        --------
        >>> from gofast.api.summary import Summary
        >>> summary = Summary()
        >>> df = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        >>> summary.summary2(df, max_text_length=15,)
        >>> print(summary)
        =========
             C  D
          -------
        0 |  5  7
        1 |  6  8
        =========
        """
        self.summary_report = format_df(df, title=self.title, **kwargs)
        return self
    
    def add_data_corr(
            self, df, min_corr=0.5, high_corr=0.8, use_symbols=False, 
            hide_diag=True, **kwargs):
        """
        Computes and stores a formatted correlation matrix for a DataFrame's numeric
        columns within the class instance. This method leverages formatting options
        to visually represent correlation strengths differently based on specified
        thresholds and conditions.
    
        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame from which the correlation matrix is computed.
        min_corr : float, optional
            The minimum correlation coefficient to display explicitly. Correlation
            values below this threshold will be replaced by a placeholder (which can
            be set via kwargs with 'placeholder'). Default is 0.5.
        high_corr : float, optional
            The threshold above which correlations are considered high and can be
            represented differently if `use_symbols` is True. Default is 0.8.
        use_symbols : bool, optional
            If True, uses symbolic representation ('++', '--', '+-') for correlation
            values, providing a simplified visual cue for strong and moderate
            correlations. Default is False.
        hide_diag : bool, optional
            If True, the diagonal elements of the correlation matrix (always 1) are
            not displayed, helping focus on the non-trivial correlations. Default is
            True.
        **kwargs : dict
            Additional keyword arguments that are passed to `format_correlations`
            function to customize the correlation matrix's appearance.
    
        Returns
        -------
        None
            This method does not return a value; it updates the instance's state by
            setting `summary_report` with the formatted correlation matrix.
    
        Notes
        -----
        This method is part of a class that handles statistical reporting. It allows
        for easy visualization of important correlations within a dataset, helping to
        identify and present statistically significant relationships in a user-friendly
        manner.
    
        Examples
        --------
        >>> import pandas as pd 
        >>> from gofast.api.summary import Summary 
        >>> summary = Summary(title="Correlation Table")
        >>> df = pd.DataFrame({
        ...     'A': np.random.randn(100),
        ...     'B': np.random.randn(100),
        ...     'C': np.random.randn(100) * 10
        ... })
        >>> summary.display_corr(df, min_corr=0.3, high_corr=0.7)
        >>> print(summary)
        Correlation Table      
        =============================
                A       B        C   
          ---------------------------
        A |           0.0430  -0.1115
        B |   0.0430           0.1807
        C |  -0.1115  0.1807         
        =============================
  
        The method updates the `summary_report` attribute of `report` with a formatted
        string representing the correlation matrix, potentially including a title and
        custom formatting as specified by the user.
        """
        self.summary_report = format_correlations(
            df, min_corr=min_corr, high_corr=high_corr, use_symbols= use_symbols, 
                            hide_diag= hide_diag,
                            title = self.title or 'Correlation Table', 
                            **kwargs)
        
        return self 
    
    def __str__(self):
        """
        String representation of the summary report.
        """
        return self.summary_report

    def __repr__(self):
        """
        Formal string representation indicating whether the summary report 
        is empty or populated.
        """
        name= to_camel_case(self.descriptor or "Summary")
        return "<{}: {}>".format(name, 
            "Empty" if not self.summary_report 
            else "Populated. Use print() to see the contents.")

class ReportFactory(FlexDict):
    """
    Represents a dynamic report generator capable of handling and formatting
    various types of data into comprehensive reports. Inherits from `FlexDict`,
    allowing attribute-like access to dictionary keys.

    The `Report` class facilitates the creation of structured text reports from
    mixed data types, recommendations, performance summaries of models, and
    summaries of pandas DataFrames. It provides methods to format each report
    section and compiles them into a single cohesive text representation.

    Attributes
    ----------
    title : str
        The title of the report. If provided, it is included at the top of
        the report output.
    report : dict or str or pandas.DataFrame
        The raw data used to generate the report section last processed.
        This attribute stores the input provided to the last method called
        (e.g., mixed data, recommendations text, model results, or DataFrame).
    report_str : str
        The formatted text representation of the report or the last report
        section processed. This string is ready for printing or logging.
        
    descriptor : str, optional
        A dynamic label or descriptor that defines the identity of the output
        when the object is represented as a string. This label is used in the 
        representation of the object in outputs like print statements. If not
        provided, defaults to 'Report'. 
        
    Methods
    -------
    mixed_types_summary(report, table_width=100):
        Formats a report containing mixed data types into a structured text
        representation.

    add_recommendations(text, key=None, **kwargs):
        Adds a recommendations section to the report, formatted with an
        optional key.

    model_performance_summary(model_results, **kwargs):
        Adds a model performance summary section to the report, using the
        results from model evaluation.

    data_summary(df, **kwargs):
        Adds a summary of a pandas DataFrame to the report, formatted as a
        structured text section.

    Examples
    --------
    >>> from gofast.api.summary import ReportFactory
    >>> report_data = {
    ...     'Total Sales': 123456.789,
    ...     'Average Rating': 4.321,
    ...     'Number of Reviews': 987
    ... }
    >>> report = ReportFactory(title="Sales Summary")
    >>> report.mixed_types_summary(report_data)
    >>> print(report)
    ================================
             Sales Summary          
    --------------------------------
    Total Sales       : 123456.7890
    Average Rating    : 4.3210
    Number of Reviews : 987
    ================================
    
    Notes
    -----
    - This class depends on external formatting functions like `format_report`,
      `format_text`, `summarize_model_results`, and `format_dataframe`, which
      need to be defined in the same scope or imported.
    - The `FlexDict` parent class is assumed to provide dynamic attribute access,
      allowing for flexible interaction with report data.
    - While `report_str` holds the formatted report ready for display, `report`
      maintains the raw input data for reference or further processing. 
    """
    def __init__(self, title=None, descriptor=None,  **kwargs):
        """
        Initializes the Report object with an optional title and additional
        keyword arguments passed to the FlexDict constructor.

        Parameters
        ----------
        title : str, optional
            The title of the report. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the FlexDict constructor,
            allowing for dynamic attribute assignment based on the provided
            dictionary.
        """
        super().__init__(**kwargs)
        self.title = title
        self.descriptor= descriptor
        self.report = None
        self.report_str = None

    def add_mixed_types(self, report, table_width='auto',
                            include_colon= True, pad_colon= True, 
                            **kwargs):
        """
        Formats a report containing mixed data types.

        Parameters:
        - report (dict): The report data.
        - table_width (int, optional): The maximum width of the table.
          Defaults to 100.
        - kwargs: dict, keyword arguments of :func:`format_dataframe`. 
        
        """
        self.report = report
        self.report_str = format_report(
            report_data=report, report_title=self.title,
            max_table_width=table_width, include_colon= include_colon, 
            pad_colon= pad_colon, **kwargs)
        return self 

    def add_recommendations(
            self, texts, keys=None, key_length=15, max_char_text=70, **kwargs):
        """
        Formats and adds a recommendations section to the report.
    
        Parameters:
        - texts : str, list of str, or dict
            The text(s) containing recommendations. Can be a single string, 
            a list of strings, or a dictionary with keys as headings 
            and values as text.
        - keys : str or list of str, optional
            An optional key or list of keys to prefix the text(s). Used only 
            if texts is a string or list of strings. Defaults to None.
        - key_length : int, optional
            Maximum key length for formatting before the ':'. Defaults to 15.
        - max_char_text : int, optional
            Number of text characters to tolerate before wrapping to a new line.
            Defaults to 70.
        """
        formatted_texts = []
        if isinstance(texts, dict):
            # If texts is a dictionary, use it directly.
            report_data = texts
        else:
            # If texts is not a dictionary, combine texts 
            # with keys into a dictionary.
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(keys, str):
                keys = [keys]
            # Ensure keys are provided for each text if texts is a list.
            if not keys or len(keys) < len(texts):
                keys = [f"Key{i+1}" for i in range(len(texts))]
            report_data = dict(zip(keys, texts))
    
        # Format each text entry with its key.
        for key, text in report_data.items():
            formatted_text = format_text(text, key, key_length=key_length,
                                         max_char_text=max_char_text, **kwargs)
            formatted_texts.append(formatted_text)
    
        # Calculate the total width for the top and bottom bars.
        max_width = max(len(ft.split('\n')[0]) for ft in formatted_texts
                        ) if formatted_texts else 0
        top_bottom_bar = "=" * max_width
    
        # Compile the formatted texts into the report string.
        title = str(self.title).center(max_width) +'\n' if self.title is not None else ''
        self.report_str = f"{title}{top_bottom_bar}\n" + "\n".join(
            formatted_texts) + f"\n{top_bottom_bar}"
    
        # Assign report data for reference.
        self.report = report_data
    
        return self

    def add_data(self, df, **kwargs):
        """
        Formats and adds a data frame summary to the report.

        Parameters:
        - df (pandas.DataFrame): The data frame to summarize.
        """
        self.report= df 
        self.report_str = format_dataframe(df, title=self.title, **kwargs)
        
        return self 
    
    def add_contents( self, contents, title=None, header=None, **kwargs ): 
        
        self.report = contents 
        self.report_str = summarize_inline_table(
            contents, title =title, header=header, **kwargs)
        
        return self 
    
    def __str__(self):
        """
        String representation of the report content.
        """
        return self.report_str or "<No report>"

    def __repr__(self):
        """
        Formal string representation of the Report object.
        """
        name =to_camel_case(self.descriptor or "Report")
        return ( f"<{name}: Print() to see the content>" 
                if self.report_str is not None 
                else f"<{name}: No content>" )
  
def assemble_reports(
    *reports, base_class=ReportFactory, 
    validation_attr='report_str', 
    display=False
    ):
    """
    Assembles multiple report strings from instances of the specified base class
    into a single formatted string. This function is especially useful for creating
    a comprehensive view from several smaller reports.

    Parameters:
    reports : tuple
        Variable number of report instances to be combined.
    base_class : type, optional
        The class type that each report instance is expected to derive from, used
        to ensure the correct type of reports are being processed.
    validation_attr : str, optional
        The attribute from each instance used for assembling the reports.
    display : bool, optional
        If True, the combined report string is printed.

    Returns:
    str
        A single string containing the concatenated content of all report instances.
    """
    #print([type())
    if not all(isinstance_(report, base_class) for report in reports):
        raise ValueError("All reports must be instances of " + str(base_class))

    combined_report = ""
    for i, report in enumerate(reports):
        if not hasattr(report, validation_attr):
            raise ValueError(f"Report at position {i} does not have the"
                             f" required attribute '{validation_attr}'.")

        # Extract the report content, assuming it's stored under `report_str` attribute
        report_content = getattr(report, validation_attr)
        # Strip the top and bottom boundary if it's not the first report
        if i > 0:
            report_content = "\n".join(report_content.split('\n')[1:])
        combined_report += report_content + "\n"

    if display:
        print(combined_report)

    return combined_report

def ensure_list_with_defaults(input_value, target_length, default=None):
    """
    Ensures that the input value is a list of a specific length, padding with 
    defaults or trimming as necessary.

    Parameters
    ----------
    input_value : str or list
        The input value to be converted into a list. If a string is provided, 
        it will be converted into a single-element list.
    target_length : int
        The desired length of the list.
    default : optional
        The default value to use for padding if the list is shorter than the 
        target length. Defaults to None.

    Returns
    -------
    list
        A list of the target length, padded or trimmed as necessary.
    """
    if isinstance(input_value, str):
        input_value = [input_value]
    elif not input_value:  # Handles None or empty input
        input_value = []
    # Check if target_length is a collection and get its length if so
    if hasattr(target_length, '__len__'):
        target_length = len(target_length)
    
    # Ensure target_length is an integer
    if isinstance(target_length, float):
        target_length = int(target_length)
    
    # Validate target_length is an integer and greater than zero
    if not isinstance(target_length, int) or target_length <= 0:
        raise ValueError("target_length must be a positive integer.")
    
    # Ensure the list is of the target length, padding with defaults if necessary
    return (input_value + [default] * target_length)[:target_length]

def summarize_tables(*contents, titles=None, headers=None, **kwargs):
    """
    Generates a summarized string representation of multiple tables, optionally 
    including titles and headers for each. Tables are first formatted individually 
    and then normalized to have uniform widths. Additional formatting options 
    can be passed through keyword arguments.

    Each table can be represented as a dictionary, a list of dictionaries, or 
    a nested dictionary structure. The function supports extracting model or 
    estimator names from the tables to use as headers when provided.

    Parameters
    ----------
    *contents : dict or list of dict
        The table(s) to be summarized. Each table can be directly provided as 
        a dictionary, a list of dictionaries for multiple tables, or a nested 
        dictionary where each key is considered a table name.
    titles : list of str, optional
        The titles for each table or group of tables. If not provided, titles 
        will be omitted.
    headers : list of str, optional
        The headers for each table. If not provided, headers will be derived 
        from the content if possible.
    **kwargs : dict
        Additional keyword arguments to be passed to the `summarize_inline_table`
        function for table formatting.

    Returns
    -------
    str
        A string representation of the summarized tables with normalized widths.

    Examples
    --------
    >>> contents = [{
    ...     "Estimator": {"Accuracy": 0.95, "Precision": 0.89, "Recall": 0.93},
    ...     "RandomForest": {"Accuracy": 0.97, "Precision": 0.91, "Recall": 0.95}
    ... }]
    >>> titles = ["Model Comparison"]
    >>> print(summarize_tables(*contents, titles=titles))
    Model Comparison
    ========================
             Estimator       
    ------------------------
      Accuracy   : 0.9500
      Precision  : 0.8900
      Recall     : 0.9300
    ========================
          RandomForest       
    ------------------------
      Accuracy   : 0.9700
      Precision  : 0.9100
      Recall     : 0.9500
    ========================
    
    This function streamlines the process of formatting and presenting multiple 
    tables of data, especially useful for comparing model performances or similar 
    datasets. It leverages the `summarize_inline_table` for individual table 
    formatting and `normalize_table_widths` for ensuring uniform table widths 
    across the summary.
    """

    def format_table(content, title=None, header=None):
        return summarize_inline_table(content, title=title, header=header, **kwargs)

    summaries = []
    
    # Manage default headers and titles   
    headers = ensure_list_with_defaults ( headers, len( contents) )
    titles = ensure_list_with_defaults ( titles, len( contents) )
    
    for content, title, header in zip(contents, titles, headers ):
        if isinstance(content, dict) and all(isinstance(val, dict) for val in content.values()):
            # Handling nested dictionary structure where each key
            # is considered a model/estimator name
            for model_name, table in content.items():
                summaries.append(format_table(table, title, model_name))
            
        elif isinstance(content, list) and all(isinstance(item, dict) for item in content):
            # If the content is a list of tables
            for table in content:
                model_name, table = extract_model_name_and_dict(
                    table, candidate_names=['Estimator', 'Model'])
                model_name = model_name or header
                summaries.append(format_table(table, title, model_name))
                
        elif isinstance(content, dict):
            # If the content directly represents a table
            summaries.append(format_table(content, title, header))
                
    summaries = "\n".join(summaries)
    # now adjust table 
    return normalize_table_widths(summaries)

def normalize_table_widths(
    contents, 
    max_width='auto', 
    header_marker='=', 
    center_table_contents=False
 ):
    """
    Adjusts the widths of table representations in a given string to a uniform
    maximum width, optionally centering the content of each table.

    This function scans through the input string, identifying tables by their
    header and separator lines. It then ensures that each table matches the 
    maximum width determined either automatically from the widest table or 
    as specified. If center_table_contents is True, it also centers the 
    content of each table within the adjusted width.

    Parameters
    ----------
    contents : str
        The string containing the table(s) to be normalized.
    max_width : int or 'auto', optional
        The maximum width to which the table widths should be adjusted.
        If 'auto', the width is determined based on the widest table
        in the input string. Defaults to 'auto'.
    header_marker : str, optional
        The character used for header and separator lines in the table(s).
        Defaults to '='.
    center_table_contents : bool, optional
        Whether to center the content of each table within the maximum
        width. Defaults to False.

    Returns
    -------
    str
        The input string with table widths normalized and content optionally
        centered.

    Examples
    --------
    >>> from gofast.api.summary import normalize_table_widths
    >>> input_str = '''\
    ... ====================
    ...         SVC       
    ... --------------------
    ... Accuracy   : 0.9500
    ... Precision  : 0.8900
    ... Recall     : 0.9300
    ... ====================
    ... =======================
    ...     RandomForest       
    ... -----------------------
    ... Accuracy   : 0.9500
    ... Precision  : 0.8900
    ... Recall     : 0.9300
    ... ========================'''

    >>> print(normalize_table_widths(input_str))
    ========================
    SVC       
    ------------------------
    Accuracy   : 0.9500
    Precision  : 0.8900
    Recall     : 0.9300
    ========================
    ========================
    RandomForest       
    ------------------------
    Accuracy   : 0.9500
    Precision  : 0.8900
    Recall     : 0.9300
    ========================

    >>> print(normalize_table_widths(input_str, center_table_contents=True))
    ========================
             SVC       
    ------------------------
      Accuracy   : 0.9500
      Precision  : 0.8900
      Recall     : 0.9300
    ========================
    ========================
       RandomForest       
    ------------------------
      Accuracy   : 0.9500
      Precision  : 0.8900
      Recall     : 0.9300
    ========================
    
    Note that the function assumes tables are separated by header and separator
    lines using the specified `header_marker` or other line-starting characters
    like '-', '~', or '='. Content lines not starting with these characters are
    considered part of the tables' content and are adjusted or centered based
    on the `center_table_contents` parameter.
    """

    # Convert contents to string if not already
    contents = str(contents)
    # Find the maximum width of the table headers if max_width is set to auto
    if max_width == 'auto':
        max_width = find_maximum_table_width(contents, header_marker)

    # Split the contents into lines for processing
    lines = contents.split('\n')

    # Initialize variables to store the processed lines and current
    # table's lines for centering purposes
    normalized_lines = []
    current_table_lines = []

    def process_current_table():
        """Adjusts the width of the current table's lines and centers them if required."""
        if center_table_contents:
            # Center each line in the current table
            for i, line in enumerate(current_table_lines):
                if line.strip():  # Ignore empty lines
                    current_table_lines[i] = line.center(max_width)
        normalized_lines.extend(current_table_lines)
        current_table_lines.clear()

    for line in lines:
        # Check if the line is empty or consists only of whitespace;
        # if so, continue to the next iteration
        if not line.strip():
            continue
        # Check if the line is a header or separator line by looking for marker 
        # characters at the start of the line
    
        if line.startswith(header_marker) or  line[0] in '-~=':
            # If starting a new table, process the previous table's lines
            if current_table_lines:
                process_current_table()
            # Adjust the header or separator line to the maximum width
            normalized_lines.append(line[0] * max_width)
        else:
            # Add non-header lines to the current table's lines for potential centering
            current_table_lines.append(line)

    # Process any remaining table lines after the loop
    if current_table_lines:
        process_current_table()

    # Join the processed lines back into a single string
    return '\n'.join(normalized_lines)


def extract_model_name_and_dict(model_dict, candidate_names=None):
    """
    Extracts the model name from a dictionary based on candidate keys and 
    returns the name and the updated dictionary.
    
    This function searches through the dictionary keys for any that match a 
    list of candidate names (e.g., 'estimator', 'model'), intended to likely 
    represent the model name. When a match is found, it removes the key-value 
    pair from the dictionary and returns the model name along with the modified
    dictionary.

    Parameters
    ----------
    model_dict : dict
        The dictionary from which to extract the model name. This dictionary 
        should potentially contain one of the candidate names.
    candidate_names : list of str or str, optional
        A list of strings or a single string representing the likely keys that 
        would hold the model name within the dictionary.
        Defaults to ['estimator', 'model'] if None is provided.

    Returns
    -------
    tuple
        A tuple containing the model name (str) if found, otherwise None, and 
        the updated dictionary (dict) with the model name key removed if found.

    Raises
    ------
    ValueError
        If the provided `model_dict` parameter is not a dictionary.

    Examples
    --------
    >>> from gofast.api.summary import extract_model_name_and_dict
    >>> model_info = {"estimator": "SVC", "accuracy": 0.95}
    >>> model_name, updated_dict = extract_model_name_and_dict(model_info)
    >>> model_name
    'SVC'
    >>> updated_dict
    {'accuracy': 0.95}
    """
    if not isinstance(model_dict, dict):
        raise ValueError("The model_dict parameter must be a dictionary.")

    # Ensure candidate_names is a list, even if a single string is provided
    candidate_names = candidate_names or ['estimator', 'model']
    candidate_names = [candidate_names] if isinstance(candidate_names, str) else candidate_names
    # Use list to avoid RuntimeError for modifying dict during iteration
    for key in list(model_dict.keys()):  
        if key.lower() in (name.lower() for name in candidate_names):
            # Extract and return the model name, and the updated dictionary
            model_name = model_dict.pop(key)
            return model_name, model_dict

    # If no model name is found, return None and the original dictionary
    return None, model_dict
    
def summarize_inline_table(
    contents, 
    title=None, 
    header=None, 
    max_width='auto',
    top_line='=', 
    sub_line='-', 
    bottom_line='='
    ):
    """
    Creates a string representation of a table summarizing the provided 
    contents, with optional title and header, and customizable table aesthetics.

    Parameters
    ----------
    contents : dict
        A dictionary containing the data to be summarized in the table. Keys
        represent the labels, and values are the corresponding data.
    title : str, optional
        A title for the table, centered above the table content. If None,
        no title is displayed.
    header : str, optional
        A header for the table, displayed below the title (if present) and
        above the table content. If None, no header is displayed.
    max_width : 'auto', int, optional
        The maximum width of the table. If 'auto', the width is adjusted
        based on the content. If an integer is provided, it specifies the
        maximum width; contents may be truncated to fit. Defaults to 'auto'.
    top_line : str, optional
        The character used to create the top border of the table. Defaults
        to '='.
    sub_line : str, optional
        The character used to create the line below the header and above the
        bottom border. Defaults to '-'.
    bottom_line : str, optional
        The character used to create the bottom border of the table. Defaults
        to '='.

    Returns
    -------
    str
        The formatted table as a string.

    Raises
    ------
    ValueError
        If `contents` is not a dictionary.

    Examples
    --------
    >>> from gofast.api.summary import summarize_inline_table
    >>> contents = {
    ...     "Estimator": "SVC",
    ...     "Accuracy": 0.95,
    ...     "Precision": 0.89,
    ...     "Recall": 0.93
    ... }
    >>> print(summarize_inline_table(contents, title="Model Performance", header="Metrics"))
     Model Performance  
    ====================
          Metrics       
    --------------------
    Estimator  : SVC
    Accuracy   : 0.9500
    Precision  : 0.8900
    Recall     : 0.9300
    ====================

    Notes
    -----
    - Numeric values in `contents` are formatted to four decimal places.
    - If the `max_width` is exceeded by a value, the value is truncated with '...'
      appended to indicate truncation.
    - The table width is dynamically determined based on the longest key-value pair
      or set to `max_width` if provided. Adjustments ensure the table's presentation
      is both aesthetic and functional.
    """
    if not isinstance(contents, dict):
        raise ValueError("summarize_inline_table expects a dict of keys and values.")
    
    # Helper function to format values
    # Initial calculations for formatting
    max_key_length = max(len(key) for key in contents) + 1  # +1 for the space after keys
    max_value_length = max(len(format_value(value)) for value in contents.values())
    
    # Adjust table width if 'max_width' is 'auto' or specified as a number
    if max_width == 'auto':
        table_width = max_key_length + max_value_length + 4  # +4 for " : " and extra space
    elif isinstance(max_width, (float, int)):
        table_width = int(max_width)
    else:
        table_width = max_key_length + max_value_length + 4  # Default behavior

    # Title and header
    title_str = f"{title.center(table_width)}" if title else ""
    top_line_str = top_line * table_width
    header_str = f"{header.center(table_width)}" if header else ""
    sub_line_str = sub_line * table_width if header else ""
    
    # Constructing content lines
    content_lines = []
    for key, value in contents.items():
        formatted_value = format_value(value)
        # Truncate long values if necessary
        space_for_value = table_width - max_key_length - 3
        if len(formatted_value) > space_for_value:
            formatted_value = formatted_value[:space_for_value-3] + "..."
    
        key_str = f"{key}"
        line = f"{key_str.ljust(max_key_length)} : {formatted_value}"
        content_lines.append(line)
    
    content_str = "\n".join(content_lines)
    bottom_line_str = bottom_line * table_width
    # Combine all parts
    if header: 
        rows = [title_str, top_line_str, header_str, sub_line_str,
                content_str, bottom_line_str]
    else: 
        rows = [title_str, top_line_str, content_str, bottom_line_str]
    table = "\n".join(rows)
    
    return table

def get_table_width(
    data, include_colon_space=True, 
    max_column_width=100, 
    include_index=True
    ):
    """
    Calculate the maximum width required for displaying a table constructed
    from a dictionary or pandas DataFrame.

    Parameters
    ----------
    data : dict or pandas.DataFrame
        The data to calculate the table width for.
    include_colon_space : bool, optional
        Whether to include extra space for ': ' in the calculation, applicable
        only if `data` is a dictionary. Defaults to True.
    max_column_width : int, optional
        The maximum allowed width for any text or value before truncation
        with '...'. Defaults to 100.
    include_index : bool, optional
        If `data` is a pandas DataFrame, this determines whether the index
        column width should be included in the total width calculation.
        Defaults to True.

    Returns
    -------
    int
        The calculated width of the table necessary to display the data
        without exceeding `max_column_width` for any single column.

    Raises
    ------
    ValueError
        If `data` is neither a dictionary nor a pandas DataFrame.
    Examples 
    -------
    >>> import pandas as pd 
    >>> from gofast.api.summary import get_table_width
    >>> report_data = {
    ...    "Estimator": "SVC",
    ...    "Best parameters": "{C: 1, gamma=0.1}",
    ...    "Accuracy": 0.95,
    ... }
    >>> print("Dictionary Table Width:", get_table_width(report_data))
    Dictionary Table Width: 35
    >>> # For a DataFrame
    >>> df_data = pd.DataFrame({
    ...    "Feature": ["Feature1", "Feature2", (
        "A very long feature name that exceeds max column width")],
    ...    "Importance": [0.1, 0.2, 0.3]
    ... })
    >>> print("DataFrame Table Width:", get_table_width(df_data))
    DataFrame Table Width: 69
    """
    def _format_value(value):
        """Format the value to a string, truncating if necessary."""
        value_str = str(value)
        value_str= format_value(value_str) # format numeric .4f 
        return value_str if len(value_str) <= max_column_width else value_str[
            :max_column_width - 3] + "..."
    
    if not isinstance(data, (dict, pd.DataFrame)):
        raise ValueError("Data must be either a dictionary or a pandas DataFrame.")
    if len(data)==0: 
        raise ValueError ("Empty data is not allowed.")
        
    if isinstance(data, dict):
        max_key_length = max(len(key) for key in data.keys())
        max_value_length = max(len(_format_value(value)) for value in data.values())
        colon_space = 3 if include_colon_space else 0  # Accounting for " : "
        table_width = max_key_length + max_value_length + colon_space
        
    else:  # pandas DataFrame
        max_index_length = max(len(str(index)) for index in data.index) if include_index else 0
        max_col_name_length = max(len(col) for col in data.columns)
        max_value_length = data.applymap(_format_value).astype(str).applymap(len).values.max()
        # Accounting for spaces and colon
        table_width = max_index_length + max_col_name_length + max_value_length + 4  
    
    return table_width

def calculate_maximum_length( report_data, max_table_length = "auto" ): 
    """ Calculate the maximum key length for alignment"""
    max_key_length = max(len(key) for key in report_data.keys())
    # calculate the maximum values length 
    max_val_length = 0 
    for value in report_data.values (): 
        value = format_cls_obj(value)
        if isinstance ( value, (int, float, np.integer, np.floating)): 
            value = f"{value}" if isinstance ( value, int) else  f"{float(value):.4f}" 
        if isinstance ( value, pd.Series): 
            value = format_series ( value)
        if max_val_length < len(value): 
            max_val_length = len(value) 
    if str(max_table_length).lower() in ['none', 'auto']:  
       max_table_length = max_key_length + max_val_length +4  # @ 4 for spaces 
    else: 
        if (max_key_length + max_val_length) >=max_table_length:  
            max_val_length = max_table_length - max_key_length -4
        
    return max_key_length, max_val_length, max_table_length

def format_report(report_data, report_title=None, max_table_width= 70, **kws ):
    """
    Formats the provided report data into a structured text report, 
    accommodating various data types including numbers, strings, pandas Series,
    and pandas DataFrames. The report is formatted with a title (if provided),
    and each key-value pair from the report data dictionary is listed with
    proper alignment and formatting.

    Parameters
    ----------
    report_data : dict
        Dictionary containing the data to be included in the report. Keys
        represent the data labels, and values are the corresponding data
        points, which can be of various data types.
    report_title : str, optional
        A title for the report. If provided, it is centered at the top of
        the report above a top line and a subsection line.
    max_table_width: int, default=70
        The maximum line width, with '...' appended to indicate truncation. 
    **kws: dict, keyword argument for `format_dataframe` function. 
    Returns
    -------
    str
        A string representation of the formatted report including the top
        line, title (if provided), each data label with its formatted value,
        and a bottom line.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.summary import format_report
    >>> report_data = {
    ...     'Total Sales': 123456.789,
    ...     'Average Rating': 4.321,
    ...     'Number of Reviews': 987,
    ...     'Sales Data': pd.Series([100, 200, 300], index=['Jan', 'Feb', 'Mar'])
    ... }
    >>> print(format_report(report_data, report_title='Sales Summary'))
                         Sales Summary
    =============================================================
    Total Sales          : 123456.7890
    Average Rating       : 4.3210
    Number of Reviews    : 987
    Sales Data           : Series ~ shape=<3> - mean: 200.0000...
    =============================================================

    Notes
    -----
    - Numeric values are formatted to four decimal places.
    - For pandas Series and DataFrame values, a brief summary is included instead
      of the full data to keep the report concise. The implementation of how
      Series and DataFrame summaries are formatted depends on the `format_series`
      and `dataframe_key_format` helper functions.
    - Long text values are truncated to fit within the maximum line width, with
      '...' appended to indicate truncation. The maximum line width is dynamically
      determined based on the longest key or value, with an option to specify a
      maximum width manually.
    """
    max_key_length, max_val_length, max_table_width = calculate_maximum_length(
        report_data, max_table_width)
    
    # Adjusting line length based on key-value spacing and aesthetics
    line_length = max(max_key_length + max_val_length + 4,
                      len(report_title) if report_title else 0, max_table_width)
    top_line = "=" * line_length
    bottom_line = "=" * line_length
    
    # Constructing report string with title and frame lines
    report_lines = [top_line]
    if report_title:
        report_lines.append(report_title.center(line_length))
        report_lines.append("-" * line_length)
    
    # Processing each key-value pair in the report data
    for key, value in report_data.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            formatted_value = format_value(value)
        elif isinstance(value, (list, tuple)):
            formatted_value = format_list(list(value))
        elif isinstance(value, np.ndarray):
            formatted_value = format_array(value)
        elif isinstance(value, pd.Series):
            formatted_value = format_series(value)
        elif isinstance(value, dict):
            formatted_value = format_dict(value)
        elif isinstance(value, pd.DataFrame):
            # DataFrame formatting using specific function to handle
            # columns and alignment
            formatted_value = key_dataframe_format(
                key, value, 
                max_key_length=max_key_length, 
                max_text_char=max_val_length, 
                **kws
                )
        elif isinstance_(value, (DataFrameFormatter, DescriptionFormatter,
                                 MultiFrameFormatter)): 
            formatted_value = key_formatter_format(
                key, value, 
                max_key_length=max_key_length, 
                max_text_char=max_val_length, 
                **kws
                )
        else:  # Treat other types as text, including strings
            formatted_value = str(format_cls_obj(value))
        
        # For DataFrame or formatter object, the formatting 
        # is already handled above
        if not isinstance_(value, (pd.DataFrame, DataFrameFormatter, 
                                   DescriptionFormatter, MultiFrameFormatter) 
                           ):
            # Formatting non-DataFrame values with key alignment
            report_lines.append(format_text(
                formatted_value, 
                key=key, 
                key_length=max_key_length, 
                max_char_text=line_length
                )
            )
        else:
            # Directly appending pre-formatted DataFrame string
            report_lines.append(formatted_value)
    
    # Finalizing report string with bottom frame line
    report_lines.append(bottom_line)
    
    return "\n".join(report_lines)

def format_cls_obj(obj):
    """
    Returns the class name of the given object if it's an instance of a class.
    If the object itself is a class, it returns the name of the class.
    Otherwise, it returns the object as is.

    This function is useful for getting a string representation of an object's
    class for display or logging purposes.

    Parameters
    ----------
    obj : Any
        The object whose class name is to be retrieved. It can be an instance of
        any class, including built-in types, or a class object itself.

    Returns
    -------
    str or Any
        The name of the class as a string if `obj` is a class instance or a class itself.
        If the object does not have a class name (unlikely in practical scenarios),
        returns the object unchanged.

    Examples
    --------
    >>> class MyClass: pass
    ...
    >>> instance = MyClass()
    >>> format_cls_obj(instance)
    'MyClass'
    >>> format_cls_obj(MyClass)
    'MyClass'
    >>> format_cls_obj(42)
    42
    """
    # If `obj` is a class instance, return its class name.
    if hasattr(obj, '__class__') and hasattr (obj, '__name__'):
        return obj.__class__.__name__
    # If `obj` itself is a type (a class), return 'type', as it is the class of all classes.
    elif isinstance(obj, type):
        return obj.__name__
    # Otherwise, return the object itself.
    return obj

def summarize_model_results(
    model_results, 
    title=None, 
    max_width=None,
    top_line='=', 
    sub_line='-', 
    bottom_line='=', 
    max_col_lengths=None, 
    ):
    """
    Summarizes the results of model tuning, including the best estimator, best 
    parameters, and cross-validation results, formatted as a string 
    representation of tables.

    Parameters
    ----------
    model_results : dict
        A dictionary containing the model's tuning results, potentially including keys
        for the best estimator, best parameters, and cross-validation results.
    title : str, optional
        The title of the summary report. Defaults to "Model Results".
    max_width: maximum columns/text width/length before the truncation. 
    
    top_line : str, optional
        The character used for the top border of the tables. Defaults to '='.
    sub_line : str, optional
        The character used for the sub-headers within the tables. Defaults to '-'.
    bottom_line : str, optional
        The character used for the bottom border of the tables. Defaults to '='.

    Returns
    -------
    str
        A formatted string that summarizes the model results, including tables for
        the best estimator and parameters as well as detailed cross-validation results.

    Raises
    ------
    ValueError
        If 'best_estimator_' or 'best_parameters_' keys are missing in the provided
        `model_results` dictionary.

    Examples
    --------
    >>> from gofast.api.summary import summarize_model_results
    >>> model_results = {
    ...    'best_parameters_': {'C': 1, 'gamma': 0.1},
    ...    'best_estimator_': "SVC",
    ...    'cv_results_': {
    ...        'split0_test_score': [0.6789, 0.8],
    ...        'split1_test_score': [0.5678, 0.9],
    ...        'split2_test_score': [0.9807, 0.95],
    ...        'split3_test_score': [0.8541, 0.85],
    ...        'params': [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01}],
    ...    },
    ...   'scoring': 'accuracy',
    ... }
    >>> sum_model = summarize_model_results( model_results ) 
    >>> print(sum_model)
                   Model Results               
    ===========================================
    Best estimator       : SVC
    Best parameters      : {'C': 1, 'gamma':...
    Scoring              : accuracy
    nCV                  : 4
    Params combinations  : 2
    ===========================================

                   Tuning Results              
    ===========================================
          Params   Mean CV Score  Std. CV Score
    -------------------------------------------
    0    (1, 0.1)         0.7704         0.1586
    1  (10, 0.01)         0.8750         0.0559
    ===========================================

    Notes
    -----
    - This function requires the presence of `standardize_keys`, 
      `prepare_cv_results_dataframe`, `get_table_width`, and `format_dataframe` 
      helper functions for processing the model results and formatting them 
      into tables.
    - The function dynamically adjusts the width of the cross-validation 
      results table based on the content of the inline summary to ensure 
      consistent presentation.
    """
    title = title or "Model Results"
    # make the inline model results 
    inline_contents, standardized_results= summarize_inline_model_results(
        model_results
    )
    # compute the max keys 
    if max_width is None: 
        max_width = get_table_width(inline_contents, max_column_width=100)
    # Preparing data for the CV results DataFrame
    formatted_table=''
 
    if 'cv_results_' in standardized_results:
        if not any ('split' in cv_key for cv_key in standardized_results['cv_results_']):
            warnings.warn(
                "Invalid CV data structure. The 'cv_results_' from scikit-learn"
                " requires the inclusion of 'splitxx_' keys.")
        df = prepare_cv_results_dataframe(standardized_results['cv_results_'])
        if not df.empty: 
            # Formatting CV results DataFrame
            formatted_table = format_dataframe(
                df, title="Tuning Results (*=score)",
                max_width=max_width, 
                top_line=top_line, sub_line=sub_line, 
                bottom_line=bottom_line, 
                max_col_lengths= max_col_lengths , 
            )
            max_width = len(formatted_table.split('\n')[0]) #
        # Combining inline content and formatted table
    summary_inline_tab = summarize_inline_table(
        inline_contents, title=title, max_width=max_width, 
        top_line=top_line, sub_line=sub_line, bottom_line=bottom_line
    )
    formatted_table=f'\n\n{formatted_table}' if 'cv_results_' in standardized_results else ''
    summary = f"{summary_inline_tab}{formatted_table}"
    return summary

def get_max_col_lengths ( model_results_list, max_text_length=50 ): 
    """ Recompute the maximum column length to let the length of all columns 
    be consistent with all dataframes."""
    # make the inline model results 
    stand_results = [ ]
    for model_result in model_results_list: 
        inline_contents, standardized_results= summarize_inline_model_results(
            model_result
        )
        stand_results.append ( standardized_results)
    
    combined_df=[] 
    for s_result in stand_results: 
        if 'cv_results_' in s_result:
            df = prepare_cv_results_dataframe(s_result['cv_results_'])
            combined_df.append ( df)
    if combined_df: 
        combined_df = pd.concat (combined_df, axis=0)
        
    max_col_lengths = {
        col: max(len(col), max(df[col].astype(str).apply(
            lambda x: len(x) if len(x) <= max_text_length else max_text_length + 3)))
        for col in combined_df.columns
    }
    
    # update max_with 
    max_width = get_table_width(combined_df, max_column_width=100) # default
    return max_col_lengths, max_width 


def summarize_inline_model_results(model_results ):
    """
    Summarizes and standardizes the results of a model fitting process into a 
    consistent inline format for easy reporting or further analysis.

    Parameters
    ----------
    model_results : dict
        A dictionary containing various keys that represent results from 
        model fitting processes, such as estimators, parameters, scores, etc.

    Returns
    -------
    tuple of dict
        A tuple containing two dictionaries:
        - The first dictionary includes standardized key names and associated 
          results, featuring information about the best estimator, best 
          parameters, scores, scoring method, number of cross-validation splits,
          and parameter combinations.
        - The second dictionary is the standardized results that have been 
          processed through the `standardize_keys` function, providing a direct
          view of the standardized keys and their corresponding values from the
          original model_results.

    Raises
    ------
    ValueError
        If any of the essential information (best estimator or best parameters)
        is missing from the provided `model_results`.

    Examples
    --------
    >>> from gofast.api.summary import summarize_inline_model_results
    >>> model_results = {
    ...     'model': "RandomForest",
    ...     'params': {'n_estimators': 100, 'max_depth': 5},
    ...     'fold_results': {'mean_test_score': [0.9, 0.95, 0.85]},
    ...     'scoring': 'accuracy'
    ... }
    >>> inline_results, standardized_results = summarize_inline_model_results(
    ... model_results)
    >>> print(inline_results)
    {
        'Best estimator': "RandomForest",
        'Best parameters': {'n_estimators': 100, 'max_depth': 5},
        'Best score': 'Undefined',
        'Scoring': 'accuracy',
        'nCV': 0,
    }
    >>> print(standardized_results['best_estimator_'])
    'RandomForest'

    Notes
    -----
    - The function leverages the `standardize_keys` function to map various
      potentially varying terminologies into a set of predefined keys.
    - This function is designed to work dynamically with any set of results as
      long as they can be mapped using predefined patterns. It is highly
      dependent on the accuracy and robustness of the `standardize_keys` function.
    - The function returns 'Unknown estimator' or 'Unknown parameters' if the 
      necessary details are missing from the input results. This is intended 
      to prevent errors in subsequent processes that use this summary.
    """
    # Standardize keys in model_results
    standardized_results = standardize_keys(model_results)
    # Validate presence of required information
    required_keys = ['best_estimator_', 'best_params_']
    if not any(key in standardized_results for key in required_keys):
        missing_keys = ', '.join(required_keys)
        raise ValueError(f"Required information ({missing_keys}) is missing.")

    # Prepare inline contents
    inline_contents = {
        "Best estimator": fetch_estimator_name(standardized_results.get(
        'best_estimator_', '<Unknown>')),
        "Best parameters": standardized_results.get('best_params_', '<Undefined>')
    }

    # Optionally add scores and scoring method if available
    if 'best_score_' in standardized_results:
        inline_contents["Best score"] = standardized_results['best_score_']

    if 'scoring' in standardized_results:
        inline_contents["Scoring"] = standardized_results['scoring']

    # Compute the number of cross-validation splits
    cv_results = standardized_results.get('cv_results_', {})
    inline_contents["nCV"] = len({k for k in cv_results if k.startswith('split')})

    # Add number of parameter combinations if available
    if 'params' in cv_results:
        inline_contents["Params combinations"] = len(cv_results['params'])

    # Include additional keys from standardized_results
    exclude_keys = set(['best_estimator_', 'best_params_', 'best_score_',
                        'scoring', 'cv_results_'])
    additional_keys = set(standardized_results) - exclude_keys
    for key in additional_keys:
        inline_contents[key] = standardized_results[key]

    return inline_contents, standardized_results


def summarize_optimized_results(model_results, result_title=None, **kwargs):
    """
    Generates a comprehensive summary of optimized results from one or
    multiple machine learning model tuning processes.

    This function dynamically formats the tuning results, including the best
    estimator, parameters, and cross-validation scores, into a structured
    textual summary. It supports both individual model results and collections
    of results from multiple models.

    Parameters
    ----------
    model_results : dict
        A dictionary containing model tuning results. It can follow two
        structures: a single model's results or a nested dictionary with
        multiple models' results.
    result_title : str, optional
        The title for the summarized results. If not provided, defaults to
        "Optimized Results".
    **kwargs : dict
        Additional keyword arguments to be passed to the summary function for
        individual model results.

    Returns
    -------
    str
        A formatted string summary of the optimized model results. For multiple
        models, each model's results are presented sequentially with a distinct
        header.

    Notes
    -----
    - The function auto-detects the structure of `model_results` to apply the
      appropriate formatting.
    - When handling multiple models, each model's results are separated by
      horizontal lines for clear distinction.

    Examples
    --------
    >>> from sklearn.svm import SVC 
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.api.summary import summarize_optimized_results
    Single model's results:

    >>> model_results = {
    ...     'best_estimator_': RandomForestClassifier(),
    ...     'best_params_': {'n_estimators': 100, 'max_depth': 5},
    ...     'cv_results_': {'mean_test_score': [0.8, 0.85, 0.83], }
    ... }
    >>> print(summarize_optimized_results(model_results))
    
    Multiple models' results:

    >>> models_results = {
    ...     'RandomForest': {
    ...         'best_estimator_': RandomForestClassifier(),
    ...         'best_params_': {'n_estimators': 100, 'max_depth': 5},
    ...         'cv_results_': {'mean_test_score': [0.8, 0.85, 0.83]}
    ...     },
    ...     'SVC': {
    ...         'best_estimator_': SVC(),
    ...         'best_params_': {'C': 1, 'kernel': 'linear'},
    ...         'cv_results_': {'mean_test_score': [0.9, 0.92, 0.91]}
    ...     }
    ... }
    >>> print(summarize_optimized_results(models_results))

    The output for multiple models will include a header for each model, 
    followed by its tuning summary, facilitating easy comparison between 
    different model performances.
    """
    # Detect the structure type of the provided model results
    structure_type = detect_structure_type(model_results)
    
    # Raise an error for unrecognized data structure
    if structure_type == 'unknown':
        raise ValueError("The provided model results do not follow the expected structure.")
    
    # Handle the case of a single model's results (structure 1)
    if structure_type == 'structure_1':
        return summarize_model_results(model_results, **kwargs)
    
    # If handling multiple models' results (structure 2):
    # Separate keys (estimator names) and values (their corresponding results)
    estimators_keys, model_results_list = zip(*model_results.items())

    # Get max_col_length to be consistent of all lines 
    max_col_lengths, base_max_width= get_max_col_lengths(model_results_list )
    # Generate formatted summaries for each model's results
    formatted_results = [
        summarize_model_results(
            model_result,
            max_col_lengths= max_col_lengths, 
            max_width= base_max_width, 
            **kwargs
        )
        for model_result in model_results_list
    ]
    # update again max_width 
    # Determine maximum width based on the '=' separator line in the first model's summary
    max_width= max(len(line) for line in formatted_results[0].split('\n') if '==' in line)
    if max_width <= base_max_width: 
        max_width=base_max_width
        
    # update the formated results with maxwidth 
    # formatted_results = [
    #     summarize_model_results(model_result, max_width=max_width, **kwargs)
    #     for model_result in model_results_list
    # ]
    # Prepare separator lines and centered estimator keys as headers
    up_line = '=' * max_width  # Top border
    down_line = '-' * max_width  # Sub-header
    
    # Format headers with estimator keys, ensuring they're properly centered
    formatted_keys = [
        f'{up_line}\n|{key.center(max_width-2)}|\n{down_line}\n'
        for key in estimators_keys
    ]
    
    # Combine headers with their corresponding model summaries
    formatted_results_with_keys = [
        formatted_keys[i] + formatted_results[i] + '\n'
        for i in range(len(estimators_keys))
    ]
    
    # Title for the combined results, defaulted to 'Optimized Results' if not provided
    table_name = result_title or 'Optimized Results'
    
    # Compile the full summary by joining individual model summaries
    return f'{table_name.center(max_width)}' + '\n' + '\n\n'.join(formatted_results_with_keys)


def standardize_keys(model_results):
    """
    Standardizes the keys of a model results dictionary to a consistent format.

    Given a dictionary containing results from model fitting processes, this function
    maps various alternative key names to a set of standardized key names. This is
    useful for ensuring consistency when working with results from different model
    training processes or APIs that may use slightly different terminology for storing
    model results.

    Parameters
    ----------
    model_results : dict
        A dictionary containing keys that represent model results such as the best
        estimator, best parameters, and cross-validation results.

    Returns
    -------
    dict
        A new dictionary where keys have been standardized according to a predefined
        mapping. This includes keys for the best estimator, best parameters, and
        cross-validation results.

    Examples
    --------
    >>> from gofast.api.summary import standardize_keys
    >>> model_results = {
    ...     'estimator': "RandomForest",
    ...     'params': {'n_estimators': 100, 'max_depth': 5},
    ...     'fold_results': {'mean_test_score': [0.9, 0.95, 0.85]}
    ... }
    >>> standardized = standardize_keys(model_results)
    >>> print(standardized)
    {
        'best_estimator_': "RandomForest",
        'best_params_': {'n_estimators': 100, 'max_depth': 5},
        'cv_results_': {'mean_test_score': [0.9, 0.95, 0.85]}
    }

    Notes
    -----
    - The function does not modify the input dictionary but returns a new dictionary
      with standardized keys.
    - If a standardized key and its alternatives are not found in the input dictionary,
      they will not be included in the output.
    - If multiple alternative keys for the same standard key are present, the first
      found alternative is used.
    """
    # map_keys = {
    #     'best_estimator_': ['model', 'estimator', 'best_estimator'],
    #     'best_params_': ['parameters', 'params', 'best_parameters', 'best_parameters_'],
    #     'best_scores_': ['scores', 'best_score', 'score', ], 
    #     'cv_results_': ['results', 'fold_results', 'cv_results'], 
    # }
    # standardized_results = {}
    # for standard_key, alternatives in map_keys.items():
    #     # Check each alternative key for presence in model_results
    #     for alt_key in alternatives:
    #         if alt_key in model_results:
    #             standardized_results[standard_key] = model_results[alt_key]
    #             break  # Break after finding the first matching alternative
    # # Additionally, check for any keys that are already correct and not duplicated
    # for key in map_keys.keys():
    #     if key in model_results and key not in standardized_results:
    #         standardized_results[key] = model_results[key]

    # return standardized_results
    regex_map = RegexMap()  
    standardized_results = {}

    # Iterate over each key in the original results and standardize it
    for original_key in model_results:
        standard_key = regex_map.find_key(original_key)
        if standard_key:
            standardized_results[standard_key] = model_results[original_key]

    return standardized_results

def prepare_cv_results_dataframe(cv_results):
    """
    Converts the cross-validation results dictionary into a pandas DataFrame
    for easier analysis and visualization.
    
    This function aggregates the results from cross-validation runs and 
    compiles them into a DataFrame. It includes parameters used for each run,
    mean and standard deviation of cross-validation (CV) scores, and overall
    performance metrics.

    Parameters
    ----------
    cv_results : dict
        The cross-validation results dictionary obtained from a model selection
        method like GridSearchCV or RandomizedSearchCV.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a set of parameters and their
        corresponding cross-validation statistics including mean CV score,
        standard deviation of CV scores, overall mean score, overall standard
        deviation score, and rank.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.datasets import load_iris
    >>> from gofast.api.summary import prepare_cv_results_dataframe
    >>> X, y = load_iris(return_X_y=True)
    >>> param_grid = {
    ...     'n_estimators': [100, 200],
    ...     'max_depth': [10, 20]
    ... }
    >>> clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    >>> clf.fit(X, y)
    >>> cv_results = clf.cv_results_
    >>> df = prepare_cv_results_dataframe(cv_results)
    >>> df.head()
         Params  Mean CV Score  ... Overall Std. Score Rank
    0  (10, 100)        0.9667  ...             0.0211    1
    1  (10, 200)        0.9667  ...             0.0211    1
    2  (20, 100)        0.9667  ...             0.0211    1
    3  (20, 200)        0.9600  ...             0.0249    4

    Notes
    -----
    - The function requires the `cv_results` dictionary from a fitted
      GridSearchCV or RandomizedSearchCV object.
    - It's designed to provide a quick overview of model selection results,
      making it easier to identify the best parameter combinations.
    """
    # Extract number of splits
    n_splits = sum(1 for key in cv_results if key.startswith(
        'split') and key.endswith('test_score'))
    
    if 'params' not in cv_results: 
        return pd.DataFrame () 
    # Gather general information
    data = []
    for i, params in enumerate(cv_results['params']):
        if n_splits:  
            cv_scores = [cv_results[f'split{split}_test_score'][i] for split in range(n_splits)]
            std_score = format_value(np.nanstd(cv_scores))
            mean_score = format_value(np.nanmean(cv_scores))
            fold_data = {
                "Params ": f"({', '.join([ str(i) for i in params.values()])})", 
                "Mean*": mean_score,
                "Std.*": std_score,
            }
        if 'mean_test_score' in cv_results:
            fold_data["Overall Mean*"] = format_value(cv_results['mean_test_score'][i])
        if 'std_test_score' in cv_results:
            fold_data["Overall Std.*"] = format_value(cv_results['std_test_score'][i])
        if 'rank_test_score' in cv_results:
            fold_data['Rank'] = cv_results['rank_test_score'][i]
        
        data.append(fold_data)
    # Create DataFrame
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    return df

def detect_structure_type(model_results):
    """
    Detects the structure type of model results.

    This function checks the structure of the model results to determine if it
    matches the format of structure_1 (a single model's results) or structure_2
    (multiple models' results stored in a nested manner).

    Parameters
    ----------
    model_results : dict
        A dictionary containing model fitting results, which could follow the format
        of either structure_1 or structure_2.

    Returns
    -------
    str
        A string indicating the detected structure type: either 'structure_1' for a
        single model's results, 'structure_2' for multiple models' results, or
        'unknown' if the structure does not match either expected format.

    Examples
    --------
    >>> structure_1 = {'best_estimator_': "Model1", 'best_params_': {}, 'cv_results_': {}}
    >>> print(detect_structure_type(structure_1))
    'structure_1'

    >>> structure_2 = {'estimator1': {'best_estimator_': "Model1",
    ...                                  'best_params_': {}, 'cv_results_': {}}}
    >>> print(detect_structure_type(structure_2))
    'structure_2'

    >>> unknown_structure = {'model': "Model1", 'parameters': {}}
    >>> print(detect_structure_type(unknown_structure))
    'unknown'

    Notes
    -----
    - The function assumes that the input dictionary contains the results of model
      fitting processes, which may include information such as the best estimator,
      best parameters, and cross-validation results.
    - The detection is based on the presence of specific keys or the structure of
      nested dictionaries.
    """
    # Check for structure_1 by looking for specific keys directly in model_results
    structure_1_keys = {'best_estimator_', 'best_params_', 'cv_results_'}
    if all(key in standardize_keys(model_results) for key in structure_1_keys):
        return 'structure_1'

    # Check for structure_2 by looking for nested dictionaries that contain specific keys
    if all(isinstance(val, dict) and all(key in standardize_keys(val) for key in structure_1_keys) 
           for val in model_results.values()):
        return 'structure_2'

    # If neither structure is detected, return 'unknown'
    return 'unknown'

def format_dataframe(
    df, title=None, 
    max_text_length=50, 
    max_width='auto',
    top_line='=', 
    sub_line='-', 
    bottom_line='=', 
    max_col_lengths=None 
    ):
    """
    Formats a pandas DataFrame into a string representation of a table,
    optionally including a title and customizing the table's appearance
    with specified line characters.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be formatted.
    title : str, optional
        The title of the table. If provided, it will be centered above
        the table. Defaults to None.
    max_text_length : int, optional
        The maximum allowed length for any text value in the table before
        truncation. Defaults to 50 characters.
    max_width : 'auto' or int, optional
        The maximum width of the table. If 'auto', the table width is
        automatically adjusted based on content. If an integer is provided,
        it specifies the maximum width; columns will be adjusted to fit.
        Defaults to 'auto'.
    top_line, sub_line, bottom_line : str, optional
        Characters used to create the top border, sub-header line, and
        bottom border of the table. Defaults to '=', '-', and '=' respectively.

    Returns
    -------
    str
        A string representing the formatted table.

    Examples
    --------
    >>> impport pandas as pd 
    >>> from gofast.api.summary import format_dataframe
    >>> data = {'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'Occupation': ['Engineer', 'Doctor', 'Artist']}
    >>> df = pd.DataFrame(data)
    >>> print(format_dataframe(df, title='Employee Table'))
         Employee Table     
    ========================
         Name Age Occupation
    ------------------------
    0   Alice  25   Engineer
    1     Bob  30     Doctor
    2 Charlie  35     Artist
    ========================

    Notes
    -----
    - The function dynamically adjusts the width of the table columns based
      on the content size and the `max_width` parameter.
    - Long text values in cells are truncated with '...' when exceeding
      `max_text_length`.
    """
    if not isinstance (df, pd.DataFrame): 
        raise TypeError(f"Expect DataFrame. Got {type(df).__name__!r} instead.")
    # Calculate the max length for the index
    max_index_length = max([len(str(index)) for index in df.index])
    
    # Calculate max length for each column including the column name,
    # and account for truncation
    if max_col_lengths is None: 
        max_col_lengths = {
            col: max(len(col), max(df[col].astype(str).apply(
                lambda x: len(x) if len(x) <= max_text_length else max_text_length + 3)))
            for col in df.columns
        }
    initial_space = max_index_length + sum(
        max_col_lengths.values()) + len(df.columns) - 1  # Spaces between columns

    if isinstance(max_width, (float, int)) and max_width > initial_space:
        # Distribute the extra_space among columns and the index
        extra_space = max_width - initial_space
        extra_space_per_col = extra_space // (len(df.columns) + (1 if max_index_length > 0 else 0))
        max_col_lengths = {col: v + extra_space_per_col for col, v in max_col_lengths.items()}

    # Adjust column values for truncation and formatting
    for col in df.columns:
        df[col] = df[col].astype(str).apply(
            lambda x: x if len(x) <= max_text_length else x[:max_text_length] + "...")
        max_col_lengths[col] = max(max_col_lengths[col], df[col].str.len().max())
    
    # Construct the header with padding for alignment
    header = " ".join([f"{col:>{max_col_lengths[col]}}" for col in df.columns])
    
    # Calculate total table width considering the corrected space calculation
    table_width = sum(max_col_lengths.values()) + len(df.columns) - 1 + max_index_length + (
        1 if max_index_length > 0 else 0)
    
    # Construct the separators
    top_header_border = top_line * table_width
    separator = sub_line * table_width
    top_bottom_border = bottom_line * table_width
    
    # Construct each row
    rows = []
    title_str = f"{title.center(table_width)}" if title else ""
    for index, row in df.iterrows():
        row_str = f"{str(index).ljust(max_index_length)} " + " ".join(
            [f"{value:>{max_col_lengths[col]}}" for col, value in iteritems_compat(row)])
        rows.append(row_str)
    
    # Combine all parts
    table = (f"{title_str}\n{top_header_border}\n{' ' * (max_index_length + 1)}"
             f"{header}\n{separator}\n") + "\n".join(rows) + f"\n{top_bottom_border}"
    
    return table

def format_key(key, max_length=None, include_colon=False, alignment='left',
               pad_colon=False):
    """
    Formats a key string according to the specified parameters.

    Parameters:
    - key (str): The key to format.
    - max_length (int, optional): The maximum length for the formatted key. 
      If None, it's calculated based on the key length.
    - include_colon (bool): Determines whether a colon and space should be 
      added after the key.
    - alignment (str): Specifies the alignment of the key. Expected values are 
     'left' or 'right'.

    Returns:
    - str: The formatted key string.
    # Example usage:
    print(format_key("ExampleKey", 20, include_colon=True, alignment='left'))
    print(format_key("Short", max_length=10, include_colon=False, alignment='right'))
    
    """
    # Calculate the base length of the key, optionally including the colon and space
    base_length = len(key) + (2 if include_colon else 0)
    
    # Determine the final max length, using the base length if no max_length is provided
    final_length = max_length if max_length is not None else base_length
    
    # Construct the key format string
    #pad key + space + ' :' to fit the max_length , adjust according when padding_colon is True
    key_format = "{}{} :".format(
        key, ' '*(max_length -len(key)) if pad_colon else '') if include_colon else key
    # Apply the specified alignment and padding
    formatted_key = ( f"{key_format: <{final_length}}" if alignment == 'left' 
                     else f"{key_format: >{final_length}}"
                     )
    return formatted_key

def key_formatter_format(
    key, formatter_obj, 
    title='', 
    max_key_length=None, 
    max_text_char=50, 
    **kws
    ):
    """
    Formats a key with an associated object (typically a DataFrame
    or a similar structure), aligning the object's string representation
    under the formatted key. Optionally includes a title above the object,
    and allows for customization of the key's maximum length and the
    object's text character width.

    If '%%' is found in the key, it is split at this point with the
    first part considered the key and the second part the title.

    Parameters
    ----------
    key : str
        The key to be formatted, potentially containing '%%' to split into key
        and title.
    frameobj : object
        The object to be formatted and aligned under the key, expected to have
        a string representation.
    title : str, optional
        A title for the frame object. If empty and '%%' is used in the key, 
        the second part becomes the title. Defaults
        to an empty string.
    max_key_length : int, optional
        The maximum length for the key, impacting alignment. If None, the 
        actual length of `key` is used. Defaults to None.
    max_text_char : int, optional
        The maximum number of characters for the object's text representation 
        before truncation. Defaults to 50.

    Returns
    -------
    str
        The formatted key (with optional title) followed by the
        aligned object as a string. The object is aligned to match
        the key's formatting for cohesive presentation.

    Examples
    --------
    Assuming `frameobj` is a DataFrame or has a similar multiline
    string representation, and `format_key` properly formats the key:

    >>> frameobj = someDataFrameRepresentation
    >>> print(key_formatter_format('MyKey%%My Title', frameobj))
    MyKey
            My Title
            Column1  Column2
            value1   value2
            value3   value4

    Notes
    -----
    - This function is designed for generating text reports or summaries where
      consistent alignment between keys and their associated objects is 
      required.
    - Utilizes `format_key` for key formatting, which should handle specific 
      formatting needs, including truncation and alignment.
    - The `title` is centered based on the first line of the `formatter_obj`'s 
      string representation for aesthetic alignment.
    """
    if not isinstance_( formatter_obj, ( MultiFrameFormatter, DataFrameFormatter, 
                                    DescriptionFormatter)) : 
        raise TypeError(
            f"Expect formatter object. Got {type(formatter_obj).__name__!r}")
    # Format the key with a colon, using the provided or calculated max key length
    # if %% in the key split and considered key and title 
    
    if "%%" in str(key): 
        key, title = key.split("%%")
        
    formatted_key = format_key(key, max_length=max_key_length, **kws)
    
    # Format the DataFrame according to specifications
    # Split the formatted DataFrame into lines for alignment adjustments
    frame_lines = formatter_obj.__str__().split('\n')
    
    # Determine the number of spaces to prepend to each line based on the 
    # length of the formatted key
    # Adding 1 additional space for alignment under the formatted key
    space_prefix = ' ' * (len(formatted_key) + 1)
    
    # Prepend spaces to each line of the formatted DataFrame except for the 
    # first line (it's already aligned) 
    aligned_df = space_prefix + frame_lines[0] + '\n' + '\n'.join(
        [space_prefix + line for line in frame_lines[1:]])
    # Center the title of dataframe 
    if title: 
        title = title.title ().center(len(frame_lines[0]))
    
    # Combine the formatted title dataframe with the aligned DataFrame
    result = f"{formatted_key}{title}\n{aligned_df}"
    
    return result
    
def key_dataframe_format(
    key, df, title ='',
    max_key_length=None, 
    max_text_char=50, 
    **kws
    ):
    """
    Formats a key-value pair where the value is a pandas DataFrame, aligning
    the DataFrame under a formatted key with an optional maximum key length and
    maximum text character width for the DataFrame.

    Parameters
    ----------
    key : str
        The key associated with the DataFrame, which will be formatted with
        a colon and aligned to the left.
    df : pandas.DataFrame
        The DataFrame to be formatted and aligned under the key.
    max_key_length : int, optional
        The maximum length of the key. If None, the actual length of the `key`
        is used. Defaults to None.
    max_text_char : int, optional
        The maximum number of characters allowed for each cell within the
        DataFrame before truncation. Defaults to 50.

    Returns
    -------
    str
        The formatted key followed by the aligned DataFrame as a string.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.summary import key_dataframe_format
    >>> df = pd.DataFrame({'A': [1, 2], 'B': ['text', 'another longer text']})
    >>> key = 'DataFrame Key'
    >>> print(dataframe_key_format(key, df))
    DataFrame Key:
                    A   B
                    1   text
                    2   another longer text

    Notes
    -----
    - The function is particularly useful for including pandas DataFrames within
      textual reports or summaries, ensuring the DataFrame's alignment matches
      the accompanying textual content.
    - The `format_key` and `format_dataframe` helper functions are utilized to
      format the key and DataFrame, respectively. These should be defined to
      handle specific formatting and alignment needs.
    - If `max_key_length` is provided and exceeds the actual length of the `key`,
      additional spaces are added to align the DataFrame's first column directly
      under the formatted key.
    """
    # Format the key with a colon, using the provided or calculated max key length
    # if %% in the key split and considered key and title 
    if "%%" in str(key): 
        key, title = key.split("%%")
        
    formatted_key = format_key(key, max_length=max_key_length,**kws)
    
    # Format the DataFrame according to specifications
    formatted_df = format_dataframe(df, max_text_length=max_text_char)
    
    # Split the formatted DataFrame into lines for alignment adjustments
    df_lines = formatted_df.split('\n')
    
    # Determine the number of spaces to prepend to each line based on the 
    # length of the formatted key
    # Adding 1 additional space for alignment under the formatted key
    space_prefix = ' ' * (len(formatted_key) + 1)
    
    # Prepend spaces to each line of the formatted DataFrame except for the 
    # first line (it's already aligned) 
    aligned_df = space_prefix + df_lines[0] + '\n' + '\n'.join(
        [space_prefix + line for line in df_lines[1:]])
    # Center the title of dataframe 
    if title: 
        title = title.title ().center(len(df_lines[0]))
    
    # Combine the formatted title dataframe with the aligned DataFrame
    result = f"{formatted_key}{title}\n{aligned_df}"
    
    return result

def format_dict(dct):
    """
    Formats a dictionary into a summary string that provides an overview
    of its content, distinguishing between numeric and non-numeric values
    and identifying the presence of NaN values among numeric entries.

    Parameters
    ----------
    dct : dict
        The dictionary to summarize, which can contain a mix of numeric
        and non-numeric values.

    Returns
    -------
    str
        A summary string of the dictionary's contents, including mean values
        for numeric data and counts of numeric vs. non-numeric entries.

    Examples
    --------
    >>> from gofast.api.summary import format_dict
    >>> mixed_dict = {
    ...     "a": "apple",
    ...     "b": 2,
    ...     "c": 3.5,
    ...     "d": float('nan'),
    ...     "e": "banana"
    ... }
    >>> print(format_dict(mixed_dict))
    Dict ~ len:5 - values: <mean: 2.7500 - numval: 3 - nonnumval: 2 ...>

    >>> numeric_dict = {
    ...     "one": 1,
    ...     "two": 2,
    ...     "three": 3,
    ...     "four": float('nan'),
    ...     "five": 5
    ... }
    >>> print(format_dict(numeric_dict))
    Dict ~ len:5 - values: <mean: 2.7500 - numval: 4 - nonnumval: 0 ...>

    Notes
    -----
    - The function calculates the mean value only for numeric entries, ignoring
      any NaN values in the calculation.
    - The function identifies the presence of NaN values among numeric entries
      and reflects this in the summary.
    - Non-numeric entries are counted separately, and the dictionary is classified
      as 'numeric', 'mixed', or 'non-numeric' based on its contents.
    """
    # Initialize counts
    num_values_count = 0
    non_num_values_count = 0
    exist_nan = False
    
    # Iterate through the dictionary values to categorize them
    for value in dct.values():
        if isinstance(value, (int, float)):
            num_values_count += 1
            if isinstance(value, float) and np.isnan(value):
                exist_nan = True
        else:
            non_num_values_count += 1
            
    # Determine dtype description
    dtype_description = ( 
        "numeric" if non_num_values_count == 0 else "mixed" 
        if num_values_count > 0 else "non-numeric"
        )
    
    # Calculate mean for numeric values if any, ignoring NaNs
    if num_values_count > 0:
        numeric_values = np.array([value for value in dct.values() if isinstance(
            value, (int, float))], dtype=float)
        mean_value = np.nanmean(numeric_values)
        summary_str = (
            "Dict ~ len:{} - values: <mean: {:.4f} - numval: {}"
            " - nonnumval: {} - dtype: {} - exist_nan:"
            " {}>").format(
            len(dct), mean_value, num_values_count, non_num_values_count,
            dtype_description, exist_nan
        )
    else:
        summary_str = ( 
            "Dict ~ len:{} - values: <numval: {} - nonnumval: {}"
            " - dtype: {} - exist_nan: {}>").format(
            len(dct), num_values_count, non_num_values_count, dtype_description,
            exist_nan
        )

    return summary_str


def format_list(lst):
    """
    Formats a list into a summary string, identifying whether the list
    is purely numeric, mixed, or non-numeric, and includes statistics
    like mean values for numeric lists and the presence of NaN values.

    Parameters
    ----------
    lst : list
        The list to summarize, which may contain numeric, non-numeric,
        or a mix of both types of values.

    Returns
    -------
    str
        A summary string of the list's contents, including the overall type
        (numeric, mixed, or non-numeric), mean values for numeric entries,
        and the presence of NaN values if applicable.

    Examples
    --------
    >>> from gofast.api.summary import format_list
    >>> numeric_list = [1, 2, 3, np.nan, 5]
    >>> print(format_list(numeric_list))
    List ~ len:5 - values: <mean: 2.7500 - dtype: numeric - exist_nan: True>

    >>> mixed_list = ["apple", 2, 3.5, np.nan, "banana"]
    >>> print(format_list(mixed_list))
    List ~ len:5 - values: <numval: 2 - nonnumval: 3 - dtype: mixed - exist_nan: True>

    Notes
    -----
    - Numeric entries are processed to calculate a mean value, excluding any NaNs.
    - The presence of NaN values among numeric entries is noted in the summary.
    - The classification of the list as 'numeric', 'mixed', or 'non-numeric' is
      based on the types of values it contains.
    """
    # Check if all elements in the list are numeric (int or float)
    all_numeric = all(isinstance(x, (int, float)) for x in lst)
    exist_nan = any(np.isnan(x) for x in lst if isinstance(x, float))

    if all_numeric:
        # Calculate mean for numeric list, ignoring NaNs
        # # Convert list to NumPy array for nanmean calculation
        numeric_values = np.array(lst, dtype=float)  
        mean_value = np.nanmean(numeric_values)
        arr_str = ("List ~ len:{} - values: < mean: {:.4f} - dtype:"
                   " numeric - exist_nan: {}>").format(
            len( lst), mean_value, exist_nan
        )
    else:
        # For mixed or non-numeric lists, calculate the count of numeric and non-numeric values
        num_values_count = sum(isinstance(x, (int, float)) for x in lst)
        non_num_values_count = len(lst) - num_values_count
        dtype_description = "mixed" if not all_numeric else "non-numeric"
        arr_str = ( "List ~ len:{} - values: <numval: {} - nonnumval: {}"
                   " - dtype: {} - exist_nan: {}>").format(
            len( lst), num_values_count, non_num_values_count, dtype_description, exist_nan
        )

    return arr_str

def format_array(arr):
    """
    Formats a NumPy array into a summary string, calculating mean values
    for numeric arrays and identifying the presence of NaN values. Non-numeric
    arrays are noted as such without attempting to summarize their contents.

    Parameters
    ----------
    arr : numpy.ndarray
        The NumPy array to summarize, which can be numeric or non-numeric.

    Returns
    -------
    str
        A summary string of the array's contents, including shape, mean value
        for numeric arrays, and the detection of NaN values if present.

    Examples
    --------
    >>> from gofast.api.summary import format_array
    >>> numeric_arr = np.array([1, 2, 3, np.nan, 5])
    >>> print(format_array(numeric_arr))
    Array ~ shape=<5> - mean: 2.7500 - dtype: float64 - exist_nan:True

    >>> mixed_arr = np.array(["apple", 2, 3.5, np.nan, "banana"], dtype=object)
    >>> print(format_array(mixed_arr))
    Array ~ shape=<5> - dtype: object - exist_nan:True

    Notes
    -----
    - For numeric arrays, the function calculates the mean while ignoring any NaNs
      and identifies the presence of NaN values.
    - Non-numeric or mixed-type arrays are labeled with their data type without
      attempting numerical summarization.
    """
 
    arr_str = ""
    # Check if the array contains NaN values; works only for numeric arrays
    exist_nan = np.isnan(arr).any() if np.issubdtype(arr.dtype, np.number) else False

    if np.issubdtype(arr.dtype, np.number):
        # Formatting string for numeric arrays
        arr_str = "Array ~ {}values: <mean: {:.4f} - dtype: {}{}>".format(
            f"shape=<{arr.shape[0]}{'x' + str(arr.shape[1]) if arr.ndim == 2 else ''}> - ",
            np.nanmean(arr),  # Use nanmean to calculate mean while ignoring NaNs
            arr.dtype.name,
            ' - exist_nan:True' if exist_nan else ''
        )
    else:
        # Attempt to handle mixed or non-numeric data without pandas,
        # acknowledging that numpy arrays are typically of a single data type
        # Here we consider the array non-numeric and don't attempt to summarize numerically
        dtype_description = "mixed" if arr.dtype == object else arr.dtype.name
        arr_str = "Array ~ {}values: <dtype: {}{}>".format(
            f"shape=<{arr.shape[0]}{'x' + str(arr.shape[1]) if arr.ndim == 2 else ''}> - ",
            dtype_description,
            ' - exist_nan:True' if exist_nan else ''
        )
    return arr_str



def format_series(series):
    """
    Formats a pandas Series into a concise summary string. The summary includes
    the series name, mean (for numeric series), length, dtype, and an indicator
    of whether NaN values are present.

    Parameters
    ----------
    series : pandas.Series
        The series to be summarized.

    Returns
    -------
    str
        A summary string describing key aspects of the series.

    Examples
    --------
    >>> numeric_series = pd.Series([1, 2, 3, np.nan, 5, 6], name='NumericSeries')
    >>> print(format_series(numeric_series))
    Series ~ name=<NumericSeries> - values: <mean: 3.4000 - length: 6 -...>

    >>> non_numeric_series = pd.Series(['apple', 'banana', np.nan, 'cherry',
                                        'date', 'eggfruit', 'fig'], name='FruitSeries')
    >>> print(format_series(non_numeric_series))
    Series ~ name=<FruitSeries> - values: <numval: 0 - nonnumval: 6 - ...>

    Notes
    -----
    - For numeric series with less than 7 values, the function calculates and
      includes the mean value, excluding any NaNs from the calculation.
    - For non-numeric series or those with 7 or more values, the function counts
      the number of numeric and non-numeric values separately and indicates the
      presence of NaN values.
    - The series' data type (`dtype`) and the presence of NaN values are always
      included in the summary.
    """
    series_str = ''
    if not isinstance ( series, pd.Series):
        return series 
    
    if series.dtype.kind in 'biufc' and len(series) < 7:  # Check if series is numeric and length < 7
        series_str = "Series ~ {}values: <mean: {:.4f} - len: {} - dtype: {}{}>".format(
            f"name=<{series.name}> - " if series.name is not None else '',
            np.mean(series.values), 
            len(series), 
            series.dtype.name, 
            ' - exist_nan:True' if series.isnull().any() else ''
        )
    else:  # Handle non-numeric series or series with length >= 7
        num_values = series.apply(
            lambda x: isinstance(x, (int, float)) and not np.isnan(x)).sum()
        non_num_values = len(
            series) - num_values - series.isnull().sum()  # Exclude NaNs from non-numeric count
        series_str = "Series ~ {}values: <numval: {} - nonnumval: {} - length: {} - dtype: {}{}>".format(
            f"name=<{series.name}> - " if series.name is not None else '',
            num_values, 
            non_num_values, 
            len(series), 
            series.dtype.name, 
            ' - exist_nan:True' if series.isnull().any() else ''
        )
    return series_str

def uniform_dfs_formatter(
        *dfs, titles=None, max_width='auto', aesthetic_space_allocation=4):
    """
    Formats a series of pandas DataFrames into a consistent width, optionally
    assigns titles to each, and compiles them into a single string. The width
    of each DataFrame is adjusted to match either the widest DataFrame in the
    series or a specified maximum width. Titles can be assigned to each DataFrame,
    and additional aesthetic spacing can be allocated for visual separation.

    Parameters
    ----------
    *dfs : pandas.DataFrame
        A variable number of pandas DataFrame objects to be formatted.
    titles : list of str, optional
        Titles for each DataFrame. If provided, the number of titles should
        match the number of DataFrames. If fewer titles are provided, the
        remaining DataFrames will have empty titles. Defaults to None, which
        results in no titles being assigned.
    max_width : int or 'auto', optional
        The maximum width for the formatted DataFrames. If 'auto', the width
        is determined based on the widest DataFrame, plus any aesthetic
        space allocation. If an integer is provided, it specifies the
        maximum width directly. Defaults to 'auto'.
    aesthetic_space_allocation : int, optional
        Additional spaces added for visual separation between the formatted
        DataFrames. Only applied when calculating `auto_max_width`. Defaults
        to 4.

    Returns
    -------
    str
        A single string containing all the formatted DataFrames, separated
        by double newlines.

    Raises
    ------
    TypeError
        If any of the inputs are not pandas DataFrame objects.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
    >>> summary_str = uniform_dfs_formatter(df1, df2, titles=['First DF', 'Second DF'])
    >>> print(summary_str)
    
    Notes
    -----
    - This function is useful for preparing multiple DataFrames for a
      unified visual presentation, such as in a report or console output.
    - The `max_width` parameter allows for control over the presentation
      width, accommodating scenarios with width constraints.
    - The `aesthetic_space_allocation` is particularly useful when formatting
      the output for readability, ensuring that adjacent DataFrames are
      visually separated.
    """
    if not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise TypeError("All inputs must be pandas DataFrame objects.")
    # Ensure titles list matches the number of DataFrames, filling with '' if necessary  
    titles = (titles or [''])[:len(dfs)] + [''] * (len(dfs) - len(titles or [])) 
    
    # Calculate the automatic maximum width among all DataFrames, adding aesthetic spaces
    auto_max_width = max(get_table_width(df) for df in dfs) + aesthetic_space_allocation

    # Adjust max_width based on the 'auto' setting or ensure it respects the calculated width
    max_width = auto_max_width if max_width == 'auto' or isinstance(
        max_width, str) else max(int(max_width), auto_max_width)

    # Format each DataFrame with its title and the determined maximum width
    formatted_dfs = [
        format_dataframe(df, title=title, max_width='auto') 
        for df, title in zip(dfs, titles)
    ]
    # max_width = max(
    #     [len(fmt_table.split('\n')[0]) for fmt_table in formatted_dfs]) 
    
    # formatted_dfs = [
    #     format_dataframe(df, title=title, max_width=max_width+ aesthetic_space_allocation) 
    #     for df, title in zip(dfs, titles)
    # ]
    # Combine all formatted DataFrame strings, separated by double newlines
    summary_report = "\n\n".join(formatted_dfs)

    return summary_report


if __name__ == "__main__":
    # Example usage:
    # from gofast.api.summary import format_report 
    report_data = {
        'Total Sales': 123456.789,
        'Average Rating': 4.321,
        'Number of Reviews': 987,
        'Key with long name': 'Example text', 
        'series': pd.Series ([1, 'banana', float('nan')])
    }

    report_title = 'Monthly Sales Report'
    formatted_report = format_report(report_data, report_title)
    print(formatted_report)
    

    df_data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, None, 8, 9],
        'C': ['foo', 'bar', 'baz', 'qux', 'quux'],
        'D': [0.1, 0.2, 0.3, np.nan, 0.5]
    }
    