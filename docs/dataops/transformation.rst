.. _transformation:

Transformation
===============

.. currentmodule:: gofast.dataops.transformation

The :mod:`gofast.dataops.transformation` module focuses on various transformations and formatting operations for data. This module provides functions to handle text summarization, data splitting, data sanitization, and column name formatting, among other operations.

Key Features
------------
- **Text Summarization**:
  Functions to extract and summarize the most representative sentences from text columns in a DataFrame.
  
  - :func:`~gofast.dataops.transformation.summarize_text_columns`: Applies extractive summarization to specified text columns in a DataFrame and optionally encodes and compresses the summaries.

- **Data Splitting**:
  Methods to split DataFrames into sub-DataFrames based on column data types.
  
  - :func:`~gofast.dataops.transformation.split_data`: Splits a DataFrame into sub-DataFrames based on the data types of the columns, with an option to treat datetime columns as numeric.

- **Data Sanitization**:
  Tools to clean DataFrames by handling missing values, duplicates, outliers, and transforming string data for consistency.
  
  - :func:`~gofast.dataops.transformation.sanitize`: Cleans a DataFrame by handling missing values, removing duplicates, detecting and removing outliers, and transforming string data for consistency.

- **Column Name Formatting**:
  Functions to shorten and format long column names for better readability and usability.
  
  - :func:`~gofast.dataops.transformation.format_long_column_names`: Modifies long column names in a DataFrame to a more concise format, changing the case as specified, and optionally returning a mapping of new column names to original names.

- **Grouping and Filtering**:
  Tools to filter a DataFrame based on specified group values and apply additional conditional filters, optionally sorting the resulting DataFrame.
  
  - :func:`~gofast.dataops.transformation.group_and_filter`: Filters a DataFrame based on specified group values in a column and applies additional conditional filters, optionally sorting the resulting DataFrame.

- **Smart Grouping**:
  Functions to group a DataFrame by specified column(s) and apply various optional transformations including aggregation, conditional filtering, and sorting.
  
  - :func:`~gofast.dataops.transformation.smart_group`: Groups a DataFrame by specified column(s) and applies various optional transformations including aggregation, conditional filtering, and sorting.

Function Descriptions
---------------------

summarize_text_columns
~~~~~~~~~~~~~~~~~~~~~~

Applies extractive summarization to specified text columns in a pandas DataFrame. Each text entry in the specified columns is summarized to its most representative sentence based on TF-IDF scores, considering the provided stop words. Optionally, encodes and compresses the summaries.

The method utilizes TF-IDF vectorization to determine the importance of sentences within a text. The most representative sentence is identified based on cosine similarity of TF-IDF vectors.

Example 1:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import summarize_text_columns

    data = {
        'id': [1, 2],
        'column1': ["Sentence one. Sentence two. Sentence three.", 
                    "Another sentence one. Another sentence two. Another sentence three."],
        'column2': ["More text here. Even more text here.", 
                    "Second example here. Another example here."]
    }
    df = pd.DataFrame(data)
    summarized_df = summarize_text_columns(df, ['column1', 'column2'], 
                                           stop_words='english', encode=True, 
                                           drop_original=True, compression_method='mean')
    print(summarized_df.columns)

Example 2:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import summarize_text_columns

    data = {
        'id': [1, 2],
        'column1': ["This is a test. Testing text summarization. Summarization works.", 
                    "Another test sentence. Checking functionality. All good."],
        'column2': ["Sample text. More sample text. Example text.", 
                    "Text for testing. More examples. Summarizing examples."]
    }
    df = pd.DataFrame(data)
    summarized_df = summarize_text_columns(df, ['column1'], stop_words='english', 
                                           encode=True, drop_original=True)
    print(summarized_df)

split_data
~~~~~~~~~~

Splits a DataFrame into sub-DataFrames based on data types of the columns, with an option to treat datetime columns as numeric. The function categorizes columns into numeric, categorical, and datetime types.

Example 1:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import split_data

    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
        'C': ['apple', 'banana', 'cherry']
    })
    numeric_df, categoric_df = split_data(df, 'biselect')
    print(numeric_df)
    print(categoric_df)

Example 2:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import split_data

    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']),
        'C': ['dog', 'cat', 'mouse'],
        'D': [True, False, True]
    })
    numeric_df = split_data(df, 'numeric_only')
    print(numeric_df)

sanitize
~~~~~~~~

Performs data cleaning on a DataFrame with many options. This function handles missing values, removes duplicates, detects and removes outliers, and transforms string data for consistency. It is useful for preparing data for analysis or modeling by ensuring it is clean and consistent.

Example 1:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import sanitize

    data = {'A': [1, 2, None, 4], 'B': ['X', 'Y', 'Y', None], 'C': [1, 1, 2, 2]}
    df = pd.DataFrame(data)
    cleaned_df = sanitize(df, fill_missing='median', remove_duplicates=True, 
                          outlier_method='z_score', consistency_transform='lower', threshold=3)
    print(cleaned_df)

Example 2:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import sanitize

    data = {'A': [1, 2, 2, 4, None], 'B': [None, 'X', 'Y', 'Y', 'Z'], 'C': [10, 20, 20, 30, 40]}
    df = pd.DataFrame(data)
    cleaned_df = sanitize(df, fill_missing='mode', remove_duplicates=False, 
                          outlier_method='iqr', consistency_transform='upper', threshold=1.5)
    print(cleaned_df)

format_long_column_names
~~~~~~~~~~~~~~~~~~~~~~~~

Modifies long column names in a DataFrame to a more concise format. The function shortens column names, changes the case as specified, and optionally returns a mapping of new column names to original names.

Example 1:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import format_long_column_names

    data = {'VeryLongColumnNameIndeed': [1, 2, 3], 'AnotherLongColumnName': [4, 5, 6]}
    df = pd.DataFrame(data)
    new_df, mapping = format_long_column_names(df, max_length=10, return_mapping=True, name_case='capitalize')
    print(new_df.columns)
    print(mapping)

Example 2:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import format_long_column_names

    data = {'ExtremelyLongName': [10, 20, 30], 'LongColumnName': [40, 50, 60]}
    df = pd.DataFrame(data)
    new_df = format_long_column_names(df, max_length=5, name_case='lowercase')
    print(new_df.columns)

group_and_filter
~~~~~~~~~~~~~~~~

Filters a DataFrame based on specified group values in a column and applies additional conditional filters. Optionally sorts the resulting DataFrame. This function is useful for segmenting data and applying complex filtering conditions.

Example 1:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import group_and_filter

    df = pd.DataFrame({
        'A': ['a', 'b', 'a', 'd'],
        'B': [1, 2, 3, 4]
    })
    filtered_df = group_and_filter(df, 'A', ['a', 'd'], sort=True)
    print(filtered_df)

Example 2:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import group_and_filter

    df = pd.DataFrame({
        'A': ['a', 'b', 'a', 'd'],
        'B': [1, 2, 3, 4]
    })
    filtered_df = group_and_filter(df, 'A', ['a'], conditional_filters={'B': ('>', 2)}, mode='soft')
    print(filtered_df)

smart_group
~~~~~~~~~~~

Groups a DataFrame by specified column(s) and applies various optional transformations including aggregation, conditional filtering, and sorting. This function is versatile for performing grouped operations and obtaining summarized results.

Example 1:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import smart_group

    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar'],
        'B': [1, 2, 3, 4],
        'C': [2, 4, 6, 8]
    })
    grouped_df = smart_group(df, group_by='A', aggregations={'B': 'sum', 'C': 'mean'})
    print(grouped_df)

Example 2:
.. code-block:: python

    import pandas as pd
    from gofast.dataops.transformation import smart_group

    df = pd.DataFrame({
        'A': ['x', 'y', 'x', 'y'],
        'B': [1, 2, 3, 4],
        'C': [5, 6, 7, 8]
    })
    grouped_df = smart_group(df, group_by='A', fill_na={'B': 0}, 
                             aggregations={'B': 'sum', 'C': 'mean'}, having={'B': '>3'})
    print(grouped_df)
