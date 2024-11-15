.. _management:

Management
==========

.. currentmodule:: gofast.dataops.management

The :mod:`gofast.dataops.management` module deals with data storage, retrieval, and dataset handling. This module provides functions to manage datasets effectively, ensuring efficient data handling and storage.

Key Features
------------
- **Data Fetching**: Functions to fetch and download remote datasets, including handling HTTP requests and scraping web data.
- **Unique Identifier Handling**: Tools to handle columns with high proportions of unique values, either by dropping or transforming them.
- **File Reading and Writing**: Functions to read from and write to various file formats, ensuring data is correctly formatted and sanitized.
- **HDF5 Storage Management**: Efficient storage and retrieval of large datasets using HDF5 files, supporting both numpy arrays and Pandas DataFrames.
- **Remote Data Management**: Utilities for handling datasets from remote locations, including fetching and storing remote data.

Function Descriptions
---------------------

handle_unique_identifiers
~~~~~~~~~~~~~~~~~~~~~~~~~
Examines columns in the DataFrame and handles columns with a high proportion of unique values. These columns can be either dropped or 
transformed based on specified criteria, facilitating better data analysis and modeling performance by reducing the number of 
effectively useless features.

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.management import handle_unique_identifiers

    # Example 1: Dropping columns with high proportions of unique values
    data = pd.DataFrame({
        'ID': range(1000),
        'Age': [25, 30, 35] * 333 + [40],
        'Salary': [50000, 60000, 75000, 90000] * 250
    })
    processed_data = handle_unique_identifiers(data, action='drop')
    print(processed_data.columns)

    # Example 2: Transforming columns with high proportions of unique values
    def cap_values(val):
        return min(val, 100)  # Cap values at 100

    processed_data = handle_unique_identifiers(data, action='transform', transform_func=cap_values)
    print(processed_data.head())

    # Example 3: Dropping columns with visualization of unique values
    processed_data = handle_unique_identifiers(data, action='drop', view=True)

read_data
~~~~~~~~~
Read all specific files and URLs allowed by the package. Readable files are systematically converted to a DataFrame.

Examples:

.. code-block:: python

    from gofast.dataops.management import read_data

    # Example 1: Reading a CSV file with sanitization and index reset
    df = read_data('data.csv', sanitize=True, reset_index=True)
    print(df.head())

    # Example 2: Reading a text file with specified delimiter
    df = read_data('data.txt', delimiter='|')
    print(df.head())

    # Example 3: Reading a numpy array from a .npy file
    df = read_data('data.npy')
    print(df.head())

    # Example 4: Reading from a remote URL
    url = 'https://example.com/data.csv'
    df = read_data(url, sanitize=True)
    print(df.head())

request_data
~~~~~~~~~~~~
Perform an HTTP request to a specified URL and process the response, with optional progress bar visualization.

Examples:

.. code-block:: python

    from gofast.dataops.management import request_data

    # Example 1: Simple GET request and parse JSON response
    response = request_data('https://api.github.com/user', auth=('user', 'pass'), as_json=True)
    print(response)

    # Example 2: POST request with data
    response = request_data('https://httpbin.org/post', method='post', data={'key': 'value'}, as_json=True)
    print(response)

    # Example 3: GET request with file download and progress bar
    response = request_data('https://example.com/largefile.zip', save_to_file=True, filename='largefile.zip', show_progress=True)
    print("Download complete!")

get_remote_data
~~~~~~~~~~~~~~~
Retrieve data from a remote location and optionally save it to a specified path.

Examples:

.. code-block:: python

    from gofast.dataops.management import get_remote_data

    # Example 1: Simple file download
    status = get_remote_data('https://example.com/file.csv')
    print(status)

    # Example 2: File download to a specific path
    status = get_remote_data('https://example.com/file.csv', save_path='/local/path')
    print(status)

handle_datasets_with_hdfstore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Handles storing or retrieving multiple Pandas DataFrames in an HDF5 file using `pd.HDFStore`.

Examples:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from gofast.dataops.management import handle_datasets_with_hdfstore

    # Example 1: Storing datasets in an HDF5 file
    df1 = pd.DataFrame(np.random.rand(100, 10), columns=[f'col_{i}' for i in range(10)])
    df2 = pd.DataFrame(np.random.randint(0, 100, size=(200, 5)), columns=['A', 'B', 'C', 'D', 'E'])
    handle_datasets_with_hdfstore('my_datasets.h5', {'df1': df1, 'df2': df2}, operation='store')

    # Example 2: Retrieving datasets from an HDF5 file
    datasets = handle_datasets_with_hdfstore('my_datasets.h5', operation='retrieve')
    print(datasets.keys())

store_or_retrieve_data
~~~~~~~~~~~~~~~~~~~~~~
Handles storing or retrieving multiple datasets (numpy arrays or Pandas DataFrames) in an HDF5 file.

Examples:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from gofast.dataops.management import store_or_retrieve_data

    # Example 1: Storing datasets
    df1 = pd.DataFrame(np.random.rand(100, 10), columns=[f'col_{i}' for i in range(10)])
    arr1 = np.random.rand(100, 10)
    store_or_retrieve_data('my_datasets.h5', {'df1': df1, 'arr1': arr1}, operation='store')

    # Example 2: Retrieving datasets
    datasets = store_or_retrieve_data('my_datasets.h5', operation='retrieve')
    print(datasets.keys())

base_storage
~~~~~~~~~~~~
Handles storing or retrieving multiple datasets (numpy arrays or Pandas DataFrames) in an HDF5 file.

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.dataops.management import base_storage

    # Example 1: Storing datasets
    data1 = np.random.rand(100, 10)
    df1 = pd.DataFrame(np.random.randint(0, 100, size=(200, 5)), columns=['A', 'B', 'C', 'D', 'E'])
    base_storage('my_datasets.h5', {'dataset1': data1, 'df1': df1}, operation='store')

    # Example 2: Retrieving datasets
    datasets = base_storage('my_datasets.h5', operation='retrieve')
    print(datasets.keys())

fetch_remote_data
~~~~~~~~~~~~~~~~~
Download a file from a remote URL and optionally save it to a specified location.

Examples:

.. code-block:: python

    from gofast.dataops.management import fetch_remote_data

    # Example 1: Simple file download
    status = fetch_remote_data('https://example.com/file.csv')
    print(status)

    # Example 2: File download to a specific path
    status = fetch_remote_data('https://example.com/file.csv', save_path='/local/path')
    print(status)

scrape_web_data
~~~~~~~~~~~~~~~
Scrape data from a web page using BeautifulSoup.

Examples:

.. code-block:: python

    from gofast.dataops.management import scrape_web_data

    # Example 1: Scraping div elements with a specific class
    url = 'https://example.com'
    element = 'div'
    class_name = 'content'
    data = scrape_web_data(url, element, class_name)
    for item in data:
        print(item.text)

    # Example 2: Scraping h1 elements
    url = 'https://example.com/articles'
    element = 'h1'
    data = scrape_web_data(url, element)
    for header in data:
        print(header.text)  # prints the text of each <h1> tag

    # Example 3: Scraping sections with specific attributes
    url = 'https://example.com/products'
    element = 'section'
    attributes = {'id': 'featured-products'}
    data = scrape_web_data(url, element, attributes=attributes)
    for product in data:
        print(product.text)  # prints the text of each section with id 'featured-products'

handle_datasets_in_h5
~~~~~~~~~~~~~~~~~~~~~
Handles storing or retrieving multiple datasets in an HDF5 file.

Examples:

.. code-block:: python

    import numpy as np
    from gofast.dataops.management import handle_datasets_in_h5

    # Example 1: Storing datasets
    data1 = np.random.rand(100, 10)
    data2 = np.random.rand(200, 5)
    handle_datasets_in_h5('my_datasets.h5', {'dataset1': data1, 'dataset2': data2}, operation='store')

    # Example 2: Retrieving datasets
    datasets = handle_datasets_in_h5('my_datasets.h5', operation='retrieve')
    print(datasets.keys())
