.. _load:

load
====

.. currentmodule:: gofast.datasets.load

The :mod:`gofast.datasets.load` module is designed to provide functions for loading various real-world datasets. These datasets span multiple domains such as biology, medicine, finance, and more. The datasets are curated to be used for machine learning tasks, including classification, regression, clustering, and exploratory data analysis.

Key Features
------------
- **Biology and Medicine**:
  Load datasets relevant to biological and medical research.

  - :func:`~gofast.datasets.load.load_iris`: Loads the classic Iris flower dataset.
  - :func:`~gofast.datasets.load.load_hlogs`: Loads hydro-logging data for hydrogeophysical analysis.
  - :func:`~gofast.datasets.load.load_dyspnea`: Loads the dyspnea dataset for medical research.

- **Finance and Economics**:
  Access datasets related to financial markets and economic studies.

  - :func:`~gofast.datasets.load.load_statlog`: Loads the Statlog Heart Disease dataset.
  - :func:`~gofast.datasets.load.load_jrs_bet`: Loads the JRS BET dataset for betting strategy analysis.
  - :func:`~gofast.datasets.load.load_bagoue`: Loads the Bagoue dataset for hydrological analysis.

- **Forensic Science**:
  Provides datasets used in forensic studies.

  - :func:`~gofast.datasets.load.load_forensic`: Loads forensic data for criminal investigation studies.

- **Environmental and Engineering**:
  Access datasets related to environmental and engineering studies.

  - :func:`~gofast.datasets.load.load_nansha`: Loads the Nansha Engineering and Hydrogeological Drilling Dataset.
  - :func:`~gofast.datasets.load.load_mxs`: Loads the dataset after implementing the mixture learning strategy (MXS).

Common Key Parameters
---------------------
- `<return_X_y>`: If `True`, returns `(data, target)` instead of a Bunch object.
- `<as_frame>`: If `True`, the data is returned as a pandas DataFrame. If `False`, a Bunch object is returned. By default, all functions in the :mod:`~gofast.datasets.load` module return a Bunch object.
- `<split_X_y>`: If `True`, splits the dataset into training and testing sets based on the specified test ratio.
- `<test_ratio>`: The proportion of the dataset to include in the test split.
- `<seed>`: Initializes the random number generator for reproducibility.

Data Handling and Attributes
----------------------------
By default, the `as_frame` parameter is `False`. However, if set to `True`, each function returns a Bunch object where information like `data`, `target`, `frame`, `target_names`, `feature_names`, and `DESCR` can be retrieved as attributes.

Here is an example of how to use the Bunch object:

.. code-block:: python

    from gofast.datasets.load import load_iris

    iris_data = load_iris()

    iris_data
    # Output: <Bunch object with keys: data, target, frame, target_names, feature_names, DESCR>

    print(iris_data.frame.head())
    # Output:
    #    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
    # 0                5.1               3.5                1.4               0.2       0
    # 1                4.9               3.0                1.4               0.2       0
    # 2                4.7               3.2                1.3               0.2       0
    # 3                4.6               3.1                1.5               0.2       0
    # 4                5.0

Function Descriptions
---------------------
Here are the descriptions and examples of the functions included in the :mod:`gofast.datasets.load` module.

load_hydro_metrics
~~~~~~~~~~~~~~~~~~

The :func:`gofast.datasets.load.load_hydro_metrics` function loads and returns the Hydro-Meteorological dataset collected in Yobouakro, 
S-P Agnibilekro, Cote d'Ivoire (West Africa). This dataset encompasses a comprehensive range of environmental and hydro-meteorological 
variables, including temperature, humidity, wind speed, solar radiation, evapotranspiration, rainfall, and river flow metrics. It's 
instrumental for studies in environmental science, agriculture, meteorology, hydrology, and climate change research, facilitating 
the analysis of weather patterns, water resource management, and the impacts of climate variability on agriculture.

Example 1:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_hydro_metrics

    # Load the dataset as a (data, target) tuple
    data, target = load_hydro_metrics(return_X_y=True)
    print(data.shape)  # (276, 8)  # Assuming 276 instances and 8 features.
    print(target.shape)  # (276,)  # Assuming 276 instances of the target variable.

Example 2:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_hydro_metrics

    # Load the dataset as a pandas DataFrame
    df, target = load_hydro_metrics(return_X_y=True, as_frame=True)
    print(df.head())  # Displays the first five rows of the feature data.
    print(target.head())  # Displays the first five rows of the target data.

.. notes:: 

    The function is designed with flexibility in mind, accommodating various forms of data analysis and machine learning tasks. By 
    providing options to return the data as arrays or a DataFrame, it enables users to leverage the full power of numpy and pandas for 
    data processing and analysis, respectively.

load_statlog
~~~~~~~~~~~~

The :func:`gofast.datasets.load.load_statlog` function loads and returns the Statlog Heart Disease dataset. The Statlog Heart dataset 
is a classic dataset in the machine learning community, used for binary classification tasks to predict the presence of heart disease 
in patients.

Example 1:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_statlog

    # Load the dataset as a (data, target) tuple
    data, target = load_statlog(return_X_y=True)
    print(data.shape)  # (270, 13)  # Assuming 270 instances and 13 features.
    print(target.shape)  # (270,)

Example 2:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_statlog

    # Load the dataset as a pandas DataFrame
    df, target = load_statlog(return_X_y=True, as_frame=True)
    print(df.head())  # Displays the first five rows of the feature data.
    print(target.head())  # Displays the first five rows of the target data.

Example 3:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_statlog

    # Load the dataset as a Bunch object
    dataset = load_statlog()
    print(dataset.data.shape)  # (270, 13)
    print(dataset.target.shape)  # (270,)
    print(dataset.DESCR)  # Displays the full description of the dataset.

.. notes:: 
    The function provides flexibility in handling the dataset, allowing users to load it as arrays, a DataFrame, or a Bunch 
    object depending on their specific needs. This dataset is particularly useful for developing and testing binary classification 
    models to predict heart disease.

load_dyspnea
~~~~~~~~~~~~

The :func:`gofast.datasets.load.load_dyspnea` function loads the dyspnea (difficulty in breathing) dataset, which is designed for 
medical research and predictive modeling in healthcare, particularly for conditions involving dyspnea. This function allows for 
flexible data loading tailored for various analysis needs, including diagnostics, severity assessment, outcome prediction, and 
symptom-based studies.

Example 1:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_dyspnea

    # Load the dataset as a Bunch object
    b = load_dyspnea()
    print(b.frame.head())  # Displays the first five rows of the dataset.
    print(b.target.head())  # Displays the first five rows of the target variable.

Example 2:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_dyspnea

    # Load the dataset as a (data, target) tuple
    X, y = load_dyspnea(return_X_y=True)
    print(X.shape)  # Assuming (1000, 20) for 1000 instances and 20 features.
    print(y.shape)  # Assuming (1000,)

Example 3:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_dyspnea

    # Load the dataset as a pandas DataFrame and split into training and test sets
    df, target = load_dyspnea(return_X_y=True, as_frame=True, split_X_y=True, test_ratio=0.3)
    print(df.head())  # Displays the first five rows of the training data.
    print(target.head())  # Displays the first five rows of the training target variable.

.. note:: 
   This function is highly versatile, supporting various configurations for data loading and preprocessing. 
   It can handle both raw and preprocessed versions of the dataset, split the data into training and test sets, and return 
   data in different formats to suit diverse analytical needs.

load_hlogs
~~~~~~~~~~

The :func:`gofast.datasets.load.load_hlogs` function loads the hydro-logging dataset for hydrogeophysical analysis. This dataset 
contains multi-target data suitable for both classification and regression tasks in the context of groundwater studies.

Example 1:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_hlogs

    # Load the dataset as a Bunch object
    b = load_hlogs()
    print(b.frame.head())  # Displays the first five rows of the dataset.
    print(b.target_names)  # Displays the names of the target variables.

Example 2:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_hlogs

    # Load the dataset as a (data, target) tuple
    data, target = load_hlogs(return_X_y=True)
    print(data.shape)  # Assuming (1000, 15) for 1000 instances and 15 features.
    print(target.shape)  # Assuming (1000,)

Example 3:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_hlogs

    # Load the dataset as a pandas DataFrame and split into training and test sets
    X, Xt, y, yt = load_hlogs(split_X_y=True, test_ratio=0.3, as_frame=True)
    print(X.head())  # Displays the first five rows of the training data.
    print(y.head())  # Displays the first five rows of the training target variable.

.. note::
   The :func:`~gofast.datasets.load.load_hlogs` function provides a flexible and comprehensive approach to loading hydro-logging data. 
   It allows for various configurations, including returning data as arrays or DataFrames, splitting data into training and test sets, 
   and handling multiple target variables for both classification and regression tasks.


load_nansha
~~~~~~~~~~~

The :func:`gofast.datasets.load.load_nansha` function loads the Nansha Engineering and Hydrogeological Drilling Dataset. This dataset contains multi-target information suitable for classification or regression problems in hydrogeological and geotechnical contexts.

Example 1:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_nansha

    # Load the dataset as a Bunch object
    b = load_nansha()
    print(b.frame.head())  # Displays the first five rows of the dataset.
    print(b.target_names)  # Displays the names of the target variables.

Example 2:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_nansha

    # Load the dataset as a (data, target) tuple
    data, target = load_nansha(return_X_y=True)
    print(data.shape)  # Assuming (1000, 21) for 1000 instances and 21 features.
    print(target.shape)  # Assuming (1000,)

Example 3:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_nansha

    # Load the dataset with specific years of land subsidence data
    b = load_nansha(key='subsidence', years='2015 2018', drop_display_rate=False)
    print(b.frame.head())  # Displays the first five rows of the land subsidence data.
    print(b.target_names)  # Displays the names of the target variables for the specified years.

.. note::
   The :func:`gofast.datasets.load.load_nansha` function provides comprehensive data for hydrogeological and geotechnical studies. 
   It supports various configurations, including selecting specific years of data, shuffling, and sampling, making it a versatile tool 
   for both research and practical applications.


load_bagoue
~~~~~~~~~~~

The :func:`gofast.datasets.load.load_bagoue` function loads the Bagoue dataset. This dataset is a classic multi-class 
classification dataset, used for various predictive modeling tasks.

Example 1:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_bagoue

    # Load the dataset as a Bunch object
    b = load_bagoue()
    print(b.frame.head())  # Displays the first five rows of the dataset.
    print(b.target_names)  # Displays the names of the target variables.

Example 2:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_bagoue

    # Load the dataset as a (data, target) tuple
    data, target = load_bagoue(return_X_y=True)
    print(data.shape)  # Assuming (150, 4) for 150 instances and 4 features.
    print(target.shape)  # Assuming (150,)

Example 3:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_bagoue

    # Load the dataset and split into training and test sets
    X, Xt, y, yt = load_bagoue(split_X_y=True, test_ratio=0.3, as_frame=True)
    print(X.head())  # Displays the first five rows of the training data.
    print(y.head())  # Displays the first five rows of the training target variable.

.. note::
   The :func:`gofast.datasets.load.load_bagoue` function is ideal for various classification tasks in predictive modeling. 
   It supports multiple configurations, including returning data in different formats, splitting data into training and test sets, 
   and handling specific target variables, providing a versatile and robust tool for machine learning applications.
   
load_iris
~~~~~~~~~

The :func:`gofast.datasets.load.load_iris` function loads and returns the iris dataset, a classic and straightforward multi-class classification dataset. This dataset is often used for testing and validating machine learning models.

Example 1:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_iris

    # Load the dataset as a Bunch object
    data = load_iris()
    print(data.frame.head())  # Displays the first five rows of the dataset.
    print(data.target[:5])  # Displays the first five rows of the target variable.

Example 2:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_iris

    # Load the dataset as a (data, target) tuple
    data, target = load_iris(return_X_y=True)
    print(data.shape)  # (150, 4) for 150 instances and 4 features.
    print(target.shape)  # (150,)

Example 3:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.datasets.load import load_iris

    # Load the dataset as a pandas DataFrame
    df, target = load_iris(return_X_y=True, as_frame=True)
    print(df.head())  # Displays the first five rows of the feature data.
    print(target.head())  # Displays the first five rows of the target data.

.. note:: 
   The function ensures flexibility by providing options to return the data as numpy arrays or pandas DataFrames. 
   This facilitates easy manipulation and analysis of the data.

load_mxs
~~~~~~~~

The :func:`gofast.datasets.load.load_mxs` function loads the dataset after implementing the mixture learning strategy (MXS). 
The dataset is composed of 11 boreholes merged with multiple-targets that can be used for a classification problem collected in 
Niming county.

Example 1:

.. code-block:: python

    from gofast.datasets.load import load_mxs

    # Load the entire dataset as a Bunch object
    data = load_mxs()
    print(data.frame.head())  # Displays the first five rows of the dataset.
    print(data.target_names)  # Displays the names of the target variables.

Example 2:

.. code-block:: python

    from gofast.datasets.load import load_mxs

    # Load the dataset as a (data, target) tuple
    data, target = load_mxs(return_X_y=True)
    print(data.shape)  # (n_samples, n_features) depending on the data size.
    print(target.shape)  # (n_samples,)

Example 3:

.. code-block:: python

    from gofast.datasets.load import load_mxs

    # Load the dataset with specific preprocessing and sampling
    data, target = load_mxs(key='scale', samples=0.5, return_X_y=True)
    print(data.shape)  # (n_samples, n_features) after scaling and sampling.
    print(target.shape)  # (n_samples,)

Example 4:

.. code-block:: python

    from gofast.datasets.load import load_mxs

    # Load the dataset, split into training and testing sets
    X_train, X_test, y_train, y_test = load_mxs(split_X_y=True, test_ratio=0.3, return_X_y=True)
    print(X_train.shape)  # Training data shape.
    print(X_test.shape)  # Testing data shape.

.. note::
   The :func:`gofast.datasets.load.load_mxs` function supports various configurations for loading and preprocessing the dataset. 
   It can handle different data formats, apply preprocessing steps, and split the data into training and testing sets.
   
load_forensic
~~~~~~~~~~~~~

The :func:`gofast.datasets.load.load_forensic` function provides access to a forensic dataset, which includes public opinion and knowledge regarding DNA databases, their potential use in criminal investigations, and concerns related to privacy and misuse. The dataset is derived from a study on the need for a forensic DNA database in the Sahel region. It comes in two forms: raw and preprocessed. The raw data includes the original responses, while the preprocessed data contains encoded and cleaned information.

Example 1:

.. code-block:: python

    from gofast.datasets.load import load_forensic

    # Load the raw dataset as a Bunch object
    forensic_data = load_forensic(key='raw', as_frame=True)
    print(forensic_data.frame.head())  # Displays the first five rows of the raw dataset.

Example 2:

.. code-block:: python

    from gofast.datasets.load import load_forensic

    # Load the preprocessed dataset, excluding the message column
    forensic_data = load_forensic(key='preprocessed', as_frame=True, 
                                  exclude_message_column=True, 
                                  exclude_vectorized_features=True)
    print(forensic_data.frame.head())  # Displays the first five rows of the preprocessed dataset.

Example 3:

.. code-block:: python

    from gofast.datasets.load import load_forensic

    # Load the dataset as a (data, target) tuple
    data, target = load_forensic(return_X_y=True)
    print(data.shape)  # (n_samples, n_features) depending on the dataset size.
    print(target.shape)  # (n_samples,)

.. note:: 
   This function is designed to handle both raw and preprocessed data. It provides options to exclude specific columns and features, 
   split the data into training and testing sets, and return the data in various formats to suit diverse analytical needs.

load_jrs_bet
~~~~~~~~~~~~

The :func:`gofast.datasets.load.load_jrs_bet` function loads and returns the JRS BET dataset. This dataset is designed for analyzing 
betting strategies and outcomes, with options to select different versions of the data based on learning methods: raw, classic machine learning, or neural network processed data.

Example 1:

.. code-block:: python

    from gofast.datasets.load import load_jrs_bet

    # Load the raw dataset as a Bunch object
    bet_data = load_jrs_bet(key='raw', as_frame=True)
    print(bet_data.frame.head())  # Displays the first five rows of the raw dataset.

Example 2:

.. code-block:: python

    from gofast.datasets.load import load_jrs_bet

    # Load the classic machine learning version of the dataset
    bet_data = load_jrs_bet(key='classic', as_frame=True)
    print(bet_data.frame.head())  # Displays the first five rows of the classic machine learning dataset.

Example 3:

.. code-block:: python

    from gofast.datasets.load import load_jrs_bet

    # Load the neural network processed version of the dataset
    bet_data = load_jrs_bet(key='neural', as_frame=True)
    print(bet_data.frame.head())  # Displays the first five rows of the neural network processed dataset.

Example 4:

.. code-block:: python

    from gofast.datasets.load import load_jrs_bet

    # Load the dataset as a (data, target) tuple
    data, target = load_jrs_bet(return_X_y=True)
    print(data.shape)  # (n_samples, n_features) depending on the dataset size.
    print(target.shape)  # (n_samples,)

.. note:: 
   The function supports various configurations for loading the dataset. It can handle raw data, classic machine learning data, 
   and neural network processed data. It also provides options to split the data into training and testing sets, exclude specific 
   columns, and return the data in different formats.



