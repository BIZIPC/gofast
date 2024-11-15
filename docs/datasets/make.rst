.. _make:

make
====

.. currentmodule:: gofast.datasets.make

The :mod:`gofast.datasets.make` module focuses on generating synthetic datasets for various domains such as geophysical surveys, 
medical diagnostics, retail store analysis, and more. This module provides functions to create datasets with different features and 
complexities, useful for testing and validating machine learning models.

Key Features
------------
- **Geophysical Survey Datasets**:
  Generate synthetic datasets for geophysical surveys, including well logging, electrical resistivity tomography (ERT), transient 
  electromagnetic (TEM) surveys, and more.
  
  - :func:`~gofast.datasets.make.make_well_logging`: Creates a synthetic dataset for geophysical well logging.
  - :func:`~gofast.datasets.make.make_ert`: Generates a dataset for electrical resistivity tomography surveys.
  - :func:`~gofast.datasets.make.make_tem`: Generates a dataset for transient electromagnetic surveys.
  - :func:`~gofast.datasets.make.make_erp`: Generates a dataset for geophysical analysis with resistivity values.

- **Medical Diagnostic Datasets**:
  Create synthetic medical datasets for testing diagnostic models and algorithms.
  
  - :func:`~gofast.datasets.make.make_medical_diagnosis`: Generates a medical dataset with diverse features for diagnostic purposes.
  - :func:`~gofast.datasets.make.make_elogging`: Generates a dataset of simulated logging data with timestamps and log levels.

- **Retail and Marketing Datasets**:
  Generate synthetic datasets for retail store analysis and marketing campaigns.
  
  - :func:`~gofast.datasets.make.make_retail_store`: Creates a retail store dataset with customer data and shopping habits.
  - :func:`~gofast.datasets.make.make_gadget_sales`: Generates a dataset of gadget sales data after the Christmas holiday.

- **Hydrogeological and Mining Datasets**:
  Create synthetic datasets for hydrogeological and mining operations.
  
  - :func:`~gofast.datasets.make.make_drill_ops`: Generates hydrogeological data for drilling operations.
  - :func:`~gofast.datasets.make.make_mining_ops`: Creates a dataset for mining operations with various features.

- **Environmental and Climate Change Datasets**:
  Generate synthetic datasets for studying environmental factors and climate change impacts.
  
  - :func:`~gofast.datasets.make.make_cc_factors`: Generates a dataset simulating factors contributing to climate change.
  - :func:`~gofast.datasets.make.make_water_demand`: Creates a synthetic water demand needs dataset.

Common Key Parameters
---------------------
- `<samples>`: The number of entries (rows) in the dataset.
- `<as_frame>`: If `True`, the data is returned as a pandas DataFrame. If `False`, a Bunch object is returned. By default, all the
   functions in :mod:`~gofast.datasets.make` module returns a dataframe.
- `<return_X_y>`: If `True`, returns `(data, target)` instead of a Bunch object.
- `<split_X_y>`: If `True`, the dataset is split into training and testing sets according to the specified test size ratio.
- `<target_names>`: The name of the target column(s) to retrieve.
- `<test_size>`: The proportion of the dataset to be used as the test set.
- `<noise>`: The proportion of entries in the dataset to randomly replace with NaN values.
- `<seed>`: Seed for the random number generator to ensure reproducibility.

Data Handling and Attributes
----------------------------
By default, the `as_frame` parameter is `True`. However, if set to `False`, each function returns a Bunch object where information 
like `data`, `target`, `frame`, `target_names`, `feature_names`, `feature_units`, `DESCR`, and `FDESCR` can be retrieved as attributes.

- `data`: The data matrix containing the features.
- `target`: The target values for classification or regression tasks.
- `frame`: A DataFrame that includes both the data and the target.
- `target_names`: A list of names for the target columns.
- `feature_names`: A list of names for the feature columns.
- `feature_units`: A dictionary describing the units of measurement for each feature.
- `DESCR`: A detailed description of the dataset.
- `FDESCR`: A detailed description of the dataset features.

Here is an example of how to use the Bunch object:

.. code-block:: python

    from gofast.datasets.make import make_drill_ops

    drill_ops = make_drill_ops(as_frame=False)

    drill_ops
    # Output: <Bunch object with keys: data, target, frame, target_names, feature_names, feature_units, DESCR, FDESCR>

    print(drill_ops.DESCR)
    # Output:
    # |==============================================================================|
    # |                               Dataset Overview                               |
    # |------------------------------------------------------------------------------|
    # | Generates a synthetic dataset tailored for drilling operations, specifically |
    # | designed for deep mining and hydrogeological exploration. This dataset       |
    # | includes a variety of hydrogeological parameters crucial for understanding   |
    # | subsurface conditions and planning drilling activities. The generated data   |
    # |                                 ...                                          |
    # | and mining.                                                                  |
    # |==============================================================================|

    print(drill_ops.FDESCR)
    # Output:
    # =================================================================================
    # |                               Dataset Features                               |
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # |Name                              | Description                               |
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # |porosity                          | Porosity (%): Represents the percentage of|
    # |                                  | void spaces in the rock or soil,          |
    # |                                  | indicating its ability to hold water. High|
    # |                                  | porosity values imply more space for fluid|
    # |                                  | storage, which is crucial for             |
    # |                                  | understanding water reserves and the      |
    # |                                  | potential for resource extraction.        |
    # .                                  .                                           .
    # |permeability                      | Permeability (Darcy or millidarcy):       |
    # |         ...                      |             ...                           |
    # |                                  | environment.                              |
    # =================================================================================

Function Descriptions
---------------------
Below are the descriptions and usage examples for each function in the 
:mod:`gofast.datasets.make` module:

make_data
~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_data` function generates synthetic datasets for both classification and regression tasks, providing options to split 
the data into training and test sets. This versatile function allows users to customize various parameters to create datasets of different complexities and types, making it ideal for machine learning experimentation and model testing.

Key Parameters
--------------
- `n_samples (int)`: Number of samples to generate. Default is 100.
- `n_features (int)`: Number of features for each sample. Default is 5.
- `task (str)`: Type of task, either 'classification' or 'regression'. Default is 'classification'.
- `n_classes (int)`: Number of classes for classification tasks. Default is 2.
- `n_informative (int)`: Number of informative features for classification tasks. Default is 2.
- `n_clusters_per_class (int)`: Number of clusters per class for classification tasks. Default is 1.
- `n_redundant (int)`: Number of redundant features for classification tasks. Default is 0.
- `n_repeated (int)`: Number of repeated features for classification tasks. Default is 0.
- `random_state (int)`: Seed for the random number generator. Default is 42.
- `test_size (float)`: Proportion of the dataset to include in the test split. Default is 0.3.
- `shuffle (bool)`: Whether or not to shuffle the data before splitting. Default is True.
- `noise (float)`: Standard deviation of Gaussian noise added to the output (for regression tasks). Default is 0.0.

Examples of Application
------------------------
The :func:`~gofast.datasets.make.make_data` function is highly useful for generating synthetic datasets that can be used to test and validate machine learning models. By allowing extensive customization, it helps in creating realistic datasets that mimic various real-world scenarios, thus enabling robust testing and experimentation.

**Example 1**: Classification Task

Generate a synthetic dataset for a classification task with 150 samples, 4 features, and 3 classes. The dataset is split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_data
    
    X_train, X_test, y_train, y_test = make_data(task='classification', 
                                                 n_samples=150, 
                                                 n_features=4, 
                                                 n_classes=3, 
                                                 test_size=0.2, 
                                                 random_state=7, 
                                                 split_X_y=True)

**Example 2**: Regression Task

Generate a synthetic dataset for a regression task with 200 samples and 6 features. Gaussian noise is added to the output to simulate real-world data variations.

.. code-block:: python

    from gofast.datasets.make import make_data
    
    X, y = make_data(task='regression', 
                     n_samples=200, 
                     n_features=6, 
                     noise=0.1, 
                     random_state=8)
                     
make_classification
~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_classification` function generates synthetic classification data for testing classification 
algorithms. This function supports multilabel classification tasks and allows control over the number of classes, features, and various 
options for data scaling.

Key Parameters
^^^^^^^^^^^^^^
- `n_samples (int)`: Number of samples to generate. Default is 100.
- `n_features (int)`: Number of features for each sample. Default is 20.
- `n_classes (int)`: Number of distinct classes or labels in the dataset. Default is 2.
- `n_labels (int)`: Number of labels per instance for multilabel classification. Default is 1.
- `noise (float)`: Standard deviation of Gaussian noise added to the output. Default is 0.0.
- `bias (float)`: Bias term to be added to the output. Default is 0.0.
- `scale (str or None)`: Method used to scale the dataset. Options are 'standard', 'minmax', and 'normalize'. Default is None.
  The `scale` parameter allows you to choose from different scaling methods to standardize the dataset's features. This is crucial for ensuring that each feature 
  contributes equally to the model's performance. The available scaling methods are:
  
  **Standard Scaler:**
  \[ z = \frac{x - \mu}{\sigma} \]
  where \(\mu\) is the mean and \(\sigma\) is the standard deviation.

  **MinMax Scaler:**
  \[ z = \frac{x - \min(x)}{\max(x) - \min(x)} \]

  **Normalize:**
  \[ z = \frac{x}{||x||} \]
  where \(||x||\) is the Euclidean norm (L2 norm).

- `class_sep (float)`: Factor multiplying the hypercube size. Larger values spread out the classes. Default is 1.0.
- `n_informative (int)`: Number of informative features. Default is 2.
- `n_redundant (int)`: Number of redundant features. Default is 2.
- `n_repeated (int)`: Number of duplicated features. Default is 0.
- `n_clusters_per_class (int)`: Number of clusters per class. Default is 2.
- `weights (array-like of shape (n_classes,) or (n_classes - 1,) or None)`: Proportions of samples assigned to each class. Default is None.
- `flip_y (float)`: Fraction of samples whose class is assigned randomly. Default is 0.01.
- `hypercube (bool)`: If True, clusters are put on the vertices of a hypercube. Default is True.
- `shift (float or ndarray of shape (n_features,) or None)`: Shift features by the specified value. Default is 0.0.
- `length (int)`: Sum of the features (number of words if documents) drawn from a Poisson distribution. Default is 50.
- `allow_unlabeled (bool)`: If True, some instances might not belong to any class. Default is True.
- `sparse (bool)`: If True, return a sparse feature matrix. Default is False.
- `return_indicator (str or False)`: Format to return indicator matrix. Options are 'dense', 'sparse'. Default is 'dense'.
- `return_distributions (bool)`: If True, return the prior class probability and conditional probabilities of features given classes. Default is False.
- `return_X_y (bool)`: If True, returns a tuple (X, y) instead of a single object. Default is False.
- `as_frame (bool)`: If True, the data is returned as a pandas DataFrame. Default is True.
- `feature_columns, target_columns (list of str or None)`: Custom names for the feature and target columns. Default is None.
- `nan_percentage (float or None)`: Percentage of values to be replaced with NaN in each column. Default is None.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_classification` function is highly useful for creating artificial datasets for classification tasks. It allows the simulation of datasets with varying degrees of class separability, making it suitable for testing the robustness of classification models. The function supports the addition of Gaussian noise to simulate real-world data and also supports the generation of multilabel datasets.

**Example 1**: Standard Classification Task

Generate a synthetic dataset for a standard classification task with 100 samples, 25 features, and 2 classes. The data is scaled using the standard scaler.

.. code-block:: python

    from gofast.datasets import make_classification
    
    X, y = make_classification(n_samples=100, 
                               n_features=25, 
                               scale='standard', 
                               n_classes=2, 
                               class_sep=2.0)
    print(X.shape, y.shape)

**Example 2**: Multiclass Classification Task

Generate a synthetic dataset for a multiclass classification task with 200 samples, 10 features, and 3 classes. The dataset includes 2 clusters per class and is controlled by a random state for reproducibility.

.. code-block:: python

    from gofast.datasets import make_classification
    
    X, y = make_classification(n_samples=200, 
                               n_features=10, 
                               n_classes=3, 
                               n_clusters_per_class=2, 
                               random_state=42)
    print(X[:5], y[:5])

make_regression
~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_regression` function generates synthetic regression data with customizable scaling and regression patterns. This function is designed to create datasets that are ideal for evaluating and testing different regression algorithms.

Key Parameters
^^^^^^^^^^^^^^
- `n_samples (int)`: Number of samples to generate. Default is 70.
- `n_features (int)`: Number of features for each sample. Default is 7.
- `noise (float)`: Standard deviation of Gaussian noise added to the output. Default is 0.1.
- `bias (float)`: Bias term to be added to the output. Default is 0.0.
- `scale (str or None)`: Method used to scale the dataset. Options are 'standard', 'minmax', and 'normalize'. Default is None.
  
  The `scale` parameter allows you to choose from different scaling methods to standardize the dataset's features. This is crucial for ensuring that each feature contributes equally to the model's performance. The available scaling methods are:

  **Standard Scaler:**
  \[ z = \frac{x - \mu}{\sigma} \]
  where \(\mu\) is the mean and \(\sigma\) is the standard deviation.

  **MinMax Scaler:**
  \[ z = \frac{x - \min(x)}{\max(x) - \min(x)} \]

  **Normalize:**
  \[ z = \frac{x}{||x||} \]
  where \(||x||\) is the Euclidean norm (L2 norm).

- `regression_type (str)`: Type of regression pattern to simulate. Options include 'linear', 'quadratic', 'cubic', 'exponential', 'logarithmic', 'sinusoidal', and 'step'. Default is 'linear'.
- `as_frame (bool)`: If True, the data is returned as a pandas DataFrame. Default is True.
- `return_X_y (bool)`: If True, returns a tuple (data, target) instead of a single object. Default is False.
- `split_X_y (bool)`: Whether to split the dataset into training and testing sets based on `test_size`. Default is False.
- `test_size (float)`: Proportion of the dataset to include in the test split. Default is 0.3.
- `target_indices (list or None)`: Indices of target features to be extracted. If specified, these columns are removed from the returned 'X' and included in 'y'. Default is None.
- `nan_percentage (float or None)`: Percentage of values to be replaced with NaN in each column. Default is None.
- `feature_columns, target_columns (list of str or None)`: Custom names for the feature and target columns. Default is None.
- `seed (int, np.random.RandomState instance, or None)`: Determines random number generation for dataset creation. Default is None.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_regression` function is useful for testing regression models by generating data with known 
properties. Different regression types allow simulation of various real-world scenarios. The scaling options help in preparing data that 
mimics different data distributions.

**Example 1**: Quadratic Regression Task

Generate a synthetic dataset for a quadratic regression task with 100 samples and 2 features. The data is scaled using the standard 
scaler.

.. code-block:: python

    from gofast.datasets import make_regression
    
    X, y = make_regression(n_samples=100, 
                           n_features=2, 
                           scale='standard', 
                           regression_type='quadratic')
    print(X.shape, y.shape)

**Example 2**: Sinusoidal Regression Task

Generate a synthetic dataset for a sinusoidal regression task with 150 samples and 3 features. Gaussian noise is added to the output, 
and the dataset is controlled by a random state for reproducibility.

.. code-block:: python

    from gofast.datasets import make_regression
    
    X, y = make_regression(n_samples=150, 
                           n_features=3, 
                           noise=0.05, 
                           regression_type='sinusoidal', 
                           seed=42)
    print(X[:5], y[:5])

This function is useful for testing regression models by generating data with known properties. Different regression types allow 
simulation of various real-world scenarios. The scaling options help in preparing data that mimics different data distributions.

make_social_media_comments
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_social_media_comments` function generates a synthetic dataset of social media comments. This 
function creates a DataFrame containing simulated social media comments, including features like comment text, timestamp, username, 
and number of likes. The function is useful for creating artificial datasets for social media analysis tasks. It allows the 
simulation of datasets with various features that are commonly found in social media data, making it suitable for testing the 
robustness of text processing and sentiment analysis models.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_social_media_comments` function is highly useful for generating synthetic social media datasets that can be used to test and validate text processing and sentiment analysis models. By allowing extensive customization, it helps in creating realistic datasets that mimic various real-world scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate 100 Social Media Comments

Generate a synthetic dataset with 100 social media comments. The dataset includes features such as username, comment text, timestamp, and number of likes.

.. code-block:: python

    from gofast.datasets.make import make_social_media_comments
    
    df = make_social_media_comments(samples=100, seed=42)
    print(df.head())

**Example 2**: Generate 200 Social Media Comments with Data Splitting

Generate a synthetic dataset with 200 social media comments and split it into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_social_media_comments
    
    X, Xt, y, yt = make_social_media_comments(samples=200, 
                                              split_X_y=True, 
                                              test_size=0.3, 
                                              seed=24)
    print(X.head(), Xt.head())


make_african_demo
~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_african_demo` function generates a dataset for African demography from 1960 to the present. This function creates a DataFrame with demographic data for specified African countries, simulating population size, birth rate, death rate, urbanization rate, and GDP per capita for each country and year.

Key Parameters
^^^^^^^^^^^^^^
- `start_year (int)`: The starting year for the dataset. Default is 1960.
- `end_year (int)`: The ending year for the dataset. Default is 2020.
- `countries (int or list of str)`: A single integer or a list of country names from Africa to be included in the dataset.
- `n_samples (int, optional)`: If provided, specifies a desired total number of samples in the dataset, adjusting the number of years to meet this target.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_african_demo` function is useful for generating synthetic demographic datasets that can be used 
to test and validate models in fields such as population studies, public health, and economic analysis. By allowing extensive 
customization, it helps in creating realistic datasets that mimic various real-world scenarios, thus enabling robust testing and 
experimentation.

**Example 1**: Generate African Demographic Data

Generate a synthetic dataset for African demography from 1960 to 2020 for selected countries.

.. code-block:: python

    from gofast.datasets.make import make_african_demo
    
    start_year = 1960
    end_year = 2020
    countries = ['Nigeria', 'Egypt', 'South Africa']
    
    demography_data = make_african_demo(start_year=start_year, 
                                        end_year=end_year, 
                                        countries=countries)
    print(demography_data.head())

**Example 2**: Generate African Demographic Data with Sample Limit

Generate a synthetic dataset for African demography with a sample limit, adjusting the number of years to fit the desired number of 
samples.

.. code-block:: python

    from gofast.datasets.make import make_african_demo
    
    n_samples = 500
    
    demography_data = make_african_demo(n_samples=n_samples)
    print(demography_data.head())


This function generates artificial datasets for demographic analysis in African countries. It is intended for simulation or testing
purposes only and does not represent real demographic statistics. The generated data includes features such as population size, 
birth rate, death rate, urbanization rate, and GDP per capita.

make_agronomy_feedback
~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_agronomy_feedback` function generates an agronomy dataset including information about crop cultivation and pesticide usage. This function creates a DataFrame with data for multiple farms over several years, including details about the type of crop grown, soil pH, temperature, rainfall, types and amounts of pesticides used, and crop yield.

Key Parameters
^^^^^^^^^^^^^^
- `num_years (int)`: The number of years for which data is generated.
- `n_specimens (int)`: Number of different crop and pesticide types to include in the dataset.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_agronomy_feedback` function is useful for generating synthetic agronomy datasets that can be used to test and validate models in agriculture and environmental studies. By allowing extensive customization, it helps in creating realistic datasets that mimic various real-world scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate Agronomy Feedback Data

Generate a synthetic dataset for agronomy feedback with 100 samples over 5 years and 3 different types of crops and pesticides.

.. code-block:: python

    from gofast.datasets.make import make_agronomy_feedback
    
    samples = 100
    num_years = 5
    n_specimens = 3
    
    agronomy_data = make_agronomy_feedback(samples=samples, 
                                           num_years=num_years, 
                                           n_specimens=n_specimens)
    print(agronomy_data.head())

**Example 2**: Generate Agronomy Feedback Data with Data Splitting

Generate a synthetic dataset for agronomy feedback with 200 samples over 10 years, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_agronomy_feedback
    
    samples = 200
    num_years = 10
    
    X, Xt, y, yt = make_agronomy_feedback(samples=samples, 
                                          num_years=num_years, 
                                          n_specimens=5, 
                                          split_X_y=True, 
                                          test_size=0.3, 
                                          seed=42)
    print(X.head(), Xt.head())

This function generates artificial datasets for agronomy studies and is intended for simulation or testing purposes only. 
The generated data includes features such as crop type, soil pH, temperature, rainfall, pesticide type and amount, and crop yield. 
Real-world data collection would involve more detailed and precise measurements, and the interaction between these variables can 
be quite complex.

make_mining_ops
~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_mining_ops` function generates a dataset for mining operations. This function creates a DataFrame with simulated data reflecting various aspects of mining operations, including geospatial coordinates for drilling, types and concentrations of ore, details of drilling and blasting operations, information about mining equipment, and daily production figures.

Key Parameters
^^^^^^^^^^^^^^
- `samples (int)`: The number of entries (rows) in the dataset.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_mining_ops` function is useful for generating synthetic datasets that can be used to test and 
validate models in the context of mining operations. By allowing extensive customization, it helps in creating realistic datasets 
that mimic various real-world scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate Mining Operations Data

Generate a synthetic dataset for mining operations with 1000 samples.

.. code-block:: python

    from gofast.datasets.make import make_mining_ops
    
    samples = 1000
    
    mining_data = make_mining_ops(samples=samples)
    print(mining_data.head())

**Example 2**: Generate Mining Operations Data with Data Splitting

Generate a synthetic dataset for mining operations with 2000 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_mining_ops
    
    samples = 2000
    
    X, Xt, y, yt = make_mining_ops(samples=samples, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())


This function generates artificial datasets for mining operations and is intended for simulation or testing purposes only. The 
generated data includes features such as geospatial coordinates, ore types, drilling and blasting details, equipment information, and 
daily production figures.

make_sounding
~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_sounding` function generates a complex dataset for geophysical sounding, typically used in ERT or 
seismic surveys. This function creates a DataFrame with data for multiple survey points, each with a specified number of subsurface 
layers, including layer depth, electrical resistivity, and seismic velocity.

Key Parameters
^^^^^^^^^^^^^^
- `num_layers (int)`: The number of subsurface layers to simulate for each survey point.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_sounding` function is useful for generating synthetic datasets that can be used to test and validate models related to geophysical sounding analysis. By allowing extensive customization, it helps in creating realistic datasets that mimic various real-world scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate Geophysical Sounding Data

Generate a synthetic dataset for geophysical sounding with 100 samples and 5 subsurface layers.

.. code-block:: python

    from gofast.datasets.make import make_sounding
    
    samples = 100
    num_layers = 5
    
    sounding_data = make_sounding(samples=samples, num_layers=num_layers)
    print(sounding_data.head())

**Example 2**: Generate Geophysical Sounding Data with Data Splitting

Generate a synthetic dataset for geophysical sounding with 200 samples and 10 subsurface layers, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_sounding
    
    samples = 200
    num_layers = 10
    
    X, Xt, y, yt = make_sounding(samples=samples, num_layers=num_layers, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())


This function generates artificial datasets for geophysical sounding and is intended for simulation or testing purposes only. The 
generated data includes features such as layer depth, electrical resistivity, and seismic velocity.

make_medical_diagnosis
~~~~~~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_medical_diagnosis` function generates a synthetic medical dataset with diverse features. 
This function creates a DataFrame containing a variety of medical data points, such as demographic information, vital signs, laboratory 
test results, medical history, and lifestyle factors. The generated data is random and should be used for simulation or testing purposes 
only.

Key Parameters
^^^^^^^^^^^^^^
- `samples (int)`: The number of entries (patients) in the dataset.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_medical_diagnosis` function is highly useful for creating artificial medical datasets for testing 
and validating medical diagnostic models. By allowing extensive customization, it helps in creating realistic datasets that mimic 
various real-world scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate Medical Diagnosis Data

Generate a synthetic dataset for medical diagnosis with 1000 samples.

.. code-block:: python

    from gofast.datasets.make import make_medical_diagnosis
    
    samples = 1000
    
    medical_data = make_medical_diagnosis(samples=samples)
    print(medical_data.head())

**Example 2**: Generate Medical Diagnosis Data with Data Splitting

Generate a synthetic dataset for medical diagnosis with 2000 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_medical_diagnosis
    
    samples = 2000
    
    X, Xt, y, yt = make_medical_diagnosis(samples=samples, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate Medical Diagnosis Data with Custom Target Names

Generate a synthetic dataset for medical diagnosis with 1500 samples and specify custom target names.

.. code-block:: python

    from gofast.datasets.make import make_medical_diagnosis
    
    samples = 1500
    target_names = ['history_of_diabetes', 'history_of_hypertension']
    
    medical_data = make_medical_diagnosis(samples=samples, target_names=target_names)
    print(medical_data.head())

This function generates artificial medical datasets and is intended for simulation or testing purposes only. The generated data includes 
features such as demographic information, vital signs, laboratory test results, medical history, and lifestyle factors.

make_well_logging
~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_well_logging` function generates a synthetic dataset for geophysical well logging. This function 
creates a DataFrame simulating typical well logging data, which is often used in subsurface geological investigations. Each row 
represents a set of measurements at a specific depth, with the depth intervals defined by the user.

Key Parameters
^^^^^^^^^^^^^^
- `depth_start (float)`: The starting depth for the well logging in meters.
- `depth_end (float)`: The ending depth for the well logging in meters.
- `depth_interval (float)`: The interval between depth measurements in meters.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_well_logging` function is useful for generating synthetic well logging data for testing and 
validating geophysical analysis models. It allows for the simulation of datasets that mimic various real-world well logging scenarios, 
thus enabling robust testing and experimentation.

**Example 1**: Generate Well Logging Data

Generate a synthetic dataset for well logging from 0 to 200 meters with a 0.5-meter interval.

.. code-block:: python

    from gofast.datasets.make import make_well_logging
    
    depth_start = 0.0
    depth_end = 200.0
    depth_interval = 0.5
    
    well_logging_data = make_well_logging(depth_start=depth_start, depth_end=depth_end, depth_interval=depth_interval)
    print(well_logging_data.head())

**Example 2**: Generate Well Logging Data with Data Splitting

Generate a synthetic dataset for well logging with depth measurements from 0 to 300 meters at 1-meter intervals, split into training 
and test sets.

.. code-block:: python

    from gofast.datasets.make import make_well_logging
    
    depth_start = 0.0
    depth_end = 300.0
    depth_interval = 1.0
    
    X, Xt, y, yt = make_well_logging(depth_start=depth_start, depth_end=depth_end, depth_interval=depth_interval, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate Well Logging Data with Custom Target Names

Generate a synthetic dataset for well logging with custom target names.

.. code-block:: python

    from gofast.datasets.make import make_well_logging
    
    depth_start = 0.0
    depth_end = 150.0
    depth_interval = 0.5
    target_names = ['neutron_porosity']
    
    well_logging_data = make_well_logging(depth_start=depth_start, depth_end=depth_end, depth_interval=depth_interval, target_names=target_names)
    print(well_logging_data.head())

This function generates artificial well logging datasets and is intended for simulation or testing purposes only. The generated data 
includes measurements for gamma-ray, resistivity, neutron porosity, and density.

make_ert
~~~~~~~~

The :func:`~gofast.datasets.make.make_ert` function generates a synthetic dataset for electrical resistivity tomography (ERT) based on the specified equipment type. This function creates a DataFrame with synthetic data representing an ERT survey, including electrode positions, cable lengths, resistivity measurements, and battery voltages (when applicable).

Key Parameters
^^^^^^^^^^^^^^
- `equipment_type (str)`: The type of ERT equipment used. Should be one of 'SuperSting R8', 'Ministing or Sting R1', or 'OhmMapper'.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_ert` function is highly useful for generating synthetic ERT datasets for testing and validating geophysical analysis models. By allowing customization of the equipment type, it helps in creating realistic datasets that mimic various real-world ERT survey scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate ERT Data

Generate a synthetic dataset for ERT using the 'SuperSting R8' equipment type with 100 samples.

.. code-block:: python

    from gofast.datasets.make import make_ert
    
    samples = 100
    equipment_type = 'SuperSting R8'
    
    ert_data = make_ert(samples=samples, equipment_type=equipment_type)
    print(ert_data.head())

**Example 2**: Generate ERT Data with Data Splitting

Generate a synthetic dataset for ERT using the 'Ministing or Sting R1' equipment type with 200 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_ert
    
    samples = 200
    equipment_type = 'Ministing or Sting R1'
    
    X, Xt, y, yt = make_ert(samples=samples, equipment_type=equipment_type, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate ERT Data with Custom Target Names

Generate a synthetic dataset for ERT using the 'OhmMapper' equipment type with custom target names.

.. code-block:: python

    from gofast.datasets.make import make_ert
    
    samples = 150
    equipment_type = 'OhmMapper'
    target_names = ['resistivity']
    
    ert_data = make_ert(samples=samples, equipment_type=equipment_type, target_names=target_names)
    print(ert_data.head())

This function generates artificial ERT datasets and is intended for simulation or testing purposes only. The generated data includes 
electrode positions, cable lengths, resistivity measurements, and battery voltages, depending on the specified equipment type.

make_tem
~~~~~~~~

The :func:`~gofast.datasets.make.make_tem` function generates a dataset for a Transient Electromagnetic (TEM) survey including equipment types. This function creates a DataFrame with synthetic geospatial and TEM survey data. It allows for the specification of ranges for latitude, longitude, time intervals, and TEM measurements.

Key Parameters
^^^^^^^^^^^^^^
- `lat_range (tuple of float)`: The range of latitude values (min_latitude, max_latitude).
- `lon_range (tuple of float)`: The range of longitude values (min_longitude, max_longitude).
- `time_range (tuple of float)`: The range of time intervals in milliseconds after the pulse (min_time, max_time).
- `measurement_range (tuple of float)`: The range of TEM measurements (min_measurement, max_measurement).

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_tem` function is useful for generating synthetic TEM survey data for testing and validating geophysical models. It allows for the creation of datasets that mimic various real-world TEM survey scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate TEM Data

Generate a synthetic dataset for a TEM survey with 500 samples, latitude range from 34.00 to 36.00, longitude range from -118.50 to -117.00, time range from 0.01 to 10.0 milliseconds, and measurement range from 100 to 10000 arbitrary units.

.. code-block:: python

    from gofast.datasets.make import make_tem
    
    samples = 500
    lat_range = (34.00, 36.00)
    lon_range = (-118.50, -117.00)
    time_range = (0.01, 10.0)
    measurement_range = (100, 10000)
    
    tem_data = make_tem(samples=samples, lat_range=lat_range, lon_range=lon_range, time_range=time_range, measurement_range=measurement_range)
    print(tem_data.head())

**Example 2**: Generate TEM Data with Data Splitting

Generate a synthetic dataset for a TEM survey with 1000 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_tem
    
    samples = 1000
    lat_range = (33.00, 37.00)
    lon_range = (-119.00, -116.00)
    time_range = (0.01, 15.0)
    measurement_range = (50, 15000)
    
    X, Xt, y, yt = make_tem(samples=samples, lat_range=lat_range, lon_range=lon_range, time_range=time_range, measurement_range=measurement_range, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate TEM Data with Custom Target Names

Generate a synthetic dataset for a TEM survey with custom target names.

.. code-block:: python

    from gofast.datasets.make import make_tem
    
    samples = 300
    lat_range = (35.00, 37.00)
    lon_range = (-119.50, -116.50)
    time_range = (0.05, 12.0)
    measurement_range = (200, 8000)
    target_names = ['tem_measurement']
    
    tem_data = make_tem(samples=samples, lat_range=lat_range, lon_range=lon_range, time_range=time_range, measurement_range=measurement_range, target_names=target_names)
    print(tem_data.head())

This function generates artificial TEM survey datasets and is intended for simulation or testing purposes only. The generated data 
includes latitude, longitude, time intervals, TEM measurements, and equipment types.

make_erp
~~~~~~~~

The :func:`~gofast.datasets.make.make_erp` function generates a dataset for geophysical analysis with easting, northing, longitude, latitude, positions, step, and resistivity values. This function creates a DataFrame with synthetic geospatial data, simulating sequential survey data.

Key Parameters
^^^^^^^^^^^^^^
- `lat_range (tuple of float)`: The range of latitude values (min_latitude, max_latitude).
- `lon_range (tuple of float)`: The range of longitude values (min_longitude, max_longitude).
- `resistivity_range (tuple of float)`: The range of resistivity values (min_resistivity, max_resistivity).

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_erp` function is useful for generating synthetic geophysical survey data for testing and validating geophysical analysis models. It allows for the creation of datasets that mimic various real-world geophysical survey scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate ERP Data

Generate a synthetic dataset for geophysical analysis with 1000 samples, latitude range from 34.00 to 36.00, longitude range from -118.50 to -117.00, and resistivity range from 10 to 1000 ohm-meters.

.. code-block:: python

    from gofast.datasets.make import make_erp
    
    samples = 1000
    lat_range = (34.00, 36.00)
    lon_range = (-118.50, -117.00)
    resistivity_range = (10, 1000)
    
    erp_data = make_erp(samples=samples, lat_range=lat_range, lon_range=lon_range, resistivity_range=resistivity_range)
    print(erp_data.head())

**Example 2**: Generate ERP Data with Data Splitting

Generate a synthetic dataset for geophysical analysis with 2000 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_erp
    
    samples = 2000
    lat_range = (33.00, 37.00)
    lon_range = (-119.00, -116.00)
    resistivity_range = (5, 2000)
    
    X, Xt, y, yt = make_erp(samples=samples, lat_range=lat_range, lon_range=lon_range, resistivity_range=resistivity_range, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate ERP Data with Custom Target Names

Generate a synthetic dataset for geophysical analysis with custom target names.

.. code-block:: python

    from gofast.datasets.make import make_erp
    
    samples = 500
    lat_range = (35.00, 37.00)
    lon_range = (-119.50, -116.50)
    resistivity_range = (20, 1500)
    target_names = ['resistivity']
    
    erp_data = make_erp(samples=samples, lat_range=lat_range, lon_range=lon_range, resistivity_range=resistivity_range, target_names=target_names)
    print(erp_data.head())

This function generates artificial geophysical survey datasets and is intended for simulation or testing purposes only. The generated 
data includes easting, northing, longitude, latitude, positions, steps, and resistivity values.

make_elogging
~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_elogging` function generates a dataset of simulated logging data. This function creates a DataFrame with synthetic log entries, including timestamps, log levels, and messages.

Key Parameters
^^^^^^^^^^^^^^
- `start_date (str)`: The start date for the logging data in 'YYYY-MM-DD' format.
- `end_date (str)`: The end date for the logging data in 'YYYY-MM-DD' format.
- `log_levels (list of str, optional)`: A list of log levels (e.g., ['INFO', 'WARNING', 'ERROR']). If None, defaults to ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'].

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_elogging` function is useful for generating synthetic logging data for testing and validating logging systems or analysis models. It allows for the creation of datasets that mimic various real-world logging scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate Logging Data

Generate a synthetic dataset for logging data with 100 samples, start date '2021-01-01', and end date '2021-01-31'.

.. code-block:: python

    from gofast.datasets.make import make_elogging
    
    start_date = '2021-01-01'
    end_date = '2021-01-31'
    samples = 100
    
    log_data = make_elogging(samples=samples, start_date=start_date, end_date=end_date)
    print(log_data.head())

**Example 2**: Generate Logging Data with Data Splitting

Generate a synthetic dataset for logging data with 200 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_elogging
    
    start_date = '2021-01-01'
    end_date = '2021-01-31'
    samples = 200
    
    X, Xt, y, yt = make_elogging(samples=samples, start_date=start_date, end_date=end_date, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate Logging Data with Custom Log Levels

Generate a synthetic dataset for logging data with custom log levels.

.. code-block:: python

    from gofast.datasets.make import make_elogging
    
    start_date = '2021-01-01'
    end_date = '2021-01-31'
    samples = 150
    log_levels = ['INFO', 'ERROR']
    
    log_data = make_elogging(samples=samples, start_date=start_date, end_date=end_date, log_levels=log_levels)
    print(log_data.head())

.. topic: notes 

    This function generates artificial logging datasets and is intended for simulation or testing purposes only. The generated data includes 
    timestamps, log levels, and messages.

make_gadget_sales
~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_gadget_sales` function generates a dataset of gadget sales data for girls and boys after the Christmas holiday. This function creates a DataFrame with synthetic sales data, including sale dates, gadget types, genders, and units sold.

Key Parameters
^^^^^^^^^^^^^^
- `start_date (str)`: The start date for the sales data in 'YYYY-MM-DD' format.
- `end_date (str)`: The end date for the sales data in 'YYYY-MM-DD' format.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_gadget_sales` function is useful for generating synthetic sales data for testing and validating sales analysis models. It allows for the creation of datasets that mimic various real-world sales scenarios, thus enabling robust testing and experimentation.

**Example 1**: Generate Gadget Sales Data

Generate a synthetic dataset for gadget sales data with 100 samples, start date '2021-12-26', and end date '2022-01-10'.

.. code-block:: python

    from gofast.datasets.make import make_gadget_sales
    
    start_date = '2021-12-26'
    end_date = '2022-01-10'
    samples = 100
    
    sales_data = make_gadget_sales(samples=samples, start_date=start_date, end_date=end_date)
    print(sales_data.head())

**Example 2**: Generate Gadget Sales Data with Data Splitting

Generate a synthetic dataset for gadget sales data with 500 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_gadget_sales
    
    start_date = '2021-12-26'
    end_date = '2022-01-10'
    samples = 500
    
    X, Xt, y, yt = make_gadget_sales(samples=samples, start_date=start_date, end_date=end_date, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate Gadget Sales Data with Custom Target Names

Generate a synthetic dataset for gadget sales data with custom target names.

.. code-block:: python

    from gofast.datasets.make import make_gadget_sales
    
    start_date = '2021-12-26'
    end_date = '2022-01-10'
    samples = 150
    target_names = ['units_sold']
    
    sales_data = make_gadget_sales(samples=samples, start_date=start_date, end_date=end_date, target_names=target_names)
    print(sales_data.head())

.. topic: notes 

   This function generates artificial gadget sales datasets and is intended for simulation or testing purposes only. The generated data includes sale dates, gadget types, genders, and units sold.

make_retail_store
~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_retail_store` function generates a retail store dataset for machine learning purposes with mixed data types. This function simulates a hypothetical scenario, such as customer data for a retail store, including features like age, income, shopping frequency, last purchase amount, preferred shopping category, and a binary target variable indicating whether the customer is likely to respond to a new marketing campaign.

Key Parameters
^^^^^^^^^^^^^^
- `samples (int)`: The number of entries (rows) in the dataset.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_retail_store` function is useful for generating synthetic retail customer data for testing and validating machine learning models. It allows for the creation of datasets that mimic various real-world retail scenarios, enabling robust testing and experimentation.

**Example 1**: Generate Retail Store Data

Generate a synthetic dataset for a retail store with 1000 samples.

.. code-block:: python

    from gofast.datasets.make import make_retail_store
    
    samples = 1000
    dataset = make_retail_store(samples=samples)
    print(dataset.head())

**Example 2**: Generate Retail Store Data with Data Splitting

Generate a synthetic dataset for a retail store with 500 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_retail_store
    
    samples = 500
    X, Xt, y, yt = make_retail_store(samples=samples, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate Retail Store Data with Custom Target Names

Generate a synthetic dataset for a retail store with custom target names.

.. code-block:: python

    from gofast.datasets.make import make_retail_store
    
    samples = 150
    target_names = ['likely_to_respond']
    
    dataset = make_retail_store(samples=samples, target_names=target_names)
    print(dataset.head())


This function generates artificial retail store datasets and is intended for simulation or testing purposes only. The generated data includes features such as age, income, shopping frequency, last purchase amount, preferred shopping category, and a binary target variable.

make_cc_factors
~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_cc_factors` function generates a synthetic dataset simulating factors contributing to climate change and their respective feedback mechanisms.

Key Parameters
^^^^^^^^^^^^^^
- `feedback_threshold (float or 'auto')`: Specifies the threshold used to classify feedback mechanisms as positive or negative. If set to 'auto', the threshold is dynamically calculated based on the data.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_cc_factors` function is useful for generating synthetic data for studying climate change factors and their feedback mechanisms. It allows for the creation of datasets that mimic various real-world climate scenarios, enabling robust testing and experimentation.

**Example 1**: Generate Climate Change Factors Data

Generate a synthetic dataset for climate change factors with 500 samples and a feedback threshold of 0.5.

.. code-block:: python

    from gofast.datasets.make import make_cc_factors
    
    samples = 500
    feedback_threshold = 0.5
    
    data, target = make_cc_factors(samples=samples, feedback_threshold=feedback_threshold, as_frame=True, return_X_y=True)
    print(data.head())
    print(target.head())

**Example 2**: Generate Climate Change Factors Data with Data Splitting

Generate a synthetic dataset for climate change factors with 1000 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_cc_factors
    
    samples = 1000
    
    X, Xt, y, yt = make_cc_factors(samples=samples, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate Climate Change Factors Data with Custom Noise

Generate a synthetic dataset for climate change factors with custom noise level.

.. code-block:: python

    from gofast.datasets.make import make_cc_factors
    
    samples = 700
    noise = 0.2
    
    data = make_cc_factors(samples=samples, noise=noise)
    print(data.head())

This function generates artificial datasets simulating the factors contributing to climate change and their respective feedback mechanisms. The generated data includes features such as greenhouse gas emissions, deforestation, fossil fuel consumption, and feedback mechanisms.

make_water_demand
~~~~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_water_demand` function generates a synthetic water demand needs dataset.
This function generates artificial water demand datasets and is intended for simulation or testing purposes only. The generated data includes features such as 
agricultural demand, industrial demand, and SDG6 challenges.
    
Key Parameters
^^^^^^^^^^^^^^
- `samples (int, default=700)`: Number of samples or data points in the dataset.
- `noise (float, optional)`: Probability of a value being missing in the dataset.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_water_demand` function is useful for generating synthetic water demand data for testing and 
validating machine learning models. It allows for the creation of datasets that mimic various real-world water demand scenarios, enabling robust testing and experimentation.

**Example 1**: Generate Water Demand Data

Generate a synthetic dataset for water demand needs with 700 samples.

.. code-block:: python

    from gofast.datasets.make import make_water_demand
    
    samples = 700
    dataset = make_water_demand(samples=samples)
    print(dataset.head())

**Example 2**: Generate Water Demand Data with Data Splitting

Generate a synthetic dataset for water demand needs with 500 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_water_demand
    
    samples = 500
    X, Xt, y, yt = make_water_demand(samples=samples, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate Water Demand Data with Custom Noise Level

Generate a synthetic dataset for water demand needs with custom noise level.

.. code-block:: python

    from gofast.datasets.make import make_water_demand
    
    samples = 1000
    noise = 0.2
    
    data = make_water_demand(samples=samples, noise=noise)
    print(data.head())

make_drill_ops
~~~~~~~~~~~~~~

The :func:`~gofast.datasets.make.make_drill_ops` function generates synthetic hydrogeological data tailored for drilling operations, 
specifically designed for deep mining and hydrogeological exploration. This data can be utilized for training and testing machine 
learning models, enabling predictive analyses and operational planning.

Key Parameters
^^^^^^^^^^^^^^
- `samples (int, optional, default=1000)`: The number of synthetic drilling operation samples to generate.
- `ops (str, optional, default='deep_mining')`: Specifies the type of drilling operations the synthetic data should represent, influencing the selection of default target parameters.

Examples of Application
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~gofast.datasets.make.make_drill_ops` function is useful for generating synthetic data for studying drilling operations. It allows for the creation of datasets that mimic various real-world drilling scenarios, enabling robust testing and experimentation.

**Example 1**: Generate Drilling Operations Data

Generate a synthetic dataset for drilling operations with 1000 samples.

.. code-block:: python

    from gofast.datasets.make import make_drill_ops
    
    samples = 1000
    dataset = make_drill_ops(samples=samples)
    print(dataset.head())

**Example 2**: Generate and Split Drilling Operations Data

Generate a synthetic dataset for drilling operations with 500 samples, split into training and test sets.

.. code-block:: python

    from gofast.datasets.make import make_drill_ops
    
    samples = 500
    X, Xt, y, yt = make_drill_ops(samples=samples, split_X_y=True, test_size=0.3, seed=42)
    print(X.head(), Xt.head())

**Example 3**: Generate Drilling Operations Data for Regular Operations

Generate a synthetic dataset for regular hydrogeological operations with custom noise level.

.. code-block:: python

    from gofast.datasets.make import make_drill_ops
    
    samples = 700
    ops = 'regular'
    noise = 0.2
    
    data = make_drill_ops(samples=samples, ops=ops, noise=noise)
    print(data.head())

.. note::
    The synthetic data generated by this function is intended for simulation and modeling purposes within the domain of hydrogeology 
    and deep mining. It is vital to consult with domain experts when applying insights derived from this data to real-world scenarios.
