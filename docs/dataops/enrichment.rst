.. _enrichment:

Enrichment
==========

.. currentmodule:: gofast.dataops.enrichment

The :mod:`gofast.dataops.enrichment` module encompasses functions that add value to the data 
through enrichment and summarization. These functions are designed to prepare, clean, and 
enhance datasets for various machine learning and analytical tasks.

Key Features
------------
- **Data Preparation**: Provides functionality for preparing datasets for machine learning models, including handling categorical encoding and NaN values.
- **Outlier Impact Analysis**: Offers methods to evaluate the impact of outliers on model performance, supporting both regression and classification tasks.
- **Data Augmentation**: Includes techniques to augment regression datasets through noise addition, resampling, synthetic data generation, and bootstrapping.
- **Summarization**: Features tools to generate concise summaries from text data using extractive summarization techniques.


Function Descriptions
---------------------

prepare_data
~~~~~~~~~~~~
Prepares the feature matrix X and target vector y from the provided DataFrame, optionally handling categorical encoding and NaN values 
according to specified policies.

This function prepares the feature matrix and target vector from a DataFrame. It can handle categorical encoding and manage NaN values 
based on the specified policy.

**Mathematical Formulation:**

1. **Categorical Encoding:**

   Categorical features can be converted to numeric codes using LabelEncoder:

   .. math::

       \text{encoded\_feature} = \text{LabelEncoder}(\text{categorical\_feature})

2. **Handling NaN values:**

   - **Propagate:** Retains NaN values.
   - **Omit:** Removes rows/columns with NaN values.
   - **Raise:** Raises an error if NaN values are found.

**Examples:**

.. code-block:: python

    import pandas as pd
    from gofast.dataops.enrichment import prepare_data

    # Simple usage with automatic encoding of categorical features
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'C'],
        'target': [0, 1, 0]
    })
    X, y = prepare_data(df, 'target', encode_categories=True)
    print(X)
    print(y)

.. code-block:: python

    # Handling NaN values by omitting them
    df_with_nan = pd.DataFrame({
        'feature1': [np.nan, 2, 3],
        'feature2': ['A', np.nan, 'C'],
        'target': [1, np.nan, 0]
    })
    X, y = prepare_data(df_with_nan, 'target', nan_policy='omit', verbose=True)
    print(X)
    print(y)

simple_extractive_summary
~~~~~~~~~~~~~~~~~~~~~~~~~
Generates a simple extractive summary from a list of texts. Function selects the sentence with the highest term frequency-inverse document frequency (TF-IDF) score and optionally returns its TF-IDF encoding.

**Mathematical Formulation:**

1. **TF-IDF Calculation:**

   Term Frequency (TF) for term :math:`t` in document :math:`d`:

   .. math::

       \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}

   Inverse Document Frequency (IDF) for term :math:`t`:

   .. math::

       \text{IDF}(t) = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents with term } t} \right)

   TF-IDF score for term :math:`t` in document :math:`d`:

   .. math::

       \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)

**Examples:**

.. code-block:: python

    from gofast.dataops.enrichment import simple_extractive_summary
    messages = [
        "Further explain the background and rationale for the study. Explain DNA in simple terms for non-scientists.",
        "Explain the objectives of the study which do not seem perceptible. THANKS",
        "We think this investigation is a good thing. In our opinion, it already allows the initiators to have an idea of what the populations think of the use of DNA in forensic investigations in Burkina Faso.",
        "And above all, know, through this survey, if these populations approve of the establishment of a possible genetic database in our country."
    ]
    summary, encoding = simple_extractive_summary(messages, encode=True)
    print(summary)
    print(encoding)

.. code-block:: python

    # Summarizing without encoding
    summary = simple_extractive_summary(messages, encode=False)
    print(summary)

outlier_performance_impact
~~~~~~~~~~~~~~~~~~~~~~~~~~
Assess the impact of outliers on the predictive performance of a model. Applicable for both regression and classification tasks, including multi-label targets.

This function evaluates a model's performance with and without outliers in the training data. The performance is measured using Mean Squared Error (MSE) for regression tasks and accuracy for classification tasks.

**Mathematical Formulation:**

For regression, the Mean Squared Error (MSE) is defined as:

.. math::

    \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

where :math:`y_i` is the actual value and :math:`\hat{y}_i` is the predicted value.

For classification, accuracy is defined as:

.. math::

    \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}

**Examples:**

.. code-block:: python

    import pandas as pd
    from gofast.dataops.enrichment import outlier_performance_impact

    # Example with regression data
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.rand(100)
    })
    mse_with_outliers, mse_without_outliers = outlier_performance_impact(df, 'target')
    print(mse_with_outliers, mse_without_outliers)

.. code-block:: python

    # Example with categorical data
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.choice([0, 1], size=100)
    })
    acc_with_outliers, acc_without_outliers = outlier_performance_impact(df, 'target', encode_categories=True)
    print(acc_with_outliers, acc_without_outliers)

enrich_data_spectrum
~~~~~~~~~~~~~~~~~~~~
Augment a regression dataset using various techniques including adding noise, resampling, generating synthetic data, and bootstrapping.

**Mathematical Formulation:**

1. **Adding Noise:**

   For each feature :math:`x_i` in the dataset, Gaussian noise is added:

   .. math::

       x_i' = x_i + \mathcal{N}(0, \sigma^2)

   where :math:`\sigma` is the standard deviation of the feature and :math:`\mathcal{N}(0, \sigma^2)` is the Gaussian noise.

2. **Resampling:**

   Randomly selects a subset of the data without replacement.

3. **Synthetic Data Generation:**

   New data points are generated by linear interpolation between pairs of existing data points:

   .. math::

       x_{\text{new}} = x_i + \alpha (x_j - x_i)

   where :math:`x_i` and :math:`x_j` are randomly chosen data points and :math:`\alpha` is a random number between 0 and 1.

4. **Bootstrapping:**

   Randomly samples data points with replacement to generate a new dataset.

**Examples:**

.. code-block:: python

    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from gofast.dataops.enrichment import enrich_data_spectrum

    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    augmented_data = enrich_data_spectrum(data, noise_level=0.02, resample_size=50, synthetic_size=50, bootstrap_size=50)
    print(augmented_data.shape)

.. code-block:: python

    # Example with different parameters
    augmented_data = enrich_data_spectrum(data, noise_level=0.01, resample_size=100, synthetic_size=100, bootstrap_size=100)
    print(augmented_data.describe())
