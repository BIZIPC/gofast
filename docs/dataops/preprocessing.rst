.. _preprocessing:


Preprocessing
=============

.. currentmodule:: gofast.dataops.preprocessing

The :mod:`gofast.dataops.preprocessing` module focuses on initial data preparation tasks. This module provides functions to transform 
dates, apply vectorization techniques, and perform data augmentation.

Key Features
------------
- **Date Transformation**: 
  Functions to detect and transform date columns in a DataFrame. These functions help in converting string representations of dates 
  into pandas datetime objects, enabling easier date manipulations and time series analysis.

- **Vectorization**: 
  Tools to apply various vectorization techniques for text data, including:
  - **Bag-of-Words (BoW)**: Represents text data as vectors of word counts, highlighting the frequency of word occurrences.
  - **TF-IDF**: Weighs words based on their occurrence in a document relative to their frequency across all documents, emphasizing 
    significant words.
  - **Word Embeddings**: Converts words into dense vector representations that capture semantic relationships, using pre-trained 
    embedding models.

- **Data Augmentation**: 
  Techniques to augment data, such as repeating datasets with random variations, which enhances training diversity and helps improve 
  model performance. This is particularly useful for small datasets, enabling models to generalize better by creating more diverse 
  training examples.

- **Statistical Transformations**: 
  Functions to apply transformations like Box-Cox to stabilize variance and make data more normally distributed. These transformations 
  are useful for improving the performance of machine learning models that assume normality in the data.

Function Descriptions
---------------------

transform_dates
~~~~~~~~~~~~~~~
Detects and optionally transforms columns in a DataFrame that can be interpreted as dates. The function uses advanced parameters for 
greater control over the conversion process.

**Mathematical Formulation:**

Transforming a column to datetime format:

.. math::

    \text{datetime\_column} = \text{pd.to\_datetime}(\text{column}, \text{format}=\text{fmt}, \text{errors}=\text{errors})

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.preprocessing import transform_dates

    # Example 1: Detecting datetime columns
    data = pd.DataFrame({
        'date': ['2021-01-01', '2021-01-02'],
        'value': [1, 2],
        'timestamp': ['2021-01-01 12:00:00', None],
        'text': ['Some text', 'More text']
    })
    datetime_columns = transform_dates(data, fmt='%Y-%m-%d', return_dt_columns=True)
    print(datetime_columns)  # Output: ['date', 'timestamp']

    # Example 2: Transforming datetime columns with format
    transformed_data = transform_dates(data, include_columns=['date', 'timestamp'], errors='ignore')
    print(transformed_data.dtypes)

apply_bow_vectorization
~~~~~~~~~~~~~~~~~~~~~~~
Applies bag-of-words (BoW) vectorization to a list of text documents, converting them into a matrix of token counts.

**Mathematical Formulation:**

BoW vectorization involves counting the occurrences of each word in a document:

.. math::

    \text{BoW}(d, w) = \sum_{t \in d} \mathbb{1}(t = w)

where \(d\) is the document, \(w\) is the word, and \(\mathbb{1}(t = w)\) is an indicator function that equals 1 if \(t = w\) and 
0 otherwise.

Examples:

.. code-block:: python

    from gofast.dataops.preprocessing import apply_bow_vectorization

    # Example 1: Simple BoW vectorization
    documents = ["This is a sample document.", "This document is another example."]
    bow_matrix = apply_bow_vectorization(documents)
    print(bow_matrix)

    # Example 2: BoW vectorization with preprocessing
    documents = ["Text preprocessing is important.", "Preprocessing text data for BoW."]
    bow_matrix = apply_bow_vectorization(documents, preprocess=True)
    print(bow_matrix)

apply_tfidf_vectorization
~~~~~~~~~~~~~~~~~~~~~~~~~
Applies TF-IDF vectorization to a list of text documents, converting them into a matrix of TF-IDF scores.

**Mathematical Formulation:**

Term Frequency (TF):

.. math::

    \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}

Inverse Document Frequency (IDF):

.. math::

    \text{IDF}(t) = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents with term } t} \right)

TF-IDF score for term \(t\) in document \(d\):

.. math::

    \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)

Examples:

.. code-block:: python

    from gofast.dataops.preprocessing import apply_tfidf_vectorization

    # Example 1: Simple TF-IDF vectorization
    documents = ["This is a sample document.", "This document is another example."]
    tfidf_matrix = apply_tfidf_vectorization(documents)
    print(tfidf_matrix)

    # Example 2: TF-IDF vectorization with preprocessing
    documents = ["TF-IDF is a common technique.", "Preprocessing text data for TF-IDF."]
    tfidf_matrix = apply_tfidf_vectorization(documents, preprocess=True)
    print(tfidf_matrix)

apply_word_embeddings
~~~~~~~~~~~~~~~~~~~~~
Applies word embeddings to a list of text documents, converting them into a matrix of word vectors.

**Mathematical Formulation:**

Word embeddings map words to high-dimensional vectors based on their context in a corpus. For example, using Word2Vec:

.. math::

    \text{embedding}(w) = \text{Word2Vec}(w)

Examples:

.. code-block:: python

    from gofast.dataops.preprocessing import apply_word_embeddings

    # Example 1: Applying word embeddings
    documents = ["This is a sample document.", "This document is another example."]
    embedding_matrix = apply_word_embeddings(documents)
    print(embedding_matrix)

    # Example 2: Word embeddings with custom model
    documents = ["Word embeddings are powerful.", "Using custom embeddings for text."]
    embedding_matrix = apply_word_embeddings(documents, model='custom_model')
    print(embedding_matrix)

augment_data
~~~~~~~~~~~~
Augments the data by applying various augmentation techniques such as noise addition, resampling, and synthetic data generation.

**Mathematical Formulation:**

Augmentation techniques can be represented as transformations applied to the original data:

.. math::

    \text{augmented\_data} = f(\text{original\_data})

where \(f\) represents the augmentation function.

Examples:

.. code-block:: python

    from gofast.dataops.preprocessing import augment_data

    # Example 1: Augmenting data with noise
    data = [1, 2, 3, 4, 5]
    augmented_data = augment_data(data, method='noise')
    print(augmented_data)

    # Example 2: Augmenting data with resampling
    data = [1, 2, 3, 4, 5]
    augmented_data = augment_data(data, method='resample')
    print(augmented_data)
    
    # Example 2: Augmenting data with target labels
    X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
    X_aug, y_aug = augment_data(X, y)
    print(X_aug.shape, y_aug.shape)  # Output: (4, 2), (4,)

    # Example 3: Augmenting data without target labels
    X = np.array([[1, 2], [3, 4]])
    X_aug = augment_data(X, y=None)
    print(X_aug.shape)  # Output: (4, 2)
    
boxcox_transformation
~~~~~~~~~~~~~~~~~~~~~
Applies Box-Cox transformation to stabilize variance and make the data more normally distributed.

**Mathematical Formulation:**

Box-Cox transformation:

.. math::

    y(\lambda) = \begin{cases}
    \frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
    \log(y) & \text{if } \lambda = 0
    \end{cases}

Examples:

.. code-block:: python

    from gofast.dataops.preprocessing import boxcox_transformation

    # Example 1: Applying Box-Cox transformation
    data = [1, 2, 3, 4, 5]
    transformed_data, lambda_ = boxcox_transformation(data)
    print(transformed_data)

    # Example 2: Applying Box-Cox transformation with different lambda
    data = [1, 2, 3, 4, 5]
    transformed_data, lambda_ = boxcox_transformation(data, lambda_=0.5)
    print(transformed_data)

boxcox_transformation
~~~~~~~~~~~~~~~~~~~~~
Applies Box-Cox transformation to stabilize variance and make the data more normally distributed.

**Mathematical Formulation:**

Box-Cox transformation:

.. math::

    y(\lambda) = \begin{cases}
    \frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
    \log(y) & \text{if } \lambda = 0
    \end{cases}

Examples:

.. code-block:: python

    from gofast.dataops.preprocessing import boxcox_transformation

    # Example 1: Applying Box-Cox transformation
    data = pd.DataFrame({
        'A': np.random.rand(10) * 100,
        'B': np.random.normal(loc=50, scale=10, size=10),
        'C': np.random.randint(1, 10, size=10)
    })
    transformed_data, lambda_values = boxcox_transformation(data)
    print(transformed_data.head())
    print(lambda_values)

    # Example 2: Box-Cox transformation with different lambda
    data = pd.DataFrame({
        'A': np.random.rand(10) * 100,
        'B': np.random.normal(loc=50, scale=10, size=10),
        'C': np.random.randint(1, 10, size=10)
    })
    transformed_data, lambda_values = boxcox_transformation(data, min_value=0.5, verbose=1)
    print(transformed_data.head())
    print(lambda_values)

base_transform
~~~~~~~~~~~~~~
Applies preprocessing transformations to the specified DataFrame, including handling of missing values, feature scaling, encoding 
categorical variables, and optionally introducing noise to numeric features. Transformations can be selectively applied to specified 
columns.

Examples:

.. code-block:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from gofast.dataops.preprocessing import base_transform

    # Example 1: Apply base_transform to preprocess features, excluding the target column
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    data = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    data['target'] = y
    preprocessed_data = base_transform(data, target_columns='target', noise_level=0.1, seed=42)
    print(preprocessed_data.head())

    # Example 2: Apply base_transform with custom columns and noise level
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    data = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    preprocessed_data = base_transform(data, columns=['feature_1', 'feature_2'], noise_level=0.2, seed=42)
    print(preprocessed_data.head())


apply_tfidf_vectorization
~~~~~~~~~~~~~~~~~~~~~~~~~
Applies TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to one or more text columns in a pandas DataFrame. The function 
concatenates the resulting features back into the original DataFrame. The TF-IDF method weighs the words based on their occurrence in a 
document relative to their frequency across all documents, helping to highlight words that are more interesting, i.e., frequent in a 
document but not across documents.

Term Frequency (TF):

.. math::

    \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}

Inverse Document Frequency (IDF):

.. math::

    \text{IDF}(t) = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents with term } t} \right)

TF-IDF score for term \(t\) in document \(d\):

.. math::

    \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.preprocessing import apply_tfidf_vectorization

    # Example 1: Applying TF-IDF vectorization to multiple text columns
    data = pd.DataFrame({
        'message_to_investigators': ['This is a sample message', 'Another sample message', 'Missing message'],
        'additional_notes': ['Note one', 'Note two', 'Note three']
    })
    processed_data = apply_tfidf_vectorization(
        data, text_columns=['message_to_investigators', 'additional_notes'], 
        max_features=50, missing_value_handling='fill', fill_value='NA')
    print(processed_data.head())

    # Example 2: Applying TF-IDF vectorization with custom stop words
    data = pd.DataFrame({
        'reviews': ['Great product', 'Terrible service', 'Will buy again', 'Not worth the price']
    })
    processed_data = apply_tfidf_vectorization(
        data, text_columns='reviews', stop_words=['great', 'terrible'])
    print(processed_data.head())

apply_word_embeddings
~~~~~~~~~~~~~~~~~~~~~
Applies word embedding vectorization followed by dimensionality reduction to text columns in a pandas DataFrame. This process converts 
text data into a numerical form that captures semantic relationships between words, making it suitable for use in machine learning models. The function leverages pre-trained word embeddings (e.g., Word2Vec, GloVe) to represent words in a high-dimensional space and then applies PCA (Principal Component Analysis) to reduce the dimensionality of these embeddings to a specified number of components.

Word embeddings map words to high-dimensional vectors based on their context in a corpus. For example, using Word2Vec:

.. math::

    \text{embedding}(w) = \text{Word2Vec}(w)

Principal Component Analysis (PCA) reduces the dimensionality of the embeddings:

.. math::

    \text{reduced\_embedding}(w) = \text{PCA}(\text{embedding}(w))

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.preprocessing import apply_word_embeddings

    # Example 1: Applying word embeddings to a single text column
    df = pd.DataFrame({'reviews': [
        'This product is great', 'Terrible customer service', 'Will buy again', 'Not worth the price']})
    processed_df = apply_word_embeddings(
        df, text_columns='reviews', embedding_file_path='path/to/embeddings.bin', n_components=50)
    print(processed_df.head())

    # Example 2: Applying word embeddings to multiple text columns
    df = pd.DataFrame({
        'comments': ['Good product', 'Bad service', 'Excellent quality', 'Poor design'],
        'feedback': ['Highly recommend', 'Will not buy again', 'Loved it', 'Disappointed']
    })
    processed_df = apply_word_embeddings(
        df, text_columns=['comments', 'feedback'], embedding_file_path='path/to/embeddings.bin', n_components=30)
    print(processed_df.head())

apply_bow_vectorization
~~~~~~~~~~~~~~~~~~~~~~~
Applies Bag of Words (BoW) vectorization to one or more text columns in a pandas DataFrame. The function concatenates the resulting 
features back into the original DataFrame. BoW is a simpler approach that creates a vocabulary of all the unique words in the dataset 
and then models each text as a count of the number of times each word appears.

**Mathematical Formulation:**

Bag of Words (BoW) model represents text data as vectors of word counts. For a document \( d \) and a vocabulary \( V \):

.. math::

    \text{BoW}(d) = [ \text{count}(w_1, d), \text{count}(w_2, d), \ldots, \text{count}(w_n, d) ]

where \( \text{count}(w_i, d) \) is the frequency of word \( w_i \) in document \( d \), and \( n \) is the size of the vocabulary \( V \).

Examples:

.. code-block:: python

    import pandas as pd
    from gofast.dataops.preprocessing import apply_bow_vectorization

    # Example 1: Applying BoW vectorization to multiple text columns
    data = pd.DataFrame({
        'message_to_investigators': ['This is a sample message', 'Another sample message', 'Missing message'],
        'additional_notes': ['Note one', 'Note two', 'Note three']
    })
    processed_data = apply_bow_vectorization(
        data, text_columns=['message_to_investigators', 'additional_notes'], 
        max_features=50, missing_value_handling='fill', fill_value='NA')
    print(processed_data.head())

    # Example 2: Applying BoW vectorization with custom stop words
    data = pd.DataFrame({
        'reviews': ['Great product', 'Terrible service', 'Will buy again', 'Not worth the price']
    })
    processed_data = apply_bow_vectorization(
        data, text_columns='reviews', stop_words=['great', 'terrible'])
    print(processed_data.head())

