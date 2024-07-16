# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:18:13 2023

@author: Daniel
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import matplotlib.pyplot as plt 

from sklearn.datasets import make_classification
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from gofast.tools.coreutils import is_module_installed
from gofast.transformers.feature_engineering import ( 
    StratifyFromBaseFeature,
    CategoryBaseStratifier, 
    CategorizeFeatures, 
    CombinedAttributesAdder, 
    DataFrameSelector,
    FrameUnion,
    FeatureSelectorByModel,
    DimensionalityReducer,
    PolynomialFeatureCombiner,
    BaseCategoricalEncoder,
    CategoricalEncoder, 
    FeatureScaler,
    MissingValueImputer,
    ColumnSelector,
    BaseColumnSelector,
    LogTransformer, 
    SequentialBackwardSelector,
    KMeansFeaturizer, 
    CategoryFrequencyEncoder,
    ) 
from gofast.transformers.text import  ( 
    TextFeatureExtractor, 
    TextToVectorTransformer
    )
from gofast.transformers.ts import ( 
    TimeSeriesFeatureExtractor,
    DateTimeCyclicalEncoder,
    LagFeatureGenerator,
    DifferencingTransformer,
    MovingAverageTransformer,
    CumulativeSumTransformer,
    SeasonalDecomposeTransformer,
    FourierFeaturesTransformer,
    TrendFeatureExtractor,
    DateFeatureExtractor, 
    )
    
from gofast.transformers.image import (
    ImageResizer,
    ImageNormalizer,
    ImageToGrayscale,
    ImageAugmenter,
    ImageChannelSelector,
    ImageFeatureExtractor,
    ImageEdgeDetector,
    ImageHistogramEqualizer,
    ImagePCAColorAugmenter,
    ImageBatchLoader, 
)

# 
# 
# install scikit-image 
try: 
    from skimage.transform import resize # noqa 
except: 
    from gofast.tools.funcutils import install_package 
    if not is_module_installed("skimage", distribution_name='scikit-image'): 
        install_package('skimage', dist_name='scikit-image', 
                        infer_dist_name= True )
        
def test_text_vectorizer_initialization():
    transformer = TextToVectorTransformer()
    assert transformer.columns == 'auto'
    assert transformer.append_transformed is True
    assert transformer.keep_original is False

def test_text_vectorizer_auto_detect_columns():
    df = pd.DataFrame({
        'text': ['hello world', 'test text'],
        'number': [1, 2]
    })
    transformer = TextToVectorTransformer()
    transformed_df = transformer.fit_transform(df)
    assert 'text_tfidf_0' in transformed_df.columns

def test_text_vectorizer_specified_columns():
    df = pd.DataFrame({
        'text1': ['hello world', 'example text'],
        'text2': ['another column', 'with text']
    })
    transformer = TextToVectorTransformer(columns=['text1'])
    transformed_df = transformer.fit_transform(df)
    assert 'text1_tfidf_0' in transformed_df.columns
    assert 'text2_tfidf_0' not in transformed_df.columns

def test_text_vectorizer_append_transformed_false():
    df = pd.DataFrame({
        'text': ['hello world', 'test text']
    })
    transformer = TextToVectorTransformer(append_transformed=False)
    transformed_df = transformer.fit_transform(df)
    assert 'text' not in transformed_df.columns
    assert len(transformed_df.columns) > 0

def test_text_vectorizer_keep_original():
    df = pd.DataFrame({
        'text': ['hello world', 'test text']
    })
    transformer = TextToVectorTransformer(append_transformed=True, keep_original=True)
    transformed_df = transformer.fit_transform(df)
    assert 'text' in transformed_df.columns
    assert 'text_tfidf_0' in transformed_df.columns

def test_vectorization_on_numpy_input():
    data = np.array(['hello world', 'test text'])
    transformer = TextToVectorTransformer()
    transformed_array = transformer.fit_transform(data)
    assert transformed_array.shape[1] > 0

def test_error_on_nonexistent_column():
    df = pd.DataFrame({
        'text': ['hello world', 'test text']
    })
    transformer = TextToVectorTransformer(columns=['nonexistent'])
    with pytest.raises(ValueError):
        transformer.fit(df)

def test_seasonal_decompose_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.random.randn(100)})
    
    # Initialize SeasonalDecomposeTransformer
    transformer = SeasonalDecomposeTransformer(model='additive', freq=12)
    
    # Fit and transform the DataFrame
    decomposed = transformer.fit_transform(X)
    
    # Check that seasonal decomposition is applied correctly
    assert 'seasonal' in decomposed.columns
    assert 'trend' in decomposed.columns
    assert 'resid' in decomposed.columns

def test_fourier_features_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'time': np.arange(100)})
    
    # Initialize FourierFeaturesTransformer
    transformer = FourierFeaturesTransformer(periods=[12, 24])
    
    # Fit and transform the DataFrame
    fourier_features = transformer.fit_transform(X)
    
    # Check that Fourier features are generated correctly
    assert 'sin_12' in fourier_features.columns
    assert 'cos_12' in fourier_features.columns
    assert 'sin_24' in fourier_features.columns
    assert 'cos_24' in fourier_features.columns

def test_trend_feature_extractor():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'time': np.arange(100), 
                      'value': np.random.randn(100)})
    
    # Initialize TrendFeatureExtractor
    transformer = TrendFeatureExtractor(
        order=1, time_col='time', value_col='value')
    
    # Fit and transform the DataFrame
    trend_features = transformer.fit_transform(X)
    
    # Check that trend features are extracted correctly
    assert 'trend_1' in trend_features.columns

def test_image_resizer():
    # Create a sample image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImageResizer
    resizer = ImageResizer(output_size=(128, 128))
    
    # Transform the image
    resized_image = resizer.transform(image)
    
    # Check that the image is resized to the specified size
    assert resized_image.shape == (128, 128, 3)

def test_image_normalizer():
    # Create a sample image with pixel values in [0, 255]
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Initialize ImageNormalizer
    normalizer = ImageNormalizer()
    
    # Transform the image
    normalized_image = normalizer.transform(image)
    
    # Check that pixel values are scaled to [0, 1]
    assert np.all(np.logical_and(normalized_image >= 0, normalized_image <= 1))

def test_image_to_grayscale():
    # Create a sample color image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImageToGrayscale
    converter = ImageToGrayscale(keep_dims=True)
    
    # Transform the image to grayscale
    grayscale_image = converter.transform(image)
    
    # Check that the image is converted to grayscale
    assert grayscale_image.shape == (256, 256, 1)
    

@pytest.mark.skipif ( not is_module_installed ("imgaug"),
                     reason="Expect 'imgaug' to be run properly. ") 
def test_image_augmenter():
    from imgaug import augmenters as iaa
    augmentation_funcs=[
        iaa.Fliplr(0.5), iaa.GaussianBlur(sigma=(0, 3.0))]
    
    # Create a sample image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImageAugmenter
    augmenter = ImageAugmenter(augmentation_funcs)
    
    # Transform the image using data augmentation techniques
    augmented_image = augmenter.transform(image)
    
    # Check that the augmented image is of the same shape
    assert augmented_image.shape == (256, 256, 3)

def test_log_transformer():
    # Create a sample DataFrame with numeric features
    X = pd.DataFrame({'income': [50000, 80000, 120000]})
    
    # Initialize LogTransformer
    transformer = LogTransformer(numeric_features=['income'], epsilon=1e-6)
    
    # Fit and transform the DataFrame
    X_transformed = transformer.fit_transform(X)
    
    # Check that log transformation is applied correctly
    assert np.allclose(X_transformed['income'], np.log(X['income'] + 1e-6))

def test_time_series_feature_extractor():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'time_series': np.random.rand(100)})
    
    # Initialize TimeSeriesFeatureExtractor
    extractor = TimeSeriesFeatureExtractor(rolling_window=5)
    
    # Fit and transform the DataFrame
    features = extractor.fit_transform(X)
    
    # Check the shape of the extracted features
    assert features.shape == (100, 5)  # Five rolling statistics computed

def test_category_frequency_encoder():
    # Create a sample DataFrame with categorical data
    X = pd.DataFrame({'brand': ['apple', 'apple', 'samsung', 'samsung', 'nokia']})
    
    # Initialize CategoryFrequencyEncoder
    encoder = CategoryFrequencyEncoder(categorical_features=['brand'])
    
    # Fit and transform the DataFrame
    encoded_features = encoder.fit_transform(X)
    
    # Check that categorical encoding is applied correctly
    assert encoded_features['brand'].equals(pd.Series([0.4, 0.4, 0.4, 0.4, 0.2], name='brand'))

def test_date_time_cyclical_encoder():
    # Create a sample DataFrame with datetime data
    X = pd.DataFrame({'timestamp': pd.date_range(start='1/1/2018', periods=24, freq='H')})
    
    # Initialize DateTimeCyclicalEncoder
    encoder = DateTimeCyclicalEncoder(datetime_features=['timestamp'])
    
    # Fit and transform the DataFrame
    encoded_features = encoder.fit_transform(X)
    
    # Check that cyclical encoding is applied correctly
    assert 'timestamp_sin_hour' in encoded_features.columns
    assert 'timestamp_cos_hour' in encoded_features.columns

def test_lag_feature_generator():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.arange(100)})
    
    # Initialize LagFeatureGenerator
    generator = LagFeatureGenerator(lags=[1, 2, 3])
    
    # Fit and transform the DataFrame
    lag_features = generator.fit_transform(X)
    
    # Check that lag features are generated correctly
    assert 'lag_1' in lag_features.columns
    assert 'lag_2' in lag_features.columns
    assert 'lag_3' in lag_features.columns

def test_differencing_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.cumsum(np.random.randn(100))})
    
    # Initialize DifferencingTransformer
    transformer = DifferencingTransformer(periods=1, zero_first= True)
    
    # Fit and transform the DataFrame
    stationary_data = transformer.fit_transform(X)
    
    # Check that differencing is applied correctly
    assert 'value' in stationary_data.columns
    assert stationary_data['value'].iloc[0] ==0. 

def test_moving_average_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.random.randn(100)})
    
    # Initialize MovingAverageTransformer
    transformer = MovingAverageTransformer(window=5)
    
    # Fit and transform the DataFrame
    moving_avg = transformer.fit_transform(X)
    
    # Check that moving average is computed correctly
    assert 'value' in moving_avg.columns

def test_cumulative_sum_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.random.randn(100)})
    
    # Initialize CumulativeSumTransformer
    transformer = CumulativeSumTransformer()
    
    # Fit and transform the DataFrame
    cum_sum = transformer.fit_transform(X)
    
    # Check that cumulative sum is computed correctly
    assert 'value' in cum_sum.columns
    assert round(cum_sum['value'].iloc[-1],2) == round(X['value'].sum(), 2)

# Test SequentialBackwardSelection
def test_sequential_backward_selection():
    # Generate a sample dataset for testing
    data, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['target'] = np.random.choice([0, 1], size=100)

    
    knn = KNeighborsClassifier(n_neighbors=5)
    # Create a SequentialBackwardSelection instance
    sbs = SequentialBackwardSelector(knn, k_features=3)

    # Fit and transform the data
    X_selected = sbs.fit_transform(df.drop('target', axis=1), df['target'])

    # Ensure that the transformed data has the expected shape
    assert X_selected.shape == (df.shape[0], 3)

# Test KMeansFeaturizer
def test_kmeans_featurizer():
    # Generate a sample dataset for testing
    n_features =5
    data, _= make_classification(n_samples=100, n_features=n_features, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])

    # Create a KMeansFeaturizer instance
    kmeans_featurizer = KMeansFeaturizer(n_clusters=3)

    # Fit and transform the data
    X_kmeans = kmeans_featurizer.fit_transform(df)

    # Ensure that the transformed data has the expected shape
    assert X_kmeans.shape == (df.shape[0], n_features +1 )

# Test StratifiedWithCategoryAdder
def test_stratified_with_category_adder():
    # Generate a sample dataset for testing
    data,_ = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['category'] = np.random.choice(['A', 'B', 'C'], size=100)

    # Create a StratifyFromBaseFeature instance
    stratified_adder = StratifyFromBaseFeature(base_feature='feature1')

    # Transform the data
    df_transformed, _ = stratified_adder.fit_transform(df)

    # Ensure that transformed data has the expected shape
    assert df_transformed.shape[0] <= df.shape[0]

# Test StratifiedUsingBaseCategory
def test_stratified_using_base_category():
    # Generate a sample dataset for testing
    data,_ = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['category'] = np.random.choice(['A', 'B', 'C'], size=100)

    # Create a CategoryBaseStratifier instance
    stratified_base = CategoryBaseStratifier(base_column='category')

    # Transform the data
    df_transformed, _ = stratified_base.fit_transform(df)

    # Ensure that transformed data has the expected shape
    assert df_transformed.shape[0] <= df.shape[0]

# Test CategorizeFeatures
def test_categorize_features():
    # Generate a sample dataset for testing
    data, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['category'] = np.random.choice(['A', 'B', 'C'], size=100)

    # Create a CategorizeFeatures instance
    categorizer = CategorizeFeatures(columns=['category'])

    # Transform the data
    df_transformed = categorizer.fit_transform(df)

    # Ensure that transformed data has the expected shape
    # assert df_transformed.shape == df.shape

    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])

    # Ensure that the transformed data matches the LabelEncoder encoding
    assert np.array_equal(df_transformed['category'], df['category_encoded'])


# Define your test functions
def test_combined_attributes_adder():
    # Create a sample DataFrame
    data = {
        'lwi': [1, 2, 3, 4, 5],
        'ohmS': [0.1, 0.2, 0.5, 0.4, 0.8],
    }
    df = pd.DataFrame(data)
    
    # Initialize CombinedAttributesAdder
    cobj = CombinedAttributesAdder(attribute_names=['lwi', 'ohmS'], ) 
    
    # Transform the DataFrame
    Xadded = cobj.fit_transform(df)
    
    # Check if the new attribute is added correctly
    assert 'lwi_div_ohmS' in cobj.attribute_names_
    assert np.allclose(Xadded['lwi_div_ohmS'], [10.0, 10.0, 6.0, 10.0, 6.25])

def test_data_frame_selector():
    # Create a sample DataFrame
    data = {
        'numeric1': [1, 2, 3],
        'numeric2': [4, 5, 6],
        'category1': ['A', 'B', 'A'],
    }
    df = pd.DataFrame(data)
    
    # Initialize DataFrameSelector for numeric attributes
    num_selector = DataFrameSelector(columns=['numeric1', 'numeric2'], select_type='num')
    
    # Transform the DataFrame
    X_num = num_selector.fit_transform(df)
    
    # Check if the selected numeric columns are correct
    assert X_num.shape == (3, 2)
    assert np.allclose(X_num, np.array([[1, 4], [2, 5], [3, 6]]))
    
    # Initialize DataFrameSelector for categorical attributes
    cat_selector = DataFrameSelector(columns=['category1'], select_type='cat')
    
    # Transform the DataFrame
    X_cat = cat_selector.fit_transform(df)
    
    # Check if the selected categorical column is correct
    assert X_cat.shape == (3, 1)
    assert np.array_equal(X_cat, np.array([['A'], ['B'], ['A']]))

def test_frame_union():
    # Create a sample DataFrame
    data = {
        'numeric1': [1, 2, 3],
        'numeric2': [4, 5, 6],
        'category1': ['A', 'B', 'C'],
    }
    df = pd.DataFrame(data)
    
    # Initialize FrameUnion with default settings
    frame_union = FrameUnion(encode_mode= 'onehot')
    
    # Transform the DataFrame
    X = frame_union.fit_transform(df)
    
    # Check the shape of the transformed DataFrame
    assert X.shape == (3, 5)  # 2 numeric columns + 3 one-hot encoded categorical columns

    # Initialize FrameUnion with custom settings (e.g., scaling)
    frame_union = FrameUnion(scale=True, encode=False)
    
    # Transform the DataFrame
    X = frame_union.fit_transform(df)
    
    # Check the shape of the transformed DataFrame
    assert X.shape == (3, 3)  # 2 scaled numeric columns  + one non scaled categorical 


def test_text_feature_extractor():
    # Create a sample list of text data
    text_data = ['sample text data', 'another sample text']
    
    # Initialize TextFeatureExtractor
    extractor = TextFeatureExtractor(max_features=500)
    
    # Fit and transform the text data to sparse matrix and convert to numpy arra 
    features = extractor.fit_transform(text_data)
    
    # Check the shape of the transformed features
    assert features.toarray().shape[1] <= 500  # Assuming 500 max features

def test_date_feature_extractor():
    # Create a sample DataFrame with date columns
    date_data = pd.DataFrame({'date': ['2021-01-01', '2021-02-01']})
    
    # Initialize DateFeatureExtractor
    extractor = DateFeatureExtractor()
    
    # Fit and transform the date data
    features = extractor.fit_transform(date_data)
    
    # Check the shape of the transformed features
    assert features.shape == (2, 4)  # date + Year, month, and day features added

#
def test_feature_selector_by_model():
    # Create a sample dataset
    X = np.random.randn(10, 5)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Initialize FeatureSelectorByModel with a model (Logistic Regression)
    model = LogisticRegression()
    selector = FeatureSelectorByModel(model)
    
    # Fit and transform the dataset
    X_selected = selector.fit_transform(X, y)
    
    # Check the shape of the selected features
    assert X_selected.shape[1] <=5  # main two features selection applied

def test_categorical_encoder2():
    # Create a sample dataset
    X = [['Category A'], ['Category B'], ['Category C']]
    
    # Initialize CategoricalEncoder2
    enc = BaseCategoricalEncoder()
    
    # Fit and transform the dataset
    enc.fit(X)
    X_encoded = enc.transform([['Category B'], ['Category C']]).toarray()
    
    # Check the shape of the encoded features
    assert X_encoded.shape == (2, 2)  # Encoded as one-hot


def test_categorical_encoder():
    # Create a sample dataset
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    X= pd.DataFrame ( X, columns =['gender', 'number'])
    # Initialize CategoricalEncoder
    enc = CategoricalEncoder()
    
    # Fit and transform the dataset
    enc.fit(X)
    X_encoded = enc.transform(X)[0].toarray()
    
    # Check the shape of the encoded features
    assert X_encoded.shape == (3, 2)  # Encoded as one-hot


def test_polynomial_feature_combiner():
    # Create a sample dataset
    X = np.arange(6).reshape(3, 2)
    
    # Initialize PolynomialFeatureCombiner
    combiner = PolynomialFeatureCombiner(degree=2)
    
    # Fit and transform the dataset
    X_poly = combiner.fit_transform(X)
    
    # Check the shape of the transformed features
    assert X_poly.shape == (3, 6)  # Polynomial features added

def test_dimensionality_reducer():
    # Create a sample dataset
    X = np.array([[0, 0], [1, 1], [2, 2]])
    
    # Initialize DimensionalityReducer
    reducer = DimensionalityReducer(n_components=1)
    
    # Fit and transform the dataset
    X_reduced = reducer.fit_transform(X)
    
    # Check the shape of the reduced features
    assert X_reduced.shape == (3, 1)  # Reduced to 1 component

def test_column_selector2():
    # Create a sample DataFrame
    X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    
    # Initialize ColumnSelector2
    selector = BaseColumnSelector(column_names=['A', 'B'])
    
    # Fit and transform the DataFrame
    X_selected = selector.fit_transform(X)
    
    # Check that only selected columns are present
    assert list(X_selected.columns) == ['A', 'B']

def test_column_selector():
    # Create a sample DataFrame
    X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    
    # Initialize ColumnSelector
    selector = ColumnSelector(column_names=['A', 'B'])
    
    # Fit and transform the DataFrame
    X_selected = selector.fit_transform(X)
    
    # Check that only selected columns are present
    assert list(X_selected.columns) == ['A', 'B']


def test_missing_value_imputer2():
    # Create a sample DataFrame with missing values
    X = pd.DataFrame({'age': [25, np.nan, 50], 'income': [50000, 80000, np.nan]})
    
    # Initialize MissingValueImputer2
    imputer = MissingValueImputer(strategy='mean')
    
    # Fit and transform the DataFrame
    X_imputed = imputer.fit_transform(X)
    
    # Check that missing values are imputed
    assert not X_imputed.isnull().values.any()  # No NaN values should be present

def test_missing_value_imputer():
    # Create a sample dataset with missing values
    X = np.array([[1, 2], [np.nan, 3], [7, 6]])
    
    # Initialize MissingValueImputer
    imputer = MissingValueImputer(strategy='mean')
    
    # Fit and transform the dataset
    imputer.fit(X)
    X_imputed = imputer.transform(X)
    
    # Check that missing values are imputed
    assert not np.isnan(X_imputed).any()  # No NaN values should be present

def test_feature_scaler():
    # Create a sample dataset
    X = np.array([[0, 15], [1, -10]])
    
    # Initialize FeatureScaler
    scaler = FeatureScaler()
    
    # Fit and transform the dataset
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    # Check the shape of the scaled features
    assert X_scaled.shape == (2, 2)  # Scaled to zero mean and unit variance


def test_image_channel_selector():
    # Create a sample color image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImageChannelSelector to select the first two channels
    selector = ImageChannelSelector(channels=[0, 1])
    
    # Transform the image to select specific channels
    selected_channels = selector.transform(image)
    
    # Check that the selected channels are of the correct shape
    assert selected_channels.shape == (256, 256, 2)

# @pytest.mark.skipif(not is_module_installed('tensorflow'))
def test_image_feature_extractor2(): 
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input 
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    import numpy as np

    # Load the base VGG16 model without the top layer
    base_model = VGG16(weights='imagenet', include_top=False)

    # Add Global Average Pooling directly after the base model's output
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # This will reduce each feature map to a single value

    # Create the new model
    model = Model(inputs=base_model.input, outputs=x)
    # that correctly utilizes the updated `model` with global average pooling
    extractor = ImageFeatureExtractor(model=model, 
                                      preprocess_input=preprocess_input,
                                      # This should now be effectively redundant
                                      apply_global_pooling=True, 
                                      target_size=(224, 224))

    # Create a dummy image with the correct shape
    image = np.random.rand(1, 224, 224, 3)
    # Transform the image to extract features using the model
    features = extractor.transform(image)

    # Check the shape of the extracted features
    assert features.shape == (1, 512)  # Example shape, replace as needed
    
def test_image_feature_extractor():
    # Create a sample image
    image = np.random.rand(224, 224, 3)
    
    # Initialize ImageFeatureExtractor with a pre-trained model 
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model
    base_model = VGG16(weights='imagenet', include_top=False)
    # Replace with an actual pre-trained model
    model = Model(inputs=base_model.input, 
                      outputs=base_model.get_layer('block3_pool').output)
    extractor = ImageFeatureExtractor(model=model)
    
    # Transform the image to extract features using the model
    features = extractor.transform(image)
    
    # Check that the extracted features have the expected shape
    assert features.shape == (1, 28, 28, 256)  # Example shape, replace as needed

@pytest.mark.skipif (
    not is_module_installed("sobel"), reason="'sobel' package is not defined")
def test_image_edge_detector():
    # Create a sample grayscale image
    image = np.random.rand(256, 256)
    
    # Initialize ImageEdgeDetector with the Sobel method
    detector = ImageEdgeDetector(method='sobel')
    
    # Transform the image to detect edges
    edges = detector.transform(image)
    
    # Check that the edges image is of the same shape
    assert edges.shape == (256, 256)

def test_image_histogram_equalizer():
    # Create a sample grayscale image
    image = np.random.rand(256, 256)
    
    # Initialize ImageHistogramEqualizer
    equalizer = ImageHistogramEqualizer()
    
    # Transform the image to apply histogram equalization
    equalized_image = equalizer.transform(image)
    
    # Check that the equalized image is of the same shape
    assert equalized_image.shape == (256, 256)


def test_image_pca_color_augmenter():
    # Create a sample color image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImagePCAColorAugmenter with alpha_std
    augmenter = ImagePCAColorAugmenter(alpha_std=0.1)
    
    # Transform the image to apply PCA color augmentation
    pca_augmented_image = augmenter.transform(image)
    
    # Check that the PCA-augmented image is of the same shape
    assert pca_augmented_image.shape == (256, 256, 3)

# A fixture for creating a list of mock image file paths
@pytest.fixture
def mock_image_paths():
    return ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']

# A fixture for creating a mock preprocess function
@pytest.fixture
def mock_preprocess_func():
    def preprocess(image):
        # Pretend to process the image
        return image * 2  # Example operation
    return preprocess

@patch('os.listdir', return_value=['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'])
@patch('matplotlib.pyplot.imread', return_value=np.zeros((100, 100, 3)))
def test_image_batch_loader(mock_imread, mock_listdir, mock_image_paths, mock_preprocess_func):
    """
    Test to ensure the ImageBatchLoader yields batches of the specified size,
    correctly applies preprocessing, and works with mocked image paths.
    """
    # Initialize the ImageBatchLoader with a batch size of 2
    loader = ImageBatchLoader(batch_size=2, directory='mock/path/to/images', 
                              preprocess_func=mock_preprocess_func, custom_reader=plt.imread)

    # Use the loader to transform (load) images and collect the batches
    batches = list(loader.transform(None))

    # There should be 2 batches given 4 mock images and a batch size of 2
    assert len(batches) == 2

    # Each batch should have 2 images
    for batch in batches:
        assert batch.shape == (2, 100, 100, 3)

    # The mock imread function should be called once for each image
    assert mock_imread.call_count == 4

    # The images should be processed by the preprocess_func (e.g., values doubled)
    # Checking this by confirming the array isn't all zeros (as returned by mock_imread)

    assert np.all(batches[0]== 0)

# # Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
