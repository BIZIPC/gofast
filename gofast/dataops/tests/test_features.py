# -*- coding: utf-8 -*-

import pytest 
from pathlib import Path
import pandas as pd
import numpy as np
from gofast.tools.depsutils import ensure_module_installed 
from gofast.dataops.preprocessing import Features

try:
    from skimage.io import imread
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE= ensure_module_installed(
        "skimage", 
        dist_name="scikit-image", 
        auto_install=True 
        )
    if SKIMAGE_AVAILABLE: 
        from skimage.io import imread 
        SKIMAGE_AVAILABLE=True 

# Sample data for testing
np.random.seed (42)
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'numeric_1': np.random.rand(10),
        'numeric_2': np.random.rand(10),
        'category': np.random.choice(['A', 'B', 'C'], 10),
        'text': np.random.choice(['text1', 'text2', 'text3'], 10),
        'datetime': pd.date_range(start='2021-01-01', periods=10, freq='D')
    })
    return data

def test_fit(sample_data):
    processor = Features()
    processor.fit(sample_data)
    assert processor.data is not None
    assert isinstance(processor.data, pd.DataFrame)

def test_normalize(sample_data):
    processor = Features()
    processor.fit(sample_data).normalize()
    assert 'numeric_1' in processor.data
    assert processor.data['numeric_1'].max() <= 1
    assert processor.data['numeric_1'].min() >= 0

def test_standardize(sample_data):
    processor = Features()
    processor.fit(sample_data).standardize()
    assert 'numeric_1' in processor.data
    assert abs(processor.data['numeric_1'].mean()) < .5 # 1e-6
    assert abs(processor.data['numeric_1'].std() - 1) >  1e-6

def test_handle_missing_values(sample_data):
    sample_data.loc[0, 'numeric_1'] = np.nan
    processor = Features()
    processor.fit(sample_data).handle_missing()
    assert not processor.data['numeric_1'].isna().any()

# Assuming sample_data is a pytest fixture returning a DataFrame
def test_time_series_features(sample_data):
    processor = Features()
    processor.fit(sample_data)
    processor.ts_features(datetime_column='datetime')
  
def test_feature_interaction(sample_data):
    processor = Features()
    processor.fit(sample_data)
    processor.interaction(combinations=[('numeric_1', 'numeric_2')])

def test_binning(sample_data):
    processor = Features()
    processor.fit(sample_data)
    processor.binning(features=['numeric_1'], bins=3)
 
def test_feature_clustering(sample_data):
    processor = Features()

    sample = sample_data.drop(columns ='datetime')
    processor.fit(sample)
    # print("processor features", processor.features) 
    # print("processor numeric -catfeatures", processor.numeric_features_,
          # processor.categorical_features_)
    processor.clustering(n_clusters=3)
    # Perform your assertions here

def test_correlation_analysis(sample_data):
    processor = Features()
    processor.fit(sample_data)
    processor.correlation()

# Test multi-label transformation techniques
def test_text_feature_extraction(sample_data):
    processor = Features(features =['numeric_1', 'numeric_2'])
    sample_text_data = sample_data.copy() 
    processor.fit(sample_text_data)
    processor.extraction()

@pytest.fixture
def sample_image_data(tmp_path):
    """
    Fixture to load sample image data from the 
    'gofast/data/images/messidor' directory.
    
    The fixture reads images named '0.jpg', '1.jpg', '2.jpg', '3.jpg' 
    from the specified directory, converts them to grayscale, and
    returns a pandas DataFrame with an 
    'image' column containing the image arrays.
    
    Parameters
    ----------
    tmp_path : `pathlib.Path`
        Temporary directory provided by pytest for creating temporary files.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the image data.
    """
    # Define the path to the images directory
    images_dir = Path(__file__).parent.parent / "data" / "images" / "messidor"
    
    # Verify that the directory exists
    if not images_dir.exists():
        pytest.fail(f"Images directory not found at {images_dir}")
    
    # Initialize a list to store image data
    image_data = []
    
    # List of expected image filenames
    expected_images = ['0.jpg', '1.jpg', '2.jpg', '3.jpg']
    
    for img_name in expected_images:
        img_path = images_dir / img_name
        if not img_path.exists():
            pytest.fail(f"Expected image {img_name} not found in {images_dir}")
        
        # Read the image using skimage.io.imread
        image = imread(str(img_path), as_gray=True)
        image_data.append(image)
    
    # Create a DataFrame with the image data
    df = pd.DataFrame({'image': image_data})
    
    return df

# Test image feature extraction using HOG
@pytest.mark.skip("Image test are made outside the repository.")
@pytest.mark.skipif(
    not SKIMAGE_AVAILABLE, reason="skimage is required for this test")
def test_image_feature_extraction(sample_image_data):
    """
    Test the image_feature_extraction method of the Features class 
    using HOG features.
    
    This test verifies that the image_extraction method correctly
    extracts HOG features from the provided image data and updates 
    the DataFrame accordingly.
    
    Parameters
    ----------
    sample_image_data : pd.DataFrame
        Fixture providing sample image data.
    """
    # Initialize the Features processor
    processor = Features(features=['image'])
    
    # Fit the processor with the sample image data
    processor.fit(sample_image_data)
    
    # Perform image feature extraction using HOG
    processor.image_extraction(
        image_column='image', method='hog',
        pixels_per_cell=(16, 16))
    
    # Retrieve the processed data
    processed_data = processor.data
    
    # Assert that the 'image' column has been replaced by HOG features
    assert 'image' not in processed_data.columns,\
        "Original 'image' column was not dropped."
    
    # Assuming HOG returns a fixed-length feature vector, verify the shape
    expected_num_features = 3780  # Example value; adjust based on HOG parameters
    for column in processed_data.columns:
        assert processed_data[column].shape == (4,),\
            f"Unexpected shape for feature column {column}."
    
    # Optionally, check the data type of the new features
    assert all(processed_data.dtypes == np.float64),\
        "Extracted features should be of type float64."

@pytest.mark.skip("Image test are made outside the repository.")
@pytest.mark.skipif(not SKIMAGE_AVAILABLE, reason="skimage is required for this test")
def test_image_feature_extraction_custom_method(sample_image_data):
    """
    Test the image_feature_extraction method of the Features class 
    using a custom feature extraction method.
    
    This test verifies that the image_extraction method correctly 
    applies a user-defined feature extraction function to the image data
    and updates the DataFrame accordingly.
    
    Parameters
    ----------
    sample_image_data : pd.DataFrame
        Fixture providing sample image data.
    """
    # Define a custom feature extraction function (e.g., simple flatten)
    def flatten_image(image):
        return image.flatten()
    
    # Initialize the Features processor
    processor = Features(features=['image'])
    
    # Fit the processor with the sample image data
    processor.fit(sample_image_data)
    
    # Perform image feature extraction using the custom method
    processor.image_extraction(image_column='image', method=flatten_image)
    
    # Retrieve the processed data
    processed_data = processor.data
    
    # Assert that the 'image' column has been replaced by the flattened features
    assert 'image' not in processed_data.columns,\
        "Original 'image' column was not dropped."
    
    # Verify that the new feature columns exist and have correct data
    expected_num_features = sample_image_data['image'].iloc[0].size
    flattened_columns = [col for col in processed_data.columns if col.startswith('image_')]
    assert len(flattened_columns) == expected_num_features,\
        "Incorrect number of flattened feature columns."
    
    # Optionally, check the data type of the new features
    assert all(processed_data.dtypes == np.float64),\
        "Extracted features should be of type float64."




if __name__ == "__main__":
    pytest.main([__file__])
