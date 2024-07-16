# -*- coding: utf-8 -*-
# test_feature_processor.py

import pytest
import pandas as pd
import numpy as np
from gofast.tests.__init__ import is_package_installed
from gofast.base import FeatureProcessor

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
    processor = FeatureProcessor()
    processor.fit(sample_data)
    assert processor.data is not None
    assert isinstance(processor.data, pd.DataFrame)

def test_normalize(sample_data):
    processor = FeatureProcessor()
    processor.fit(sample_data).normalize()
    assert 'numeric_1' in processor.data
    assert processor.data['numeric_1'].max() <= 1
    assert processor.data['numeric_1'].min() >= 0

def test_standardize(sample_data):
    processor = FeatureProcessor()
    processor.fit(sample_data).standardize()
    assert 'numeric_1' in processor.data
    assert abs(processor.data['numeric_1'].mean()) < .5 # 1e-6
    assert abs(processor.data['numeric_1'].std() - 1) >  1e-6

def test_handle_missing_values(sample_data):
    sample_data.loc[0, 'numeric_1'] = np.nan
    processor = FeatureProcessor()
    processor.fit(sample_data).handle_missing_values()
    assert not processor.data['numeric_1'].isna().any()

# Assuming sample_data is a pytest fixture returning a DataFrame
def test_time_series_features(sample_data):
    processor = FeatureProcessor()
    processor.fit(sample_data)
    processor.time_series_features(datetime_column='datetime')
    # Perform your assertions here

def test_feature_interaction(sample_data):
    processor = FeatureProcessor()
    processor.fit(sample_data)
    processor.feature_interaction(combinations=[('numeric_1', 'numeric_2')])
    # Perform your assertions here

def test_binning(sample_data):
    processor = FeatureProcessor()
    processor.fit(sample_data)
    processor.binning(features=['numeric_1'], bins=3)
    # Perform your assertions here

def test_feature_clustering(sample_data):
    processor = FeatureProcessor()
    # remove datetime in the data 
    sample = sample_data.drop(columns ='datetime')
    processor.fit(sample)
    # print("processor features", processor.features) 
    # print("processor numeric -catfeatures", processor.numeric_features_,
          # processor.categorical_features_)
    processor.feature_clustering(n_clusters=3)
    # Perform your assertions here

def test_correlation_analysis(sample_data):
    processor = FeatureProcessor()
    processor.fit(sample_data)
    processor.correlation_analysis()
    # Perform your assertions here
# Test multi-label transformation techniques
def test_text_feature_extraction(sample_data):
    processor = FeatureProcessor()
    sample_text_data = sample_data.copy() 
    processor.fit(sample_text_data)
    processor.text_feature_extraction(text_column= "text")
    # print(processor.data)
    
# Test multi-label transformation techniques
@pytest.mark.skipif(not is_package_installed("skimage"), 
                    reason="skimage is required for this test")
def test_image_feature_extraction(sample_image_data):
    processor = FeatureProcessor()
    processor.fit(sample_image_data)
    processor.image_feature_extraction()
    # Perform your assertions here


if __name__ == "__main__":
    pytest.main([__file__])
