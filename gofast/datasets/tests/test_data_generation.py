# -*- coding: utf-8 -*-
"""
test_data_generation.py 

@author: LKouadio <etanoyau@gmail.com>
"""

import pytest
import numpy as np
import pandas as pd
from gofast.datasets.generate import make_classification, make_regression
from gofast.datasets.generate import make_social_media_comments, make_african_demo 
from gofast.datasets.generate import make_agronomy_feedback, make_mining_ops 
from gofast.datasets.generate import  make_sounding, make_medical_diagnosis
from gofast.datasets.generate import  make_well_logging, make_ert,make_tem
from gofast.datasets.generate import make_erp, make_elogging, make_gadget_sales
from gofast.datasets.generate import make_retail_store, make_cc_factors, make_water_demand

@pytest.mark.parametrize("function, config", [
    (make_classification, {'n_samples': 100, 'n_features': 20, 'return_X_y': True}),
    (make_regression, {'n_samples': 70, 'n_features': 7, 'return_X_y': True}),
    (make_social_media_comments, {'samples': 1000, 'return_X_y': True}),
    (make_african_demo, {'start_year': 1960, 'end_year': 2020, 'return_X_y': True}),
    (make_agronomy_feedback, {'samples': 100, 'num_years': 5, 'return_X_y': True}),
    (make_mining_ops, {'samples': 1000, 'return_X_y': True}),
    (make_sounding, {'samples': 100, 'num_layers': 5, 'return_X_y': True}),
    (make_medical_diagnosis, {'samples': 1000, 'return_X_y': True}),
])
def test_dataset_generators_batch1(function, config):
    X, y = function(**config)
    assert isinstance(X, (np.ndarray, pd.DataFrame)), "X should be an ndarray or DataFrame"
    assert isinstance(y, (np.ndarray, pd.Series)), "y should be an ndarray or Series"
    if 'samples' in config or 'n_samples' in config: 
        assert X.shape[0] == (config['samples'] if 'samples' in config else config['n_samples']),( 
            "Number of samples in X does not match expected")
        assert len(y) == (config['samples'] if 'samples' in config else config['n_samples']), ( 
                "Number of samples in y does not match expected"
                )
    print(f"Test passed with configuration: {config}")
    
@pytest.mark.parametrize("function, config", [
    (make_well_logging, {'depth_start': 0., 'depth_end': 200., 
                          'depth_interval': .5, 'return_X_y': True}),
    (make_ert, {'samples': 100, 'equipment_type': 'SuperSting R8', 
                'return_X_y': True}),
    (make_tem, {'samples': 500, 'lat_range': (34.00, 36.00),
                'lon_range': (-118.50, -117.00), 'return_X_y': True}),
])
def test_dataset_generators_advanced(function, config):
    X, y = function(**config)
    assert isinstance(X, (np.ndarray, pd.DataFrame)), "X should be an ndarray or DataFrame"
    assert isinstance(y, (np.ndarray, pd.Series)), "y should be an ndarray or Series"
    if config.get("samples", False): 
        assert len(X) == config.get('samples', 10), ( 
            "Number of samples in X does not match expected" )
        assert len(y) == config.get('samples', 10), ( 
            "Number of samples in y does not match expected"
            )
    print(f"Test passed with configuration: {config}")

# By default `return_X_y=True`. 
@pytest.mark.parametrize("function, config", [
    (make_erp, {'samples': 1000, 'lat_range': (34.00, 36.00), 'lon_range': (-118.50, -117.00),
                'resistivity_range': (10, 1000) }),
    (make_elogging, {'start_date': '2021-01-01', 'end_date': '2021-01-31', 
                      'samples': 100}),
    (make_gadget_sales, {'start_date': '2021-12-26', 'end_date': '2022-01-10',
                          'samples': 500}),
    (make_retail_store, {'samples': 1000}),
    (make_cc_factors, {'samples': 1000, 'noise': .1}),
    (make_water_demand, {'samples': 700}),
])
def test_dataset_generators_batch2(function, config):
    # Generate dataset
    data = function(**config)

    # Check if the data is in the expected format (DataFrame or tuple of X, y)
    assert isinstance(data, (pd.DataFrame, tuple)), ( 
        "Data should be a DataFrame or a tuple of (X, y)") 
    
    if isinstance(data, tuple):
        X, y = data
        # Check if X and y have the correct shape
        assert X.shape[0] == config.get('samples', 100), ( 
            "Number of samples in X does not match expected")
        assert y.shape[0] == config.get('samples', 100), ( 
            "Number of samples in y does not match expected") 
    else:
        # For functions returning a DataFrame, check if the number 
        # of rows match the expected samples
        assert data.shape[0] == config.get('samples', 100), ( 
            "Number of samples in the DataFrame does not match expected"
            )
    print(f"Test passed with configuration: {config}")

if __name__=="__main__": 
    pytest.main([__file__])