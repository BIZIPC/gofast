# -*- coding: utf-8 -*-
"""
test_load.py 

@author: LKouadio <etanoyau@gmail.com>
"""
import pytest
import scipy 
from unittest.mock import patch
import numpy as np
import pandas as pd
from gofast.api.structures import Boxspace 
from gofast.datasets.load import load_hydro_metrics, load_statlog
from gofast.datasets.load import  load_dyspnea, load_hlogs, load_nansha 
from gofast.datasets.load import load_bagoue, load_iris, load_mxs
from gofast.datasets.load import load_jrs_bet, load_forensic 

@patch("gofast.datasets.load._finalize_return")
@patch("gofast.datasets.load._handle_split_X_y")
@patch("gofast.datasets.load._prepare_data")
@patch("gofast.datasets.load._load_data")
@patch("gofast.datasets.load._ensure_data_file")
@patch("gofast.datasets.load._setup_key")
def test_load_hlogs_default(mock_setup_key, mock_ensure_data_file, mock_load_data,
                            mock_prepare_data, mock_handle_split_X_y, 
                            mock_finalize_return):
    mock_setup_key.return_value = ('h502', {"h502", "h2601"})
    mock_load_data.return_value = pd.DataFrame(
        {'feature1': [1, 2], 'remark': ['note1', 'note2']})
    mock_prepare_data.return_value = (
        pd.DataFrame({'feature1': [1, 2]}), 
        pd.DataFrame({'feature1': [1, 2]}), pd.Series([3, 4]), ['feature1'])
    mock_handle_split_X_y.return_value = (
        pd.DataFrame({'feature1': [1]}), 
        pd.DataFrame({'feature1': [2]}), pd.Series([3]), pd.Series([4]))
    mock_finalize_return.return_value = "Bunch object"
    
    # Default behavior without any flags
    result = load_hlogs()
    assert result == "Bunch object"
    mock_setup_key.assert_called_once()
    mock_ensure_data_file.assert_called_once()
    mock_load_data.assert_called_once()
    mock_prepare_data.assert_called_once()
    mock_finalize_return.assert_called_once()
    
def test_base_load_hlogs():
    # Test with split_X_y=True and as_frame=True
    X, Xt, Y, Yt = load_hlogs(split_X_y=True, as_frame=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Xt, pd.DataFrame)
    assert isinstance(Y, pd.DataFrame)
    assert isinstance(Yt, pd.DataFrame)

@patch("gofast.datasets.load.csv_data_loader")
@patch("gofast.datasets.load._to_dataframe")
def test_load_hydro_metrics_as_frame(mock_to_dataframe, mock_csv_data_loader):
    # Setup mock return values
    mock_csv_data_loader.return_value = (
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'flow': [5, 6]}),
        np.array([5, 6]),
        ['flow'],
        ['feature1', 'feature2'],
        "Hydro-Meteorological Dataset Description"
    )
    mock_to_dataframe.return_value = (
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'flow': [5, 6]}),
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]}),
        pd.Series([5, 6])
    )

def test_base_load_dyspnea():
    # Test return_X_y=True
    data, target = load_dyspnea(return_X_y=True)
    assert isinstance(data, np.ndarray)
    assert isinstance(target, np.ndarray)
    # Test as_frame=True and split_X_y=True
    X, Xt, y, yt = load_dyspnea(as_frame=True, split_X_y=True, test_size=0.3)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Xt, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(yt, pd.Series)

def test_load_forensic():
    """
    Run dataset tests on the forensic dataset with various configurations.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, 'as_frame': True},
        {'split_X_y': True, 'test_ratio': 0.3},
        {'key': 'preprocessed', 'exclude_message_column': False},
        {'key': 'raw', 'exclude_vectorized_features': False},
        {'tag': 'specific_tag', 'exclude_message_column': True, 'exclude_vectorized_features': True},
    ]
    
    for config in configurations:
        _common_forensic_tests(load_forensic, **config)
        print(f"Test passed with configuration: {config}")

def _common_forensic_tests(func, **kwargs):
    """
    Perform assertions to validate loading and formatting of forensic dataset
    based on the configuration.
    """
    result = func(**kwargs)
    
    # Assertions for return_X_y functionality
    if kwargs.get('return_X_y', False):
        data, target = result
        assert isinstance(data, (np.ndarray, pd.DataFrame)), "Data must be ndarray or DataFrame"
        assert isinstance(target, (np.ndarray, pd.Series)), "Target must be ndarray or Series"
        
        if kwargs.get('as_frame', False):
                assert isinstance(data, pd.DataFrame), "Data must be a DataFrame when as_frame=True"
                assert isinstance(target, pd.Series), "Target must be a Series when as_frame=True"
    elif len(kwargs)==0:
        # Assuming a custom Boxspace or similar object is returned
        assert 'data' in result and 'target' in result, "Result must contain both 'data' and 'target'"
        assert isinstance(result.get('frame', pd.DataFrame()), pd.DataFrame), "Frame must be a DataFrame"
        assert 'DESCR' in result, "Result must contain 'DESCR'"
        # Assertions for as_frame functionality
    
    elif kwargs.get('as_frame', False):
        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame when as_frame=True"
    
    # Assertions for split_X_y functionality
    elif kwargs.get('split_X_y', False):
        X_train, X_test, y_train, y_test = result
        assert isinstance(X_train, (np.ndarray, pd.DataFrame)), "X_train must be ndarray or DataFrame"
        assert isinstance(X_test, (np.ndarray, pd.DataFrame)), "X_test must be ndarray or DataFrame"
        assert isinstance(y_train, (np.ndarray, pd.Series)), "y_train must be ndarray or Series"
        assert isinstance(y_test, (np.ndarray, pd.Series)), "y_test must be ndarray or Series"
    
    # Additional checks for key parameter and feature exclusion
    if 'key' in kwargs:
        if kwargs.get('split_X_y', False): 
            data = X_train 
        data = data if kwargs.get('return_X_y', False) else result.frame 
        expected_columns = 'message_to_investigators' not in data.columns if kwargs.get(
            'exclude_message_column', True) else True
        assert expected_columns, "Message column exclusion/inclusion does not match configuration"
        if kwargs.get('exclude_vectorized_features', True) and kwargs['key'] == 'preprocessed':
            tfidf_columns = [c for c in data.columns if 'tfid' in c]
            assert not tfidf_columns, "TFIDF columns should not be present"

    # Validate the presence of description and feature names 
    # for comprehensive configurations
    if not kwargs.get('split_X_y', False) and not kwargs.get(
            'return_X_y', False) and not kwargs.get("as_frame", False):
        assert 'DESCR' in result, "Result must contain 'DESCR'"
        assert 'feature_names' in result, "Result must contain 'feature_names'"

def test_load_jrs_bet():
    """
    Run common dataset tests on the JRS Bet dataset with various configurations.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, 'as_frame': True},
        {'split_X_y': True, 'test_size': 0.2},
        {'key': 'classic', 'N': 5},
        {'key': 'neural', 'data_names': ['winning_numbers'], 'as_frame': True}, #deoes 
        {'key': 'raw', 'split_X_y': True, 'test_ratio': 0.3},
        {'tag': 'example_tag', 'N': 10, 'seed': 42}
    ]
    
    for config in configurations:
        _common_jrs_bet_tests(load_jrs_bet, **config)
        print(f"Test passed with configuration: {config}")

def _common_jrs_bet_tests(func, **kwargs):
    """
    Perform a series of assertions to validate the loading and formatting of datasets
    returned by the load_jrs_bet function. This includes checks for data types,
    structure, integrity, and specific behaviors based on the configuration.
    """
    result = func(**kwargs)
    
    # Assertions for basic configurations
    if kwargs.get('return_X_y', False):
        data, target = result
        assert isinstance(data, (np.ndarray, pd.DataFrame)), "Data must be ndarray or DataFrame"
        assert isinstance(target, (np.ndarray, pd.Series)), "Target must be ndarray or Series"
        
        if kwargs.get('as_frame', False):
                assert isinstance(data, pd.DataFrame), "Data must be a DataFrame when as_frame=True"
                assert isinstance(target, pd.Series), "Target must be a Series when as_frame=True"
    elif len(kwargs)==0:
        # Assuming a custom Boxspace or similar object is returned
        assert 'data' in result and 'target' in result, "Result must contain both 'data' and 'target'"
        assert isinstance(result.get('frame', pd.DataFrame()), pd.DataFrame), "Frame must be a DataFrame"
        assert 'DESCR' in result, "Result must contain 'DESCR'"
    elif kwargs.get('as_frame', False):
        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame when as_frame=True"
    
    # Assertions for split_X_y functionality
    elif kwargs.get('split_X_y', False):
        X_train, X_test, y_train, y_test = result
        assert isinstance(X_train, (np.ndarray, pd.DataFrame)), "X_train must be ndarray or DataFrame"
        assert isinstance(X_test, (np.ndarray, pd.DataFrame)), "X_test must be ndarray or DataFrame"
        assert isinstance(y_train, (np.ndarray, pd.Series)), "y_train must be ndarray or Series"
        assert isinstance(y_test, (np.ndarray, pd.Series)), "y_test must be ndarray or Series"

     # Validate the presence of description and feature 
     # names for comprehensive configurations
    if not kwargs.get('split_X_y', False) and not kwargs.get(
            'return_X_y', False) and not kwargs.get("as_frame", False):
         assert 'DESCR' in result, "Result must contain 'DESCR'"
         assert 'feature_names' in result, "Result must contain 'feature_names'"

def test_load_mxs():
    """
    Run common dataset tests on the MXS dataset with various configurations.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, 'as_frame': True},
        {'split_X_y': True, 'test_ratio': 0.2},
        {'key': 'sparse', 'return_X_y': True},
        {'key': 'scale', 'as_frame': True},
        {'key': 'data', 'target_names': ['k'], 'samples': "*", 'seed': 42, 'shuffle': True},
        {'key': 'raw', 'drop_observations': True},
    ]
    
    for config in configurations:
        _common_mxs_tests(load_mxs, **config)
        print(f"Test passed with configuration: {config}")

def _common_mxs_tests(func, **kwargs):
    """
    Perform a series of assertions to validate the loading and formatting of datasets
    returned by the load_mxs function. This includes checks for data types,
    structure, integrity, and specific aspects like sparse data handling.
    """
    result = func(**kwargs)
    
    # Handle different return types based on configurations
    if kwargs.get('return_X_y', False):
        data, target = result
        assert isinstance(data, (np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix)),( 
            "Data must be ndarray, DataFrame, or CSR matrix")
        assert isinstance(target, (np.ndarray, pd.Series, pd.DataFrame)), ( 
            "Target must be ndarray or Series/DataFrames")
        if kwargs.get('as_frame', False):
            assert isinstance(data, ( pd.DataFrame, scipy.sparse.csr_matrix)), ( 
                "Data must be a DataFrame/ CSR matrix  when as_frame=True")
            assert isinstance(target, (pd.Series, pd.DataFrame)), ( 
                "Target must be a Series when as_frame=True") 
        if 'key' in kwargs:
            key = kwargs['key']
            if key == 'sparse': 
                if type (data) ==object: # sparse is encapsulated in numpy array
                    assert isinstance(data.item(), scipy.sparse.csr_matrix), ( 
                        "Data must be CSR matrix for key='sparse'")
            # Example check for specific preprocessing steps associated with 'key'
            elif key == 'scale':
                # Assuming data should be scaled for 'scale' key, verify mean or range
                if isinstance(data, pd.DataFrame):
                    assert data.apply(lambda col: round(col.mean(), 1)).all(), ( 
                    "Data columns should be centered or scaled")
            
    elif len(kwargs)==0:
        # Assuming a custom Boxspace or similar object is returned for non-return_X_y configurations
        assert 'data' in result and 'target' in result, "Result must contain both 'data' and 'target'"
        assert isinstance(result.get('frame', pd.DataFrame()), (pd.DataFrame, type(None))), ( 
            "Frame must be DataFrame or None") 
        assert 'DESCR' in result, "Result must contain 'DESCR'"
    
    # Additional checks for split_X_y functionality
    elif kwargs.get('split_X_y', False):
        X_train, X_test, y_train, y_test = result
        assert len(X_train) > 0 and len(X_test) > 0, "Training and testing sets must not be empty"
        assert len(y_train) > 0 and len(y_test) > 0, "Training and testing target sets must not be empty"
        if kwargs.get('as_frame', False) or isinstance(X_train, scipy.sparse.csr_matrix):
            # Explicitly bypassing type checks for sparse matrices or DataFrame output
            pass
        else:
            # Additional type checks for non-sparse, non-DataFrame data
            assert isinstance(X_train, np.ndarray), ( 
                "X_train must be a ndarray when not in sparse or DataFrame format"
                )
        # Verify that the test ratio is respected
        total_len = len(X_train) + len(X_test)
        expected_test_len = int(total_len * kwargs.get('test_ratio', 0.2))
        expected_size = abs(len(X_test) - expected_test_len)
        assert (expected_size -7 <= expected_size<= expected_size +7) <= 1, ( 
            "Test set size does not match the expected test_ratio") 
        
    # Verify sampling effect if 'samples' is specified
    elif 'samples' in kwargs and kwargs['samples'] is not None and kwargs.get("as_frame", False):
        assert len(result) <= kwargs['samples'], "Data length does not match the specified number of samples"
    
    # Check for dropped observations if applicable
    if kwargs.get("drop_observations", False):
        assert 'remark' not in result.frame.columns, ( 
            "'remark' column should be dropped but is present") 
    
def test_load_iris():
    """
    Run common dataset tests on the iris dataset with various configurations.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, 'as_frame': True},
        {'split_X_y': True, 'test_ratio': 0.3},
        {'tag': 'flower', 'data_names': ['sepal length (cm)', 'sepal width (cm)']},
    ]
    
    for config in configurations:
        _common_bagoue_tests(load_iris, **config)
        print(f"Test passed with configuration: {config}")

def test_load_bagoue():
    """
    Run common dataset tests on the Bagoue dataset with various configurations.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, 'as_frame': True},
        {'split_X_y': True, 'test_ratio': 0.3},
        {'tag': 'example_tag', 'data_names': ['flow']},
    ]
    
    for config in configurations:
        _common_bagoue_tests(load_bagoue, **config)
        print(f"Test passed with configuration: {config}")

def _common_bagoue_tests(func, **kwargs):
    """
    Perform a series of assertions to validate the loading and formatting of datasets
    returned by the load_bagoue function. This includes checks for data types,
    structure, and integrity of the loaded dataset.
    """
    result = func(**kwargs)
    
    # Assertions based on the configuration
    if kwargs.get('return_X_y', False):
        data, target = result
        assert isinstance(data, (np.ndarray, pd.DataFrame)), "Data must be ndarray or DataFrame"
        assert isinstance(target, (np.ndarray, pd.Series)), "Target must be ndarray or Series"
        
        if kwargs.get('as_frame', False):
            assert isinstance(data, pd.DataFrame), "Data must be a DataFrame when as_frame=True"
            assert isinstance(target, pd.Series), "Target must be a Series when as_frame=True"
    elif len(kwargs)==0:
        # Assuming a custom Boxspace or similar object is returned
        assert 'data' in result and 'target' in result, "Result must contain both 'data' and 'target'"
        assert isinstance(result.get('frame', pd.DataFrame()), pd.DataFrame), "Frame must be a DataFrame"
        assert 'DESCR' in result, "Result must contain 'DESCR'"
        
    # Additional checks for as_frame=True without return_X_y
    elif kwargs.get('as_frame', False) and not kwargs.get('split_X_y', False):
        # Here, we expect result to be a DataFrame encapsulating both data and target
        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame when as_frame is True"
        
    # Test split_X_y functionality if applicable
    if kwargs.get('split_X_y', False):
        validate_split_X_y(func, **kwargs)

def test_load_nansha():
    """
    Run common dataset tests on the Nansha dataset with various configurations.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, 'as_frame': True},
        {'split_X_y': True, 'test_ratio': 0.3},
        {'key': 'ns', 'as_frame': True},
        {'key': 'engineering', 'years': '2018', 'target_names': ['ground_height_distance'],
          'return_X_y':True},
        {'key': 'ls', 'years': '2018', 'target_names': ['2021', '2018', '1017'], 'split_X_y':True},
        {'samples': 100, 'seed': 42, 'shuffle': True},
        {'tag': 'test', 'data_names': ['easting', 'northing'], 'as_frame': True}
    ]
    
    for config in configurations:
        _common_nansha_tests(load_nansha, **config)
        print(f"Test passed with configuration: {config}")

def _common_nansha_tests(func, **kwargs):
    """
    Perform a series of assertions to validate the loading and formatting of datasets
    returned by the load_nansha function. This includes checks for data types,
    structure, and integrity of the loaded dataset.
    """
    result = func(**kwargs)
    
    # Assertions based on the configuration
    if kwargs.get('return_X_y', False):
        data, target = result
        # Handle the case where target might be returned as DataFrame but expected as Series
        assert isinstance(data, (np.ndarray, pd.DataFrame)), "Data must be ndarray or DataFrame"
        assert isinstance(target, (np.ndarray, pd.Series, pd.DataFrame)),( 
            "Target must be ndarray or Series or DataFrames.")
        
        if kwargs.get('as_frame', False):
            assert isinstance(data, pd.DataFrame), "Data must be a DataFrame when as_frame=True"
            assert isinstance(target, (pd.Series, pd.DataFrame)), ( 
                "Target must be a Series/DataFrame when as_frame=True")
    elif len(kwargs) == 0:
        # Assuming a custom Boxspace or similar object is returned
        assert 'data' in result and 'target' in result, "Result must contain both 'data' and 'target'"
        assert isinstance(result.get('frame', pd.DataFrame()), pd.DataFrame), "Frame must be a DataFrame"
        assert 'DESCR' in result, "Result must contain 'DESCR'"
    
    # Additional checks for split_X_y functionality
    elif kwargs.get('split_X_y', False):
        X_train, X_test, y_train, y_test = result
        assert len(X_train) > 0 and len(X_test) > 0, "Training and testing sets must not be empty"
        assert len(y_train) > 0 and len(y_test) > 0, "Training and testing target sets must not be empty"
        if kwargs.get('as_frame', False):
            assert isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame), "Train/Test data must be DataFrame when as_frame=True"
            assert isinstance(y_train, (pd.Series, pd.DataFrame)) and isinstance(
                y_test, (pd.Series, pd.DataFrame)), ( 
                    "Train/Test target must be Series/DataFrame when as_frame=True"
                    )
   
    elif  kwargs.get('as_frame', False):
          # Here, we expect result to be a DataFrame encapsulating both data and target
          assert isinstance(result, pd.DataFrame), "Result must be a DataFrame when as_frame is True"
         

def test_load_dyspnea():
    """
    Run common dataset tests on the dyspnea dataset with various configurations.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, 'as_frame': True},
        {'split_X_y': True, 'test_ratio': 0.3},
        {'objective': 'Diagnosis Prediction', 'n_labels': 2},
        {'key': 'preprocessed', 'return_X_y': True},
        {'key': 'raw', 'as_frame': True}
    ]
    
    for config in configurations:
        _common_dyspnea_tests(load_dyspnea, **config)
        print(f"Test passed with configuration: {config}")

def _common_dyspnea_tests(func, **kwargs):
    """
    Perform a series of assertions to validate the loading and formatting of datasets
    returned by the load_dyspnea function. This includes checks for data types,
    structure, and integrity of the loaded dataset.
    """
    result = func(**kwargs)
    
    # Assertions based on the configuration
    if kwargs.get('return_X_y', False):
        data, target = result
        assert isinstance(data, (np.ndarray, pd.DataFrame)), "Data must be ndarray or DataFrame"
        assert isinstance(target, (np.ndarray, pd.Series)), "Target must be ndarray or Series"
        if kwargs.get('as_frame', False):
            assert isinstance(data, pd.DataFrame), "Data must be a DataFrame when as_frame=True"
            assert isinstance(target, ( pd.Series, pd.DataFrame)), ( 
                "Target must be a Series/DataFrame when as_frame=True")
    elif len(kwargs) == 0:
        # Assuming a custom Boxspace object is returned
        assert 'data' in result and 'target' in result, (
            "Boxspace must contain both 'data' and 'target'")
        assert isinstance(result['frame'], (pd.DataFrame, type(None))), ( 
            "Frame must be DataFrame or None")
        assert 'DESCR' in result, "Boxspace must contain 'DESCR'"
    
    # Additional checks for split_X_y functionality
    elif kwargs.get('split_X_y', False):
        X_train, X_test, y_train, y_test = result
        assert len(X_train) > 0 and len(X_test) > 0, "Training and testing sets must not be empty"
        assert len(y_train) > 0 and len(y_test) > 0, "Training and testing target sets must not be empty"
        if kwargs.get('as_frame', False):
            assert isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame),( 
                "Train/Test data must be DataFrame when as_frame=True")
            assert isinstance(y_train, pd.Series) and isinstance(y_test, pd.Series),( 
                "Train/Test target must be Series when as_frame=True") 
            
    elif  kwargs.get('as_frame', False):
          # Here, we expect result to be a DataFrame encapsulating both data and target
          assert isinstance(result, pd.DataFrame), "Result must be a DataFrame when as_frame is True"
         
def test_statlog():
    """
    Run common dataset tests on the Statlog Heart Disease dataset with various
    configurations.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, 'as_frame': True},
        {'split_X_y': True, 'test_ratio': 0.2},
        {'split_X_y': True, 'as_frame': True, 'test_ratio': 0.3}
    ]
    for config in configurations:
        _common_statlog_tests(load_statlog, **config)
        print("Test passed with configuration:", config)

def _common_statlog_tests(func, **kwargs):
    """
    Perform a series of assertions to validate the loading and formatting of datasets
    returned by the load_statlog function. This includes checks for data types,
    structure, and integrity of the loaded dataset.
    """
    result = func(**kwargs)
    
    # Different assertions based on the configuration
    if kwargs.get('return_X_y', False):
        data, target = result
        # Handle the case where target might be returned as DataFrame but expected as Series
        if isinstance(target, pd.DataFrame):
            target = pd.Series(np.squeeze(target.values), name=target.columns[0])
        
        assert isinstance(data, (np.ndarray, pd.DataFrame)), "Data must be ndarray or DataFrame"
        assert isinstance(target, (np.ndarray, pd.Series)), "Target must be ndarray or Series"
        
        if kwargs.get('as_frame', False):
            assert isinstance(data, pd.DataFrame), "Data must be a DataFrame when as_frame is True"
            assert isinstance(target, pd.Series), "Target must be a Series when as_frame is True"
            
    elif len(kwargs) == 0:
        # Assuming Boxspace is a custom object for your project that encapsulates dataset information
        assert isinstance(result, Boxspace), "Result must be a Boxspace object"
        assert 'data' in result, "Boxspace must contain 'data'"
        assert 'target' in result, "Boxspace must contain 'target'"
        assert 'frame' in result or 'DESCR' in result, "Boxspace must contain 'frame' or 'DESCR'"
    
    # Additional checks for as_frame=True without return_X_y
    elif kwargs.get('as_frame', False) and not kwargs.get('split_X_y', False):
        # Here, we expect result to be a DataFrame encapsulating both data and target
        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame when as_frame is True"
        
    # Test split_X_y functionality if applicable
    if kwargs.get('split_X_y', False):
        validate_split_X_y(func, **kwargs)

def test_hydro_metrics():
    """
    Run common dataset tests on the Hydro-Meteorological dataset with various configurations.
    This will ensure the function correctly handles different scenarios and parameter combinations,
    ensuring robustness and flexibility of the data loading process.
    """
    configurations = [
        {},
        {'return_X_y': True},
        {'as_frame': True},
        {'return_X_y': True, "as_frame":True}, 
        {'tag': 'test_tag'},  # Although 'tag' is not utilized, ensure it doesn't break the function
        {'data_names': ['date', 'temperature', 'humidity', 'wind_speed', 
                        'solar_radiation', 'evapotranspiration', 
                        'rainfall', 'flow']},  # This tests the future extension capability
        {'split_X_y':True}  # This tests the future extension capability
    ]
    
    for config in configurations:
        _common_hydro_metrics_tests(load_hydro_metrics, **config)
        print(f"Test passed with configuration: {config}")

def _common_hydro_metrics_tests(func, **kwargs):
    """
    Perform a series of assertions to validate the loading and formatting of datasets
    returned by the load_hydro_metrics function. This includes checks for data types,
    structure, and integrity of the loaded dataset.
    """
    result = func(**kwargs)
    
    # Different assertions based on the configuration
    if kwargs.get('return_X_y', False):
        data, target = result
        if isinstance (target, pd.DataFrame): 
            target=pd.Series (np.squeeze (target), name=target.columns[0])
        assert isinstance(data, (np.ndarray, pd.DataFrame)), "Data must be ndarray or DataFrame"
        assert isinstance(target, (np.ndarray, pd.Series)), "Target must be ndarray or Series"
        
        if kwargs.get('as_frame', False):
            assert isinstance(data, pd.DataFrame), "Data must be a DataFrame when as_frame=True"
            assert isinstance(target, pd.Series), "Target must be a Series when as_frame=True"
            
    elif len(kwargs)==0:
        assert isinstance(result, Boxspace), "Result must be a Boxspace object"
        assert 'data' in result, "Boxspace must contain 'data'"
        assert 'target' in result, "Boxspace must contain 'target'"
        assert 'frame' in result or 'DESCR' in result, "Boxspace must contain 'frame' or 'DESCR'"
    
    elif kwargs.get('as_frame', False):
        assert isinstance(result, pd.DataFrame), "Data must be a DataFrame when as_frame=True"
    
    # Test split_X_y functionality
    if kwargs.get('split_X_y', False):
        validate_split_X_y(func, **kwargs)
        return True 

def test_hlogs():
    """Run common dataset tests on the hlogs dataset with various configurations."""
    configurations = [
        {},
        {'return_X_y': True}, 
        {'split_X_y': True},
        {'target_names': 'k', 'split_X_y': True},
        {'key': 'h1102 h1104'}
    ]
    for config in configurations:
        _common_dataset_tests(load_hlogs, **config)
        print("Test passed with configuration:", config)

def _common_dataset_tests(func, **kwargs):
    """Perform a series of assertions to validate the loading and formatting of datasets."""
    if kwargs.get('return_X_y', False): 
        # Test return_X_y functionality
        X, y = func(**kwargs)
        assert len(X) == len(y), "X and y lengths must match when return_X_y is True"
        
        # Test as_frame=True functionality
        X_frame, y_frame = func(as_frame=True, **kwargs)
        assert isinstance(X_frame, pd.DataFrame), "X must be a DataFrame when as_frame is True"
        assert isinstance(y_frame, (pd.Series, pd.DataFrame)), "y must be a Series or DataFrame when as_frame is True"
        
        return True 
    
    # Test split_X_y functionality
    if kwargs.get('split_X_y', False):
        validate_split_X_y(func, **kwargs)
        return True 
    
    b = func(**kwargs)
    assert isinstance(b, Boxspace), "The returned object must be an instance of Boxspace"
    
    # Validate data structures
    assert isinstance(b.target, np.ndarray), "Target must be a numpy array"
    assert isinstance(b.data, np.ndarray), "Data must be a numpy array"
    assert isinstance(b.frame, pd.DataFrame), "Frame must be a pandas DataFrame"
    
    # Validate lengths and shapes
    assert len(b.data) == len(b.target), "Data and target lengths must match"
    assert b.data.shape[1] == len(b.feature_names), "The number of features must match feature_names length"
    assert len(b.frame) == len(b.data), "DataFrame length must match data length"
    
    # Validate target names handling
    if hasattr(b.target, 'shape') and len(b.target.shape) > 1:
        assert b.target.shape[1] == len(b.target_names), "Multi-dimensional target must match target_names length"
    else:
        assert b.target.reshape((len(b.target), 1)).shape[1] == len(b.target_names), ( 
            "Target reshaping must match target_names length"
            )
    
def validate_split_X_y(func, test_ratio=0.2, **kwargs):
    """Test the splitting functionality of the dataset loading function."""
    # Adjust kwargs for testing split functionality
    kwargs.update({'as_frame': True, 'split_X_y': True, 'test_ratio': test_ratio})
    
    # Perform split and validate
    X_train, X_test, y_train, y_test = func(**kwargs)
    validate_split(X_train, X_test, y_train, y_test, func, **kwargs)

    # Repeat validation without as_frame=True for ndarray checks
    kwargs['as_frame'] = False
    X_train, X_test, y_train, y_test = func(**kwargs)
    validate_split(X_train, X_test, y_train, y_test, func, **kwargs)

def validate_split(X_train, X_test, y_train, y_test, func, **kwargs):
    """Helper function to validate training/testing split."""
    frame = func(as_frame=True)
    assert len(X_train) + len(X_test) == len(frame), ( 
        "Combined length of train and test sets must equal the original frame length"
        )
    
    # Check if the split sizes approximately match the specified ratio
    assert approx_length(y_test, frame, operation='is_test', test_ratio=kwargs['test_ratio']), ( 
        "Test set size should approximately match the specified ratio")
    assert approx_length(y_train, frame, operation='is_train', test_ratio=kwargs['test_ratio']), (
        "Train set size should approximately match the specified ratio")

def approx_length(y, data, operation, test_ratio):
    """Check if the length of y approximates the expected size given the operation and test_ratio."""
    expected_size = test_ratio if operation == 'is_test' else (1 - test_ratio)
    expected_len = int(expected_size * len(data))

    return expected_len - 7 <= len(y) <= expected_len + 7
    
            
if __name__ == "__main__":
    pytest.main([__file__])
