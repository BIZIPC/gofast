# -*- coding: utf-8 -*-
"""
test_baseutils.py
"""
import tempfile
import os
import unittest
import pytest # noqa 
from unittest.mock import patch, mock_open 
import numpy as np 
import pandas as pd 

from gofast.tools.coreutils import cleaner, smart_label_classifier
from gofast.datasets.load import load_bagoue
from gofast.tools.baseutils import download_file
from gofast.tools.baseutils import lowertify
from gofast.tools.baseutils import fancier_downloader
from gofast.tools.baseutils import save_or_load
from gofast.tools.baseutils import array2hdf5
from gofast.tools.baseutils import extract_target 
from gofast.tools.baseutils import labels_validator #  
from gofast.tools.baseutils import rename_labels_in  # 
from gofast.tools.baseutils import select_features 
from gofast.tools.baseutils import categorize_target
from gofast.tools.baseutils import get_target

DOWNLOAD_FILE='https://raw.githubusercontent.com/WEgeophysics/gofast/main/gofast/datasets/data/iris.csv'


# This test case is conceptual and may need adjustments for real-world usage
class TestSaveOrLoad(unittest.TestCase):
    @patch('gofast.tools.baseutils.np.save')
    def test_save_array(self, mock_save):
        arr = np.array([1, 2, 3])
        save_or_load("dummy.npy", arr, task='save', format='.npy')
        mock_save.assert_called_once()

    @patch('gofast.tools.baseutils.np.load')
    def test_load_array(self, mock_load):
        mock_load.return_value = np.array([1, 2, 3])
        result = save_or_load("dummy.npy", task='load')
        self.assertTrue(np.array_equal(result, np.array([1, 2, 3])))
        mock_load.assert_called_once_with("dummy.npy")
        
@pytest.mark.skip(reason="If Module Not found")
class TestDownloadFile(unittest.TestCase):
    @patch('gofast.tools.baseutils.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_file(self, mock_file, mock_get):
        mock_get.return_value.__enter__.return_value.iter_content.return_value = [b'chunk1', b'chunk2']
        download_file(DOWNLOAD_FILE, 'iris.csv')
        mock_file.assert_called_with('iris.csv', 'wb')
        handle = mock_file()
        handle.write.assert_any_call(b'chunk1')
        handle.write.assert_any_call(b'chunk2')
        
@pytest.mark.skip(reason="If Module Not found")
class TestFancierDownloader(unittest.TestCase):
    @patch('gofast.tools.baseutils.requests.get')
    def test_fancier_downloader(self, mock_get):
        mock_get.return_value.__enter__.return_value.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value.__enter__.return_value.headers = {'content-length': '1024'}
        
        with patch('gofast.tools.baseutils.tqdm'), \
              patch('builtins.open', mock_open()) as mocked_file:
            local_filename = fancier_downloader(DOWNLOAD_FILE, 'iris.csv')
            mocked_file.assert_called_once_with('iris.csv', 'wb')
            self.assertIn('iris.csv', local_filename)
            
def test_get_target():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': [0, 1, 0]
    })
    target, modified_df = get_target(df, 'target', True)
    
    assert 'target' not in df.columns  # Original DataFrame should remain unchanged
    assert 'target'  not in modified_df.columns  # Modified DataFrame should not have the target column

def test_get_target2():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': ['A', 'B', 'C']
    })
    
    target, modified_df = get_target(df, 'target', inplace=False)
    assert 'target' in df.columns  # Original df should be unchanged
    
def test_get_target_inplace_true():
    df = pd.DataFrame({'feature': [1, 2, 3], 'target': ['A', 'B', 'C']})
    target, _ = get_target(df, 'target')
    assert 'target' not in df.columns
    assert all( item in target.values for item in  ['A', 'B', 'C'])

def test_get_target_inplace_false():
    df = pd.DataFrame({'feature': [1, 2, 3], 'target': ['A', 'B', 'C']})
    _, new_df = get_target(df, 'target', inplace=False)
    assert 'target' in df.columns
    assert 'target'in new_df.columns

def test_select_features_by_name():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = select_features(df, features=['A', 'C'])
    assert list(result.columns) == ['A', 'C']

def test_select_features_include_exclude():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['x', 'y', 'z']})
    result = select_features(df, include=['number'])
    assert 'C' not in result.columns
    result = select_features(df, exclude=['number'])
    assert list(result.columns) == ['C']

def test_extract_target_dataframe():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': ['A', 'B', 'C']
    })
    target, modified_df = extract_target(df, 'target', return_y_X=True)
    assert len(target) == len(['A', 'B', 'C'])
    assert 'target' not in modified_df.columns

def test_extract_target_array():
    arr = np.array([[1, 2, 'A'], [3, 4, 'B'], [5, 6, 'C']])
    target, modified_arr = extract_target(arr, 2,  columns=['feature1', 'feature2', 'target'], 
                                          return_y_X= True )
    assert np.array_equal(np.squeeze (target), np.array(['A', 'B', 'C'])) # for consistency
    assert modified_arr.shape == (3, 2)  # Ensure target column was removed

def test_categorize_target_with_function():
    arr = np.array([1, 2, 3, 4, 5])
    def categorize_func(x): return 0 if x <= 3 else 1
    categorized_arr = categorize_target(arr, func=categorize_func)
    assert np.array_equal(categorized_arr, np.array([0, 0, 0, 1, 1]))

def test_categorize_target_with_labels():
    arr = np.array([1, 2, 3, 4, 5])
    categorized_arr = categorize_target(arr, labels=2)
    # Assuming the function divides the range into equal intervals
    assert len(np.unique(categorized_arr)) == 2

def test_categorize_target_with_rename_labels():
    arr = np.array([1, 2, 3, 4, 5])
    categorized_arr = categorize_target(arr, labels=2, rename_labels=['A', 'B'], coerce=True)
    assert set(categorized_arr) == {'A', 'B'}
    
def test_labels_validator_with_existing_labels():
    y = np.array([0, 1, 2, 0, 1, 2])
    labels = [0, 1, 2]
    assert labels_validator(y, labels, return_bool=True)

def test_labels_validator_with_missing_labels():
    y = np.array([0, 1, 0, 1])
    labels = [0, 1, 2]
    assert not labels_validator(y, labels, return_bool=True)

def test_labels_validator_raises_value_error_on_missing_labels():
    y = np.array([0, 1, 0, 1])
    labels = [0, 1, 2]
    try:
        labels_validator(y, labels)
    except ValueError as e:
        assert str(e).startswith("Label")

def test_rename_labels_in():
    arr = np.array([0, 1, 2, 0, 1, 2])
    new_labels = ["A", "B", "C"]
    expected = np.array(["A", "B", "C", "A", "B", "C"])
    result = rename_labels_in(arr, new_labels)
    np.testing.assert_array_equal(result, expected)

def test_rename_labels_in_with_coerce():
    arr = np.array([0, 1, 2, 0, 1, 2])
    new_labels = ["A", "B"]  # Intentionally missing label for "2"
    expected = np.array(["A", "B", 2, "A", "B", 2])  # Original "2" labels should remain unchanged
    result = rename_labels_in(arr, new_labels, coerce=True)
    np.testing.assert_array_equal(result, expected)

def test_select_features2():
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'C'],
        'feature3': [0.1, 0.2, 0.3]
    })
    selected_data = select_features(data, features=['feature1', 'feature3'])
    assert list(selected_data.columns) == ['feature1', 'feature3']
    
# get the data for a test 
def _prepare_dataset ( return_encoded_data =False, return_raw=False ): 
    from gofast.tools.mlutils import ( 
        soft_imputer, 
        soft_scaler, bi_selector, make_pipe,
        ) 
    X, y = load_bagoue (as_frame =True, return_X_y= True  )
    
    
    if return_raw: 
        return X, y 
    # prepared data 
    # 1-clean data 
    cleaned_data = cleaner (X , columns = 'name num lwi', mode ='drop')
    num_features, cat_features= bi_selector (cleaned_data)
    # categorizing the labels 
    yc = smart_label_classifier (y , values = [1, 3], 
                                     labels =['FR0', 'FR1', 'FR2'] 
                                     ) 
    data_imputed = soft_imputer(cleaned_data,  mode= 'bi-impute')
    num_scaled = soft_scaler (data_imputed[num_features],) 
    # print(num_scaled)
    # print(cleaned_data)
    # we can rencoded the target data from `make_naive_pipe` as 
    Xenc, yenc= make_pipe ( cleaned_data, y = yc ,  transform=True )
    if return_encoded_data : 
        return Xenc, yenc 
    
    return num_scaled, yenc 

def test_categorize_data (): 
    def binfunc(v): 
        if v < 3 : return 0 
        else : return 1 
    arr = np.arange (10 )
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    categorize_target(arr, func =binfunc)
    # array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    # array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    # array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    categorize_target(arr, labels =3 , order =None )
    # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    categorize_target(arr[::-1], labels =3 , order =None )
    # array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    categorize_target(arr, labels =[0 , 2,  4]  )
    # array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])
    
def store_data  (as_frame =False,  task='None', return_X_y=False ): 
    from gofast.tools.mlutils import ( 
        soft_imputer, 
        soft_scaler, 
        bi_selector, 
        make_pipe, 
        codify_variables, 
        ) 
    
    def bin_func ( x): 
        if x ==1 or x==2: 
            return 1 
        else: return 0 
    # ybin = categorize_target( y, func = func_clas)
        
    def func_ (x): 
        if x<=1: return 0 
        elif x >1 and x<=3: 
            return 1 
        else: return 2 
    X, y = load_bagoue (as_frame =True , return_X_y= True )
    
    if str(task).lower().find('bin')>=0: 
        y = categorize_target ( y, func= bin_func)
        # y = np.array (y )
        # y [y <=1] = 0;  y [y > 0]=1 
        if as_frame : 
            y = pd.Series ( y, name ='flow') 
    else: 
        y= categorize_target ( y, func= func_)
        
    
    cleaned_data = cleaner (X , columns = 'name num lwi', mode ='drop')
    #$print(cleaned_data.columns)
    num_features, cat_features= bi_selector (cleaned_data)
    # categorizing the labels 
    data_imputed = soft_imputer(cleaned_data,  mode= 'bi-impute')
    num_scaled = soft_scaler (data_imputed[num_features],) 
    #print(num_scaled.columns)
    # we can rencoded the target data from `make_naive_pipe` as 
    pipe= make_pipe ( cleaned_data, y = y  )
    Xenc, yenc= make_pipe ( cleaned_data, y = y ,  transform=True )

    Xr, _= _prepare_dataset(return_raw= True ) 

    Xr = cleaner ( Xr, columns = 'name num lwi', mode ='drop' )
    # get the categorical variables 
    num_var , cat_var = bi_selector ( Xr )
    
    Xcoded = codify_variables (Xr, columns = cat_var )
    # get the categ
    Xnew = pd.concat ((X[num_var], Xcoded), axis = 1 )
    Xanalysed= pd.concat ( (num_scaled, Xcoded), axis=1 )
    

    data = {"preprocessed": ( num_scaled, y ), 
      "encoded": (Xenc, yenc),
      "codified": ( Xnew, y ), 
      "analysed": (Xanalysed, y  ), 
      "pipe": pipe, 
          }
    return data 

# Note: This test assumes h5py is installed and a temporary file can be written and read
class TestArray2HDF5(unittest.TestCase):
    def test_array2hdf5_store_and_load(self):
        arr = np.array([[1, 2], [3, 4]])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            tmp_name = tmp.name
        array2hdf5(tmp_name, arr, task='store')
        loaded_arr = array2hdf5(tmp_name, task='load')
        os.remove(tmp_name)  # Clean up the temporary file
        self.assertTrue(np.array_equal(arr, loaded_arr))

class TestLowertify(unittest.TestCase):
    def test_lowertify(self):
        result = lowertify("Test", "STRING", 123)
        self.assertEqual(result, ('test', 'string', '123'))
        test_result, string_result= lowertify("Test", "STRING", return_origin=True)
        self.assertEqual(test_result, ('test', 'Test'))
        self.assertEqual(string_result, ('string', 'STRING'))



def test_select_features (): 
    X, _= _prepare_dataset(return_raw= True ) 
    select_features(X, exclude='number')
    select_features(  X, include="number") 
    select_features (X, features = 'ohmS num shape geol lwi', 
          parse_features =True ) 


if __name__=='__main__': 
    pytest.main([__file__])



