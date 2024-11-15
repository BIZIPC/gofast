# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:26:17 2023
Author: Daniel

Testing utilities of :mod:`gofast.tools.coreutils`
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt

from gofast.datasets.load import load_bagoue
from gofast.tools.baseutils import normalizer, interpolate_grid
from gofast.tools.coreutils import reshape, to_numeric_dtypes
from gofast.tools.datautils import (
    cleaner, random_selector,
)
from gofast.tools.mlutils import smart_label_classifier
from gofast.tools.baseutils import remove_outliers

# Load data for testing
X, y = load_bagoue(as_frame=True, return_X_y=True)

def test_to_numeric_dtypes():
    """ Test conversion of data to numeric types """
    X0 = X[['shape', 'power', 'magnitude']]
    print(X0.dtypes)
    # Display data types and check the converted data types
    print(to_numeric_dtypes(X0))


def test_reshape():
    """ Test reshaping functionality of arrays """
    np.random.seed(0)
    array = np.random.randn(50)
    print(array.shape)
    
    # Reshaping the array into (1, 50)
    ar1 = reshape(array, 1)
    print(ar1.shape)
    
    # Reshaping the array into (50, 1)
    ar2 = reshape(ar1, 0)
    print(ar2.shape)
    
    # Reshaping back to the original shape (50,)
    ar3 = reshape(ar2, axis=None)
    print(ar3.shape)


def test_smart_label_classifier():
    """ Test smart labeling functionality """
    sc = np.arange(0, 7, 0.5)
    
    # Default label classification based on given values
    smart_label_classifier(sc, values=[1, 3.2])
    # Expected output: array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=int64)
    
    # Custom labels for classification
    smart_label_classifier(sc, values=[1, 3.2], labels=['l1', 'l2', 'l3'])
    # Expected output: array(['l1', 'l1', 'l1', 'l2', 'l2', 'l2', 'l2', 'l3', ...], dtype=object)
    
    # Classification using a custom function
    def f(v):
        if v <= 1:
            return 'l1'
        elif 1 < v <= 3.2:
            return "l2"
        else:
            return "l3"
    smart_label_classifier(sc, func=f)
    # Expected output: array(['l1', 'l1', 'l1', 'l2', 'l2', 'l2', 'l2', 'l3', ...], dtype=object)
    
    # Classifying based on a single value
    smart_label_classifier(sc, values=1.0)
    # Expected output: array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    
    smart_label_classifier(sc, values=1.0, labels='l1')
    # Expected output: array(['l1', 'l1', 'l1', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=object)


def test_remove_outliers():
    """ Test removal of outliers in data """
    data = np.random.randn(7, 3)
    data_r = remove_outliers(data)
    print(data.shape, data_r.shape)
    # Expected output: (7, 3) (5, 3)
    
    # Removing outliers and filling with NaN
    remove_outliers(data, fill_value=np.nan)
    # Expected output: array([[ 0.49671415, -0.1382643 ,  0.64768854], ..., [nan, nan]])
    
    # Handling one-dimensional data
    remove_outliers(data[:, 0], fill_value=np.nan)
    # Expected output: array([ 0.49671415, 1.52302986, 1.57921282, 0.54256004, 0.24196227, ..., nan])
    
    # Interpolation to replace outliers with interpolated values
    data_r0 = remove_outliers(data[:, 0], fill_value=np.nan, interpolate=True)
    
    plt.plot(np.arange(len(data)), data, 'ok')
    plt.plot(np.arange(len(data)), data[:, 0], 'ok-', np.arange(len(data_r0)), data_r0, 'or-')


def test_normalizer():
    """ Test normalization of data """
    np.random.seed(42)
    arr = np.random.randn(3, 2)
    # Expected output: array([[ 0.49671415, -0.1382643 ], [ 0.64768854, 1.52302986], ...])
    normalizer(arr)
    
    # Normalizing along axis=0
    normalizer(arr, method='01')
    # Expected output: array([[0.82879654, 0.05456093], [1., 1.], [0., 0.]])
    
    # Handling NaN values during normalization
    arr[0, 1] = np.nan
    arr[1, 0] = np.nan
    normalizer(arr, allow_nan= True)
    # Expected output: array([[ 0.41593131, nan], [nan, 1.], [ 0., 9.34323403e-06]])
    
    # Normalizing with NaN values along axis=0
    normalizer(arr, method='01', allow_nan= True)


def test_cleaner():
    """ Test cleaning of data """
    cleaner(X, columns='num, ohmS')
    cleaner(X, mode='drop', columns='power shape, type')


def test_random_selector():
    """ Test random selection of data """
    dat = np.arange(42)
    
    # Select 7 random values
    random_selector(dat, 7, seed=42)
    # Expected output: array([0, 1, 2, 3, 4, 5, 6])
    
    # Select multiple random values
    random_selector(dat, (23, 13, 7))
    # Expected output: array([7, 13, 23])
    
    # Select random values based on percentage
    random_selector(dat, "7%", seed=42)
    # Expected output: array([0, 1])
    
    # Select 70% of the data, with shuffling
    random_selector(dat, "70%", seed=42, shuffle=True)
    # Expected output: array([0, 5, 20, 25, 13, 7, 22, 10, 12, 27, 23, 21, 16, 3, 1, 17, 8, 6, 4, 2, 19, 11, 18, 24, 14, 15, 9, 28, 26])


def test_interpolate_grid():
    """ Test interpolation of data on a grid """
    x = [28, np.nan, 50, 60]
    y = [np.nan, 1000, 2000, 3000]
    xy = np.vstack((x, y)).T
    xyi = interpolate_grid(xy, view=True)
    print(xyi)
    # Expected output: array([[28., 28.], [22.78880663, 1000.], [50., 2000.], [60., 3000.]])

    
if __name__=="__main__":
    pytest.main([__file__])

    
    
    
    
    
    
    
    
    
    
    
    
    