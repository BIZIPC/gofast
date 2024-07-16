# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:26:17 2023
@author: Daniel

Testing some utilities of :mod:`gofast.tools.coreutils` 
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from gofast.tools.coreutils import  ( 
    reshape, 
    to_numeric_dtypes, 
    smart_label_classifier, 
    cleaner, 
    random_selector, 
    pair_data, 
    random_sampling, 
    replace_data, 
    ) 
from gofast.tools.baseutils import normalizer, interpolate_grid
from gofast.datasets.load import load_bagoue
from gofast.datasets.load import load_hlogs
from gofast.tools.baseutils import remove_outliers 
# get the data for a test 
X, y = load_bagoue (as_frame =True , return_X_y=True ) 

def test_to_numeric_dtypes (): 
    """ test data types """
    X0 =X[['shape', 'power', 'magnitude']]
    print(X0.dtypes)
    # print X0.dtypes and check the 
    # datatypes 
    print( to_numeric_dtypes(X0)) 
    
def test_reshape (): 
    np.random.seed (0) 
    array = np.random.randn(50 )
    print(array.shape)
    ar1=reshape(array, 1)
    print(ar1.shape) 
    # ... (1, 50)
    ar2 =reshape(ar1 , 0) 
    print(ar2.shape) 
    # ... (50, 1)
    ar3 = reshape(ar2, axis = None)
    print( ar3.shape) # goes back to the original array  
    # ... (50,)
def test_smart_label_classifier () :
    
    sc = np.arange (0, 7, .5 ) 
    smart_label_classifier (sc, values = [1, 3.2 ]) 
    # array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=int64)
    # >>> # rename labels <=1 : 'l1', ]1; 3.2]: 'l2' and >3.2 :'l3'
    smart_label_classifier (sc, values = [1, 3.2 ], labels =['l1', 'l2', 'l3'])
    # >>> array(['l1', 'l1', 'l1', 'l2', 'l2', 'l2', 'l2', 'l3', 'l3', 'l3', 'l3',
    #        'l3', 'l3', 'l3'], dtype=object)
    def f (v): 
        if v <=1: return 'l1'
        elif 1< v<=3.2: return "l2" 
        else : return "l3"
    smart_label_classifier (sc, func= f )
    # array(['l1', 'l1', 'l1', 'l2', 'l2', 'l2', 'l2', 'l3', 'l3', 'l3', 'l3',
    #        'l3', 'l3', 'l3'], dtype=object)
    smart_label_classifier (sc, values = 1.)
    # array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    smart_label_classifier (sc, values = 1., labels='l1')  
    # array(['l1', 'l1', 'l1', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=object)
    
def test_remove_outliers () : 
    data = np.random.randn (7, 3 )
    data_r = remove_outliers ( data )
    print( data.shape , data_r.shape )
    # (7, 3) (5, 3)
    remove_outliers ( data, fill_value =np.nan )
    # array([[ 0.49671415, -0.1382643 ,  0.64768854],
    #        [ 1.52302986, -0.23415337, -0.23413696],
    #        [ 1.57921282,  0.76743473, -0.46947439],
    #        [ 0.54256004, -0.46341769, -0.46572975],
    #        [ 0.24196227,         nan,         nan],
    #        [-0.56228753, -1.01283112,  0.31424733],
    #        [-0.90802408,         nan,  1.46564877]])
    # >>> # for one dimensional 
    remove_outliers ( data[:, 0] , fill_value =np.nan )
    # array([ 0.49671415,  1.52302986,  1.57921282,  0.54256004,  0.24196227,
    #        -0.56228753,         nan])
    data_r0 =remove_outliers ( data[:, 0] , fill_value =np.nan, interpolate=True  )
    plt.plot (np.arange (len(data )), data, 'ok') 
    plt.plot (np.arange (len(data )), data[:, 0], 'ok-', 
           np.arange (len(data_r0 )), data_r0, 'or-'
           )
    
    
def test_normalizer(): 
    np.random.seed (42)
    arr = np.random.randn (3, 2 ) 
    # array([[ 0.49671415, -0.1382643 ],
    #        [ 0.64768854,  1.52302986],
    #        [-0.23415337, -0.23413696]])
    normalizer (arr )
    # array([[4.15931313e-01, 5.45697636e-02],
    #        [5.01849720e-01, 1.00000000e+00],
    #        [0.00000000e+00, 9.34323403e-06]])
    normalizer (arr , method ='min-max')  # normalize data along axis=0 
    # array([[0.82879654, 0.05456093],
    #        [1.        , 1.        ],
    #        [0.        , 0.        ]])
    arr [0, 1] = np.nan; arr [1, 0] = np.nan 
    normalizer (arr )
    # array([[4.15931313e-01,            nan],
    #        [           nan, 1.00000000e+00],
    #        [0.00000000e+00, 9.34323403e-06]])
    normalizer (arr , method ='min-max')
    
def test_cleaner (): 
    
    cleaner (X,  columns = 'num, ohmS')
    cleaner (X, mode ='drop', columns ='power shape, type')
    
    
def test_random_selector (): 
    dat= np.arange (42 ) 
    random_selector (dat , 7, seed = 42 ) 
    # array([0, 1, 2, 3, 4, 5, 6])
    random_selector ( dat, ( 23, 13 , 7))
    # array([ 7, 13, 23])
    random_selector ( dat , "7%", seed =42 )
    # array([0, 1])
    random_selector ( dat , "70%", seed =42 , shuffle =True )
    # array([ 0,  5, 20, 25, 13,  7, 22, 10, 12, 27, 23, 21, 16,  3,  1, 17,  8,
    #         6,  4,  2, 19, 11, 18, 24, 14, 15,  9, 28, 26])
    
    
def test_interpolate_grid(): 
    
    x = [28, np.nan, 50, 60] ; y = [np.nan, 1000, 2000, 3000]
    xy = np.vstack ((x, y)).T
    xyi = interpolate_grid (xy, view=True ) 
    print(xyi)  
    # array([[  28.        ,   28.        ],
    #        [  22.78880663, 1000.        ],
    #        [  50.        , 2000.        ],
    #        [  60.        , 3000.        ]])
    
def test_twinning (): 
    # >>> data = gf.make_erp (seed =42 , n_stations =12, as_frame =True ) 
    table1 = pd.DataFrame ( {"dipole": 10, "longitude":110.486111, 
                                 "latitude":26.05174, "shape": "C", 
                                 "type": 'EC','sfi':  1.141844, }, 
                           index =range(1))

    data_no_xy =pd.DataFrame ({"AB": [ 1.0, 2.0], 
                                   "MN": [0.4 , .4 ], 
                                   "resistivity": [448.860148, 449.060335 ]
                                   }
                                  ) 
    #     AB   MN  resistivity
    # 0  1.0  0.4   448.860148
    # 1  2.0  0.4   449.060335
    data_xy =pd.DataFrame ( 
        {"AB":[1., 1.] , 
         "MN":[ .4, .4 ],  
         "resistivity":[448.860148, 449.060335] , 
         "latitude":[28.41193, 28.41193] , 
         "longitude":[109.332931, 109.332931] 
             
             }
        )
    #     AB   MN  resistivity   longitude  latitude
    # 0  1.0  0.4   448.860148  109.332931  28.41193
    # 1  2.0  0.4   449.060335  109.332931  28.41193
    sounding_table = pd.DataFrame ( 
        { 'AB': 200.,     
         'MN':20.,    
         'arrangememt': "Schlumberger", 
         'nareas':2, 
         'longitude': 110.486111, 
         'latitude': 26.05174
            }, index =range(1))
    #          AB    MN   arrangememt  ... nareas   longitude  latitude
    # area                             ...                             
    # None  200.0  20.0  schlumberger  ...      1  110.486111  26.05174
    pair_data (table1, sounding_table,  ) 
    #        dipole   longitude  latitude  ...  nareas   longitude  latitude
    # line1    10.0  110.486111  26.05174  ...     NaN         NaN       NaN
    # None      NaN         NaN       NaN  ...     1.0  110.486111  26.05174
    # twinning (table1, sounding_table, on =['longitude', 'latitude'] ) 
    # Empty DataFrame 
    # >>> # comments: Empty dataframe appears because, decimal is too large 
    # >>> # then it considers values longitude and latitude differents 
    pair_data (table1, sounding_table, on =['longitude', 'latitude'], decimals =5 ) 
    #     dipole  longitude  latitude  ...  max_depth  ohmic_area  nareas
    # 0      10  110.48611  26.05174  ...      109.0  690.063003       1
    # >>> # Now is able to find existing dataframe with identical closer coordinates. 
    pair_data(data_no_xy, data_xy, coerce=True , parse_on= True )
    
    
def test_random_sampling (): 

    data= load_hlogs().frame
    print( random_sampling( data, samples = 7 ).shape ) 
    # (7, 27)
       
def test_replace_data (): 
    X, y = np.random.randn ( 7, 2 ), np.arange(7)
    print( X.shape, y.shape)  
    # ((7, 2), (7,))
    X_new, y_new = replace_data (X, y, n_times =10 )
    print( X_new.shape , y_new.shape) 
    
    
# if __name__=="__main__": 
    
#     test_reshape(), 
#     test_to_numeric_dtypes(), 
#     test_smart_label_classifier(), 
#     test_remove_outliers(), 
#     test_normalizer(),  
#     test_cleaner(),  
#     test_random_selector(),  
#     test_interpolate_grid(), 
#     test_twinning(), 
#     test_random_sampling(),  
#     test_replace_data(),  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    