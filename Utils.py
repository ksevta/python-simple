# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def load_data(folder_path = None, file_path = None, reduce_mem_usage_ = True, dtype = None):
    if ((not file_path) & (not folder_path)):
        raise ValueError("Folder Path And File Path Both Missing")
    if (file_path):
        df = pd.read_csv(file_path, dtype = dtype);
        if reduce_mem_usage_:
            df = reduce_mem_usage(df, verbose = True)
        return df
    else:
        print(os.listdir(folder_path))
        for f in os.listdir(folder_path):
            f_path = folder_path+f
            if "train" in f.lower():
                train = pd.read_csv(f_path, dtype=dtype)
                if reduce_mem_usage_:
                    train = reduce_mem_usage(train,verbose=True)
            elif "test" in f.lower():
                test = pd.read_csv(f_path,dtype=dtype)
                if reduce_mem_usage_:
                    test = reduce_mem_usage(test,verbose=True)
        return train, test   
    
def print_details(df, target_feature = None):
    
    print("Total number of samples: ",len(df))
    print("Total number of features: ", df.shape[1])
    print("Total number of categorical features: ", df.select_dtypes(include=["object"]).shape[1])
    print("Total number of numerical features: ",df.select_dtypes(exclude=["object"]).shape[1])
    
    if target_feature:
        print("Total unique values in target class: ",df[target_feature].nunique())
        print("Target Class Distribution:")
        print(df[target_feature].value_counts()[:10]/len(df))
    print(df.describe())
    
def get_details(df):
    df_info = pd.DataFrame(columns = ['feature_name','nunique',
                                      'missing_value_per','majority_class_per',
                                      'dtype'])
    for i,col in enumerate(df.columns.tolist()):
        df_info.loc[i] = {"feature_name": col ,
                          "nunique": df[col].nunique,
                          "missing_value_per": df[col].isnull().sum()/len(df),
                          "majority_class_per": df[col].value_counts().iloc[0]/len(df),
                          "dtype":df[col].dtypes}
    df_info['feature_name'] = df.columns.tolist()
    df_info['nunique'] = [df[col].nunique() for col in df.columns.tolist()]
    df_info['missing_value_per'] = [df[col].isnull().sum()/len(df) for col in df.columns.tolist()]
    
    return df_info

def data_clean(df):
    
    # duplicate columns
    df = df.T.drop_duplicates(keep='first').T
    
    # constant columns
    cols = df.columns.tolist()
    cols.remove([c for c in cols if df[c].nunique(dropna=False) == 1])
    df = df[cols]
    return df
