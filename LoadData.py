# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import numpy as np
import os

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