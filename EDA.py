# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    for i,c in enumerate(df.columns.tolist()):
        df_info.loc[i] = {"feature_name": c ,
                          "nunique": df[c].nunique,
                          "missing_value_per": df[c].isnull().sum()/len(df),
                          "majority_class_per": df[c].value_counts().iloc[0]/len(df),
                          "dtype":df[c].dtypes}
    df_info['feature_name'] = df.columns.tolist()
    df_info['nunique'] = [df[col].nunique() for col in df.columns.tolist()]
    df_info['missing_value_per'] = [df[col].isnull().sum()/len(df) for col in df.columns.tolist()]
    
    return df_info