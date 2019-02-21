#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:21:47 2019

@author: zapad
"""

def data_clean(df):
    
    # duplicate columns
    df = df.T.drop_duplicates(keep='first').T
    
    # constant columns
    cols = df.columns.tolist()
    cols.remove([c for c in cols if df[c].nunique(dropna=False) == 1])
    df = df[cols]
     
    
