
# -*- coding: utf-8 -*-
import LoadData
import pandas as pd
import numpy as np
import EDA
# Creating csv file for testing
np.random.seed(0)
x1 = np.random.randn(100)
t = np.random.randint(0,2,100)
train = pd.DataFrame({"X1_1":x1, "X1_dup":x1,"Target":t})
train.to_csv('train.csv',index=False)

df = LoadData.load_data(file_path = 'train.csv')
EDA.print_details(df,target_feature="Target")
print(df)


