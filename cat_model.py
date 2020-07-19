# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:16:47 2020

@author: dibya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("D:\\Data Science\\Projects\\Jnathack-Demand Forecasting\\train.csv",parse_dates = ['week'])
test = pd.read_csv("D:\\Data Science\\Projects\\Jnathack-Demand Forecasting\\test.csv",parse_dates = ['week'])

train.columns
test.columns
train.info()
test.info()

#combining store id and sku id
train['store_sku_id'] = (train['store_id'].astype('str')) + "_" + (train['sku_id'].astype('str'))
test['store_sku_id'] = (test['store_id'].astype('str') + "_" + test['sku_id'].astype('str'))

len(train['store_sku_id'].unique()) - len(test['store_sku_id'].unique())

#null value check and subtituing with base_price
train.isnull().sum()
x = train[train['total_price'].isnull()].index.tolist()
train['total_price'] = train['total_price'].fillna(train['base_price'].iloc[136949,])

#Appending train and test together for faster manipulation of data
test['units_sold'] = -1
data = train.append(test, ignore_index = True)

print('Checking Data distribution for Train! \n')
for col in train.columns:
    print(f'Distinct entries in {col}: {train[col].nunique()}')
    print(f'Common # of {col} entries in test and train: {len(np.intersect1d(train[col].unique(), test[col].unique()))}')
    
data.describe()

x = train.drop('units_sold',axis=1)
y= np.log(train['units_sold'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,random_state=1)

cat = CatBoostRegressor()
model2 = cat.fit(x_train,y_train)

pred_cat = model2.predict(x_val)
print("RMLSE IS ",1000*np.sqrt(mean_squared_log_error(np.exp(y_val),np.exp(pred_cat)))) ##461

#submission
submit_cat = pd.read_csv("D:\Data Science\\Projects\\Jnathack-Demand Forecasting\\sample_submission.csv")
sub_preds = model2.predict(test)
submit_cat['units_sold']= np.exp(sub_preds)
submit_cat.to_csv("D:\Data Science\\Projects\\Jnathack-Demand Forecasting\\Cat_submission.csv")











sns.distplot(train.units_sold)
sns.distplot(np.log(train.units_sold))

# Making price based new features
train['diff'] = train['base_price'] - train['total_price']
train['relative_diff_base'] = train['diff']/train['base_price']
train['relative_diff_total'] = train['diff']/train['total_price']

test['diff'] = test['base_price'] - test['total_price']
test['relative_diff_base'] = test['diff']/test['base_price']
test['relative_diff_total'] = test['diff']/test['total_price']

train.head(2)
train.info()