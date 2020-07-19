# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:40:31 2020

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
#studying co-relation between inputs
cols = ['base_price', 'total_price', 'diff', 'relative_diff_base', 'relative_diff_total'
        , 'is_featured_sku', 'is_display_sku', 'units_sold']
train[cols].corr().loc['units_sold']

cols.remove('units_sold')
print('current # of features to be used: {len(cols)}')

x= train[cols]
y= np.log(train['units_sold'])

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.2,random_state=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
reg = RandomForestRegressor()
model = reg.fit(x_train,y_train)

#making pred on xtest data
pred = reg.predict(x_val)
print("RMLSE IS ",1000*np.sqrt(mean_squared_log_error(np.exp(y_val),np.exp(pred))))

#submission
submit = pd.read_csv("D:\Data Science\\Projects\\Jnathack-Demand Forecasting\\sample_submission.csv")
sub_preds = reg.predict(test[cols])
submit['units_sold']= np.exp(sub_preds)
submit.to_csv("D:\Data Science\\Projects\\Jnathack-Demand Forecasting\\sample_submission.csv")
