# sales_forecasting_part1.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Define output directory
output_dir = "output"

# Loading Data
df_store = pd.read_csv('walmart/walmart/stores.csv')
df_train = pd.read_csv('walmart/walmart/train.csv')
df_features = pd.read_csv('walmart/walmart/features.csv')

# Merging DataFrames
df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')

# Data Cleaning and Processing
df.drop(['IsHoliday_y'], axis=1, inplace=True)
df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)
df = df.loc[df['Weekly_Sales'] > 0]

# Converting 'Date' from string to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Processing Holiday Columns
df.loc[(df['Date'] == '2010-02-12') | (df['Date'] == '2011-02-11') | (df['Date'] == '2012-02-10'), 'Super_Bowl'] = True
df.loc[(df['Date'] != '2010-02-12') & (df['Date'] != '2011-02-11') & (df['Date'] != '2012-02-10'), 'Super_Bowl'] = False
df.loc[(df['Date'] == '2010-09-10') | (df['Date'] == '2011-09-09') | (df['Date'] == '2012-09-07'), 'Labor_Day'] = True
df.loc[(df['Date'] != '2010-09-10') & (df['Date'] != '2011-09-09') & (df['Date'] != '2012-09-07'), 'Labor_Day'] = False
df.loc[(df['Date'] == '2010-11-26') | (df['Date'] == '2011-11-25'), 'Thanksgiving'] = True
df.loc[(df['Date'] != '2010-11-26') & (df['Date'] != '2011-11-25'), 'Thanksgiving'] = False
df.loc[(df['Date'] == '2010-12-31') | (df['Date'] == '2011-12-30'), 'Christmas'] = True
df.loc[(df['Date'] != '2010-12-31') & (df['Date'] != '2011-12-30'), 'Christmas'] = False

# Filling missing values
df = df.fillna(0)

# Save the processed DataFrame
df.to_csv(f"{output_dir}/processed_data.csv")
