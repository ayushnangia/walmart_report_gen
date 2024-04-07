# sales_forecasting_part2.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Define output directory
output_dir = "output"

# Load the processed data
df = pd.read_csv(f"{output_dir}/processed_data.csv", parse_dates=['Date'])
output_dir = "plot"
# Exploratory Data Analysis

# Plotting the number of stores and departments
plt.figure(figsize=(10, 6))
sns.countplot(x='Store', data=df)
plt.title('Count of Stores')
plt.savefig(f"{output_dir}/store_count.png")

plt.figure(figsize=(20, 6))
sns.countplot(x='Dept', data=df)
plt.title('Count of Departments')
plt.savefig(f"{output_dir}/dept_count.png")

# Sales Distribution
plt.figure(figsize=(10, 6))
sns.distplot(df['Weekly_Sales'])
plt.title('Weekly Sales Distribution')
plt.savefig(f"{output_dir}/weekly_sales_distribution.png")

# Store Type Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Type', data=df)
plt.title('Store Type Distribution')
plt.savefig(f"{output_dir}/store_type_distribution.png")

# Sales by Store Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Weekly_Sales', data=df)
plt.title('Weekly Sales by Store Type')
plt.savefig(f"{output_dir}/sales_by_store_type.png")

# Additional Data Processing

# Encoding categorical variables
df['Type'] = df['Type'].map({'A': 1, 'B': 2, 'C': 3})
df['IsHoliday'] = df['IsHoliday'].astype(int)

# Date Features
df['Week'] = df['Date'].dt.isocalendar().week
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
output_dir = "output"
# Saving the updated DataFrame
df.to_csv(f"{output_dir}/updated_processed_data.csv")
