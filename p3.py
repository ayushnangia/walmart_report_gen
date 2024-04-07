# sales_forecasting_models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# WMAE metric
def wmae_test(test, pred, weights):
    return np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Train-test split
# Train-test split
def train_test_split(df, target):
    train_data = df[:int(0.7 * len(df))]
    test_data = df[int(0.7 * len(df)):]
    
    columns_to_drop = [target, 'Date']
    if 'month' in df.columns:
        columns_to_drop.append('month') # Add 'month' to drop list if it exists

    X_train = train_data.drop(columns_to_drop, axis=1)
    X_test = test_data.drop(columns_to_drop, axis=1)
    y_train = train_data[target]
    y_test = test_data[target]

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    pipeline = make_pipeline(RobustScaler(), model)
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    weights = X_test['IsHoliday'].apply(lambda x: 5 if x else 1)
    return wmae_test(y_test, predictions, weights)

if __name__ == "__main__":
    df = load_data('output/updated_processed_data.csv')
    target = "Weekly_Sales"
    X_train, X_test, y_train, y_test = train_test_split(df, target)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=40, max_features='log2'),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42)
    }

    results = []
    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train)
        wmae = evaluate_model(trained_model, X_test, y_test)
        results.append({'Model': name, 'WMAE': wmae})
        print(f"{name} - WMAE: {wmae}")

    df_results = pd.DataFrame(results)
    print(df_results)
    df_results.to_csv('output/model_comparison_results.csv')
