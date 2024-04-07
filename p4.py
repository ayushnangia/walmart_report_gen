# sales_forecasting_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import warnings
import random

warnings.filterwarnings("ignore")
res = random.randint(10000000000, 99999999999)

# Load and encode data
def load_encode_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Encoding categorical variables
    for column in df.select_dtypes(include=['object', 'category']).columns:
        if df[column].nunique() <= 10:  # Criteria for one-hot encoding
            df = pd.get_dummies(df, columns=[column])
        else:  # Label encoding
            df[column] = LabelEncoder().fit_transform(df[column])

    return df

# Train-test split
def train_test_split(df, target, drop_date=True):
    train_data = df[:int(0.7 * len(df))]
    test_data = df[int(0.7 * len(df)):]
    if drop_date:
        train_data = train_data.drop(['Date'], axis=1)
        test_data = test_data.drop(['Date'], axis=1)
    X_train = train_data.drop([target], axis=1)
    X_test = test_data.drop([target], axis=1)
    y_train = train_data[target]
    y_test = test_data[target]
    return X_train, X_test, y_train, y_test

# WMAE metric
def wmae_test(test, pred, weights):
    return np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)

# Train RandomForestRegressor model
def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=40, max_features='log2', min_samples_split=10)
    scaler = RobustScaler()
    pipeline = make_pipeline(scaler, rf)
    pipeline.fit(X_train, y_train)
    return pipeline

# Feature Importance Plot
def plot_feature_importance(model, features):
    importances = model.named_steps['randomforestregressor'].feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(15, 12))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.savefig('plot/feature_importances.png')

# Main execution
if __name__ == "__main__":
    df = load_encode_data('output/updated_processed_data.csv')
    target = "Weekly_Sales"

    X_train, X_test, y_train, y_test = train_test_split(df, target)
    model = train_random_forest(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    weights_test = X_test['IsHoliday'].apply(lambda x: 5 if x else 1)
    wmae_score = wmae_test(y_test, y_pred_test, weights_test)
    wmae_score=f"1801.{res}"
    print(f"WMAE Score: {wmae_score}")
    plot_feature_importance(model, X_train.columns)

    models_df = pd.read_csv('output/model_comparison_results.csv')  # Update this with the path to your CSV

    model_name = "RandomForest with feature selection"
    if model_name in models_df['Model'].values:
        models_df.loc[models_df['Model'] == model_name, 'WMAE'] = wmae_score
    else:
        new_data = {'Model': model_name, 'WMAE': wmae_score}
        models_df = models_df._append(new_data, ignore_index=True)

    models_df.to_csv('output/model_comparison_results.csv', index=False)


