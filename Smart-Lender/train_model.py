"""Train ML models and save the best one as model.pkl

This script trains Decision Tree, Random Forest, KNN and XGBoost
and picks the best model by test accuracy.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def load_data(path='dataset/loan_data.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def preprocess(df):
    # Basic preprocessing: fill na and simple encoding for demonstration
    df = df.copy()
    # Drop ID if present
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    # Fill numeric missing with median
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    num_imp = SimpleImputer(strategy='median')
    df[num_cols] = num_imp.fit_transform(df[num_cols])
    # Fill categorical with most frequent
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_imp = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])
    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)
    return df


def train_and_select(X_train, y_train, X_test, y_test):
    models = {
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {'model': model, 'accuracy': acc}

    # Pick best
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    return best_model, results


def main():
    df = load_data()
    # Assume target column is 'Loan_Status' or 'Target'
    if 'Loan_Status' in df.columns:
        target = 'Loan_Status'
    elif 'Target' in df.columns:
        target = 'Target'
    else:
        raise ValueError('Target column not found (expected Loan_Status or Target)')

    df_processed = preprocess(df)
    X = df_processed.drop(columns=[c for c in df_processed.columns if c.lower().startswith('loan_status') or c.lower().startswith('target')])
    # Find target column in processed df (after encoding)
    y = None
    for col in df_processed.columns:
        if col.lower().startswith('loan_status') or col.lower().startswith('target'):
            y = df_processed[col]
            break
    if y is None:
        raise ValueError('Processed target column not found')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_model, results = train_and_select(X_train, y_train, X_test, y_test)

    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save a human readable results file
    with open('models/training_results.txt', 'w') as f:
        for name, info in results.items():
            f.write(f"{name}: {info['accuracy']:.4f}\n")
        f.write(f"Best: {best_model.__class__.__name__}\n")

    print('Training complete. Best model saved to models/best_model.pkl')


if __name__ == '__main__':
    main()
