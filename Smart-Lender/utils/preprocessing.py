"""Preprocessing utilities for Smart Lender
Contains functions to clean and encode the loan dataset.
"""
import pandas as pd
from sklearn.impute import SimpleImputer


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    num_imp = SimpleImputer(strategy='median')
    if len(num_cols) > 0:
        df[num_cols] = num_imp.fit_transform(df[num_cols])
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_imp = SimpleImputer(strategy='most_frequent')
    if len(cat_cols) > 0:
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])
    df = pd.get_dummies(df, drop_first=True)
    return df
