"""Preprocessing utilities for transaction data."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical features."""

    # Numeric pipeline: fill missing values with median and scale
    numeric_pipeline = (
        "numeric",
        Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        ),
        numeric_cols,
    )

    # Categorical pipeline: fill missing with a sentinel value and one-hot encode
    categorical_pipeline = (
        "categorical",
        Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        ),
        categorical_cols,
    )

    transformer = ColumnTransformer(
        transformers=[numeric_pipeline, categorical_pipeline],
        remainder="drop",
        sparse_threshold=0,
    )

    return transformer


def prepare_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare X, y feature matrix and list column names.

    Returns:
        X (DataFrame): features
        y (Series): target variable (isFraud)
        categorical_cols (list): categorical column names
        numeric_cols (list): numeric column names
    """

    df = df.copy()

    # Basic missing-value handling
    # - For numeric columns, filling missing values with median is done in the pipeline
    # - For categorical columns, missing values are treated as a separate category

    # Identify columns
    target_col = "isFraud"
    y = df[target_col].astype(int)

    # Candidate feature columns (exclude id and target)
    drop_cols = {"transaction_id", target_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Determine numeric vs categorical
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.to_list()
    categorical_cols = df[feature_cols].select_dtypes(exclude="number").columns.to_list()

    X = df[feature_cols].copy()

    return X, y, categorical_cols, numeric_cols
