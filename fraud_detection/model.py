"""Model training, saving, loading, and evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelResult:
    accuracy: float
    precision: float
    recall: float


def build_model_pipeline(
    preprocessor: ColumnTransformer, model_type: str = "logistic", random_state: int | None = 42
) -> Pipeline:
    """Build a full sklearn pipeline including preprocessing and a classifier."""
    model_type = model_type.lower()

    if model_type == "tree" or model_type == "decision_tree":
        clf = DecisionTreeClassifier(random_state=random_state)
    else:
        clf = LogisticRegression(random_state=random_state, max_iter=500)

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    return pipeline


def train_model(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[Pipeline, ModelResult]:
    """Train the pipeline and return metrics."""
    model = pipeline.fit(X, y)
    y_pred = model.predict(X)

    result = ModelResult(
        accuracy=accuracy_score(y, y_pred),
        precision=precision_score(y, y_pred, zero_division=0),
        recall=recall_score(y, y_pred, zero_division=0),
    )

    return model, result


def save_model(model: Pipeline, path: str) -> None:
    """Save trained pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    """Load a saved pipeline from disk."""
    return joblib.load(path)
