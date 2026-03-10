"""Data loading utilities."""

from __future__ import annotations

import pandas as pd


def load_transactions_csv(path: str) -> pd.DataFrame:
    """Load transactions from a CSV file.

    Expects a header row with at least the following columns:
    - transaction_id
    - amount
    - transaction_type
    - oldbalanceOrg
    - newbalanceOrig
    - oldbalanceDest
    - newbalanceDest
    - isFraud
    """

    df = pd.read_csv(path)

    # Ensure required columns exist
    required = {
        "transaction_id",
        "amount",
        "transaction_type",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df
