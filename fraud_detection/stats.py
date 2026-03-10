"""Utilities to compute and display dataset statistics."""

from __future__ import annotations

import pandas as pd


def dataset_statistics(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute basic statistics from a dataset."""
    total = len(df)
    fraud = int(df["isFraud"].sum())
    legit = total - fraud

    return {
        "total_transactions": total,
        "fraudulent": fraud,
        "legitimate": legit,
        "fraud_rate": float(fraud) / total if total else 0.0,
    }


def print_stats(stats: dict[str, float | int]) -> None:
    """Print statistics to the console."""
    print("\n=== Dataset Statistics ===")
    print(f"Total transactions: {stats['total_transactions']}")
    print(f"Fraudulent: {stats['fraudulent']}")
    print(f"Legitimate: {stats['legitimate']}")
    print(f"Fraud rate: {stats['fraud_rate']:.2%}\n")
