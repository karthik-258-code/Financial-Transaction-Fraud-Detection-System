"""Simple visualizations for transaction data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_class_balance(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot the class distribution (fraud vs legitimate)."""
    counts = df["isFraud"].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=["#4CAF50", "#E53935"])
    plt.title("Transaction class balance")
    plt.ylabel("Number of transactions")
    plt.grid(axis="y", alpha=0.25)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved visualization: {output_path}")
    else:
        plt.show()


def plot_amount_distribution(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot distribution of transaction amounts."""
    plt.figure(figsize=(8, 4))
    plt.hist(df["amount"].fillna(0), bins=30, color="#1976D2", edgecolor="#FFFFFF")
    plt.title("Transaction amount distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved visualization: {output_path}")
    else:
        plt.show()
