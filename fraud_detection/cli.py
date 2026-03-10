"""Command-line interface helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class TransactionInput:
    transaction_id: str
    amount: float
    transaction_type: str
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a single-row DataFrame for prediction."""
        return pd.DataFrame([asdict(self)])


def prompt_transaction() -> TransactionInput:
    """Prompt user for transaction fields."""
    print("Enter transaction details (leave blank to use defaults):")
    transaction_id = input("Transaction ID (e.g. 1001): ").strip() or "user-1"
    amount = float(input("Amount: ").strip() or "0")
    transaction_type = input("Transaction type (TRANSFER/PAYMENT): ").strip() or "TRANSFER"
    oldbalanceOrg = float(input("Origin balance before: ").strip() or "0")
    newbalanceOrig = float(input("Origin balance after: ").strip() or "0")
    oldbalanceDest = float(input("Destination balance before: ").strip() or "0")
    newbalanceDest = float(input("Destination balance after: ").strip() or "0")

    return TransactionInput(
        transaction_id=transaction_id,
        amount=amount,
        transaction_type=transaction_type,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
    )
