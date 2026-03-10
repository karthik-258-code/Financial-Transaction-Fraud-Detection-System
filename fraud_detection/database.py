"""SQLite persistence for transaction records."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class TransactionDB:
    """Simple wrapper around a SQLite database for storing transaction predictions."""

    def __init__(self, db_path: str = "db/transactions.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT,
                    amount REAL,
                    transaction_type TEXT,
                    oldbalanceOrg REAL,
                    newbalanceOrig REAL,
                    oldbalanceDest REAL,
                    newbalanceDest REAL,
                    isFraud INTEGER,
                    predicted INTEGER,
                    predicted_at TEXT
                )"""
            )
            conn.commit()

    def insert_transaction(self, record: Dict[str, Any]) -> None:
        """Insert a single transaction record into the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO transactions (
                    transaction_id,
                    amount,
                    transaction_type,
                    oldbalanceOrg,
                    newbalanceOrig,
                    oldbalanceDest,
                    newbalanceDest,
                    isFraud,
                    predicted,
                    predicted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.get("transaction_id"),
                    record.get("amount"),
                    record.get("transaction_type"),
                    record.get("oldbalanceOrg"),
                    record.get("newbalanceOrig"),
                    record.get("oldbalanceDest"),
                    record.get("newbalanceDest"),
                    int(record.get("isFraud", 0)),
                    int(record.get("predicted", 0)),
                    record.get("predicted_at", datetime.utcnow().isoformat()),
                ),
            )
            conn.commit()

    def get_statistics(self) -> Dict[str, int]:
        """Return basic statistics from stored transactions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(1) FROM transactions")
            total = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(1) FROM transactions WHERE isFraud = 1")
            fraud = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(1) FROM transactions WHERE predicted = 1")
            predicted_fraud = cursor.fetchone()[0] or 0

        return {
            "total": total,
            "fraud_records": fraud,
            "predicted_fraud": predicted_fraud,
        }

    def list_recent(self, limit: int = 10) -> list[Dict[str, Any]]:
        """Return a list of most recent transaction records."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT transaction_id, amount, transaction_type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, predicted, predicted_at "
                "FROM transactions ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()

        columns = [
            "transaction_id",
            "amount",
            "transaction_type",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "predicted",
            "predicted_at",
        ]

        return [dict(zip(columns, row)) for row in rows]
