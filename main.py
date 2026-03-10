"""CLI entrypoint for the Fraud Detection System."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from fraud_detection.cli import prompt_transaction
from fraud_detection.database import TransactionDB
from fraud_detection.data_loader import load_transactions_csv
from fraud_detection.model import ModelResult, build_model_pipeline, load_model, save_model, train_model
from fraud_detection.preprocess import build_preprocessing_pipeline, prepare_feature_matrix
from fraud_detection.stats import dataset_statistics, print_stats
from fraud_detection.visualize import plot_amount_distribution, plot_class_balance


DEFAULT_MODEL_PATH = "models/fraud_model.joblib"
DEFAULT_DB_PATH = "db/transactions.db"


def train_command(args: argparse.Namespace) -> None:
    df = load_transactions_csv(args.data)

    print("Loaded dataset with", len(df), "rows")
    stats = dataset_statistics(df)
    print_stats(stats)

    X, y, cat_cols, num_cols = prepare_feature_matrix(df)
    preprocessor = build_preprocessing_pipeline(categorical_cols=cat_cols, numeric_cols=num_cols)

    pipeline = build_model_pipeline(preprocessor, model_type=args.model)
    model, results = train_model(pipeline, X, y)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    save_model(model, args.model_path)

    print("\nModel trained and saved to", args.model_path)
    print_metrics(results)

    if args.db:
        db = TransactionDB(args.db)
        # Store training data in DB (optional): mark predictions and label
        y_pred = model.predict(X)
        for row_idx in range(len(df)):
            record = {
                "transaction_id": str(df.iloc[row_idx].get("transaction_id", f"train-{row_idx}")),
                "amount": float(df.iloc[row_idx].get("amount", 0)),
                "transaction_type": df.iloc[row_idx].get("transaction_type", ""),
                "oldbalanceOrg": float(df.iloc[row_idx].get("oldbalanceOrg", 0)),
                "newbalanceOrig": float(df.iloc[row_idx].get("newbalanceOrig", 0)),
                "oldbalanceDest": float(df.iloc[row_idx].get("oldbalanceDest", 0)),
                "newbalanceDest": float(df.iloc[row_idx].get("newbalanceDest", 0)),
                "isFraud": int(df.iloc[row_idx].get("isFraud", 0)),
                "predicted": int(y_pred[row_idx]),
            }
            db.insert_transaction(record)

        print("Stored training records in database:", args.db)


def predict_command(args: argparse.Namespace) -> None:
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}. Please train first.")
        sys.exit(1)

    model = load_model(args.model_path)
    db = TransactionDB(args.db)

    while True:
        transaction = prompt_transaction()
        X = transaction.to_dataframe()
        prediction = int(model.predict(X)[0])

        print("\nPrediction result:")
        print("  -> Fraudulent" if prediction == 1 else "  -> Legitimate")

        record = {
            "transaction_id": transaction.transaction_id,
            "amount": transaction.amount,
            "transaction_type": transaction.transaction_type,
            "oldbalanceOrg": transaction.oldbalanceOrg,
            "newbalanceOrig": transaction.newbalanceOrig,
            "oldbalanceDest": transaction.oldbalanceDest,
            "newbalanceDest": transaction.newbalanceDest,
            "isFraud": 0,
            "predicted": prediction,
        }
        db.insert_transaction(record)

        again = input("\nPredict another transaction? (y/n): ").strip().lower()
        if again != "y":
            break


def stats_command(args: argparse.Namespace) -> None:
    db = TransactionDB(args.db)
    stats = db.get_statistics()

    print("\n=== Stored Transaction Statistics ===")
    print(f"Total stored transactions: {stats['total']}")
    print(f"Fraudulent (labeled): {stats['fraud_records']}")
    print(f"Predicted fraud: {stats['predicted_fraud']}")

    recent = db.list_recent(limit=10)
    if recent:
        print("\nRecent transactions (most recent first):")
        for item in recent:
            print(
                f"  id={item['transaction_id']} type={item['transaction_type']} amount={item['amount']} "
                f"predicted={item['predicted']} isFraud={item['isFraud']}"
            )


def visualize_command(args: argparse.Namespace) -> None:
    df = load_transactions_csv(args.data)
    plot_class_balance(df, output_path=args.output)
    plot_amount_distribution(df, output_path=args.output_amount)


def print_metrics(result: ModelResult) -> None:
    print("\n=== Model Evaluation ===")
    print(f"Accuracy : {result.accuracy:.3f}")
    print(f"Precision: {result.precision:.3f}")
    print(f"Recall   : {result.recall:.3f}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Financial Transaction Fraud Detection System"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a fraud detection model")
    train_parser.add_argument(
        "--data",
        type=str,
        default="data/transactions_sample.csv",
        help="Path to CSV dataset",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "tree"],
        help="Model type (logistic or tree)",
    )
    train_parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="SQLite database path for storing transactions",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict a single transaction")
    predict_parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to a trained model",
    )
    predict_parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="SQLite database path where predictions are stored",
    )

    stats_parser = subparsers.add_parser("stats", help="Show stored transaction statistics")
    stats_parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="SQLite database path",
    )

    vis_parser = subparsers.add_parser("visualize", help="Generate basic plots")
    vis_parser.add_argument(
        "--data",
        type=str,
        default="data/transactions_sample.csv",
        help="Path to CSV dataset",
    )
    vis_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for class balance plot (PNG). If omitted, shows interactively.",
    )
    vis_parser.add_argument(
        "--output-amount",
        type=str,
        default=None,
        help="Output path for amount distribution plot (PNG). If omitted, shows interactively.",
    )

    args = parser.parse_args(argv)

    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "stats":
        stats_command(args)
    elif args.command == "visualize":
        visualize_command(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
