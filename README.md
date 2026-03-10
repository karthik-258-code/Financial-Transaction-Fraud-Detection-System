# Financial Transaction Fraud Detection System

A beginner-to-intermediate Python project that analyzes financial transaction data and classifies transactions as **fraudulent** or **legitimate**.

## Features
- Load transaction data from a CSV file
- Basic preprocessing (missing values, encoding categorical features)
- Train a machine learning model (Logistic Regression or Decision Tree)
- Evaluate model using **accuracy**, **precision**, and **recall**
- CLI to input a transaction and predict whether it is fraud
- Store transaction records in SQLite
- Show basic statistics (total transactions, fraud counts, etc.)
- Basic visualizations using Matplotlib

## Project Structure
```
├── data/
│   └── transactions_sample.csv
├── db/
│   └── transactions.db          # created when running the app
├── models/
│   └── fraud_model.joblib       # created after training
├── fraud_detection/             # core library modules
│   ├── __init__.py
│   ├── cli.py
│   ├── data_loader.py
│   ├── database.py
│   ├── model.py
│   ├── preprocess.py
│   ├── stats.py
│   └── visualize.py
├── main.py                      # CLI entrypoint
├── requirements.txt
└── README.md
```

## Setup

1. Create & activate a Python virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1) Train the model

```bash
python main.py train --data data/transactions_sample.csv
```

### 2) Predict a single transaction (interactive)

```bash
python main.py predict
```

### 3) Show basic statistics

```bash
python main.py stats
```

### 4) Generate a simple visualization

```bash
python main.py visualize
```

---

## 🧠 Notes
- The sample dataset is small and synthetic, intended for learning.
- Use your own transaction CSV by following the same column format as in `data/transactions_sample.csv`.
- The SQLite database is stored in `db/transactions.db` and will be created automatically.
