from pathlib import Path
import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    transaction_path = RAW_DIR / "train_transaction.csv"
    identity_path = RAW_DIR / "train_identity.csv"
    if not transaction_path.exists() or not identity_path.exists():
        raise FileNotFoundError("Expected train_transaction.csv and train_identity.csv in data/raw.")

    labels = pd.read_csv(transaction_path, usecols=["TransactionID", "isFraud"])
    identity_ids = pd.read_csv(identity_path, usecols=["TransactionID"])
    print(f"transactions: {len(labels):,}")
    print(f"identity rows: {len(identity_ids):,}")
    print(f"fraud rows: {int(labels['isFraud'].sum()):,}")
    print(f"fraud rate: {labels['isFraud'].mean():.6f}")
    print(f"transactions with identity: {labels['TransactionID'].isin(identity_ids['TransactionID']).sum():,}")


if __name__ == "__main__":
    main()
