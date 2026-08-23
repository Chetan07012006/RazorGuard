from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    source = RAW_DIR / "train_transaction.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}")
    frame = pd.read_csv(source, usecols=["TransactionID", "TransactionDT", "isFraud"])
    frame = frame.sort_values("TransactionDT").reset_index(drop=True)
    cutoff = int(len(frame) * 0.8)
    train = frame.iloc[:cutoff]
    valid = frame.iloc[cutoff:]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train_ids.csv", index=False)
    valid.to_csv(PROCESSED_DIR / "valid_ids.csv", index=False)
    print(f"train rows: {len(train):,}; fraud rate: {train['isFraud'].mean():.6f}")
    print(f"valid rows: {len(valid):,}; fraud rate: {valid['isFraud'].mean():.6f}")
    print(f"cutoff TransactionDT: {train['TransactionDT'].max()}")


if __name__ == "__main__":
    main()
