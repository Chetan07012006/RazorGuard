from pathlib import Path
import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def summarize(path: Path) -> None:
    frame = pd.read_csv(path, nrows=1000)
    print(f"{path.name}: sample_rows={len(frame)}, columns={len(frame.columns)}")
    print(f"columns: {list(frame.columns)}")
    if "isFraud" in frame:
        print(f"sample fraud rate: {frame['isFraud'].mean():.6f}")


def main() -> None:
    for filename in ("train_transaction.csv", "train_identity.csv"):
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/download_data.py first.")
        summarize(path)


if __name__ == "__main__":
    main()
