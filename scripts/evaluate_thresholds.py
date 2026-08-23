from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_baseline import FEATURES


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    frame = pd.read_csv(RAW_DIR / "train_transaction.csv", usecols=FEATURES + ["isFraud"])
    cutoff = int(len(frame) * 0.8)
    train, valid = frame.iloc[:cutoff], frame.iloc[cutoff:]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=200, class_weight="balanced"),
    )
    model.fit(train[FEATURES], train["isFraud"])
    labels = valid["isFraud"]
    probabilities = model.predict_proba(valid[FEATURES])[:, 1]
    print(f"validation PR-AUC: {average_precision_score(labels, probabilities):.6f}")
    print("threshold,precision,recall,f1,flagged_rate")
    for threshold in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        predictions = (probabilities >= threshold).astype(int)
        print(
            f"{threshold:.2f},{precision_score(labels, predictions, zero_division=0):.6f},"
            f"{recall_score(labels, predictions, zero_division=0):.6f},"
            f"{f1_score(labels, predictions, zero_division=0):.6f},"
            f"{predictions.mean():.6f}"
        )


if __name__ == "__main__":
    main()
