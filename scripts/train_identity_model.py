from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_baseline import FEATURES


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    transaction = pd.read_csv(RAW_DIR / "train_transaction.csv", usecols=FEATURES + ["TransactionID", "isFraud"])
    identity = pd.read_csv(RAW_DIR / "train_identity.csv", usecols=["TransactionID", "DeviceType", "DeviceInfo"])
    merged = transaction.merge(identity, on="TransactionID", how="left")
    merged["has_identity"] = merged["DeviceType"].notna().astype(int)
    merged["device_type_code"] = pd.factorize(merged["DeviceType"].fillna("missing"))[0]
    merged["device_info_length"] = merged["DeviceInfo"].fillna("").str.len()
    merged["device_info_present"] = merged["DeviceInfo"].notna().astype(int)
    model_features = FEATURES + ["has_identity", "device_type_code", "device_info_length", "device_info_present"]
    cutoff = int(len(merged) * 0.8)
    train, valid = merged.iloc[:cutoff], merged.iloc[cutoff:]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=200, class_weight="balanced"),
    )
    model.fit(train[model_features], train["isFraud"])
    probabilities = model.predict_proba(valid[model_features])[:, 1]
    predictions = (probabilities >= 0.8).astype(int)
    print(f"validation PR-AUC: {average_precision_score(valid['isFraud'], probabilities):.6f}")
    print("threshold: 0.800000")
    print(classification_report(valid["isFraud"], predictions, digits=6, zero_division=0))


if __name__ == "__main__":
    main()
