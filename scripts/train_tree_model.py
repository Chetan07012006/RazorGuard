from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report
import joblib

from train_baseline import FEATURES


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    frame = pd.read_csv(RAW_DIR / "train_transaction.csv", usecols=FEATURES + ["isFraud"])
    cutoff = int(len(frame) * 0.8)
    train, valid = frame.iloc[:cutoff], frame.iloc[cutoff:]
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURES])
    x_valid = imputer.transform(valid[FEATURES])
    positive_weight = (train["isFraud"] == 0).sum() / (train["isFraud"] == 1).sum()
    weights = train["isFraud"].map({0: 1.0, 1: positive_weight})
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(x_train, train["isFraud"], sample_weight=weights)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"imputer": imputer, "model": model, "features": FEATURES}, MODEL_DIR / "tree_model.joblib")
    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.8).astype(int)
    print(f"validation PR-AUC: {average_precision_score(valid['isFraud'], probabilities):.6f}")
    print("threshold: 0.800000")
    print(classification_report(valid["isFraud"], predictions, digits=6, zero_division=0))
    print(f"saved model: {MODEL_DIR / 'tree_model.joblib'}")


if __name__ == "__main__":
    main()
