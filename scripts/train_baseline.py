from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
FEATURES = [
    "TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "C1", "C2", "C5", "C13", "D1", "D2", "V1", "V2", "V3",
    "V4", "V10", "V12", "V14", "V17", "V20", "V30", "V33", "V34",
    "V40", "V44", "V45", "V53", "V70", "V87", "V91", "V100", "V127",
    "V130", "V156", "V170", "V187", "V204", "V221", "V257", "V274",
    "V283", "V289", "V294", "V307", "V310", "V312", "V315", "V317", "V320",
]


def main() -> None:
    path = RAW_DIR / "train_transaction.csv"
    columns = FEATURES + ["isFraud"]
    frame = pd.read_csv(path, usecols=columns).reset_index(drop=True)
    cutoff = int(len(frame) * 0.8)
    train, valid = frame.iloc[:cutoff], frame.iloc[cutoff:]
    x_train, y_train = train[FEATURES], train["isFraud"]
    x_valid, y_valid = valid[FEATURES], valid["isFraud"]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1),
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print(f"validation PR-AUC: {average_precision_score(y_valid, probabilities):.6f}")
    print("threshold: 0.500000")
    print(classification_report(y_valid, predictions, digits=6, zero_division=0))
    print("confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_valid, predictions))


if __name__ == "__main__":
    main()
