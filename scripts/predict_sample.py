from pathlib import Path
import joblib
import pandas as pd

from razorguard.policy import decide


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    bundle = joblib.load(PROJECT_ROOT / "models" / "tree_model.joblib")
    frame = pd.read_csv(RAW_DIR / "train_transaction.csv", nrows=1)
    features = bundle["features"]
    row = frame[features]
    score = float(bundle["model"].predict_proba(bundle["imputer"].transform(row))[:, 1][0])
    decision = decide(score)
    print(f"risk score: {decision.score:.6f}")
    print(f"risk level: {decision.level}")
    print(f"action: {decision.action}")


if __name__ == "__main__":
    main()
