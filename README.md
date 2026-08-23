# RazorGuard

AI-assisted payment risk and fraud detection for the Razorpay AI Risk Manager challenge.

## Dataset

RazorGuard is designed for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset. The raw files are not committed because they are large and subject to Kaggle's competition terms.

1. Accept the competition rules on Kaggle.
2. Install the Kaggle CLI and authenticate it.
3. Open PowerShell and move into the project directory:

```powershell
cd "C:\Users\Dell\Documents\Codex\2026-08-23\referenced-chatgpt-conversation-this-is-an\outputs\RazorGuard"
```

4. From that directory, run:

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/inspect_data.py
python scripts/profile_data.py
python scripts/create_time_split.py
python scripts/train_baseline.py
python scripts/evaluate_thresholds.py
python scripts/train_identity_model.py
python scripts/train_tree_model.py
python scripts/predict_sample.py
```

The expected files are `data/raw/train_transaction.csv` and `data/raw/train_identity.csv`.

## Structure

```text
RazorGuard/
├── data/raw/          # Downloaded data; ignored by Git
├── notebooks/         # Exploratory analysis
├── scripts/           # Reproducible data utilities
├── src/razorguard/    # Application and modeling code
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

No evaluation metrics are reported until the dataset is downloaded and a held-out evaluation is run.

The first validation split is time-aware: the earliest 80% of labeled transactions are used for training and the latest 20% are held out for evaluation.

See [docs/evaluation.md](docs/evaluation.md) for the measured validation results.
