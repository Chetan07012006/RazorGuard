from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        "ieee-fraud-detection",
        "-p",
        str(RAW_DIR),
    ]
    subprocess.run(command, check=True)
    archive = RAW_DIR / "ieee-fraud-detection.zip"
    if archive.exists():
        import zipfile

        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(RAW_DIR)
        archive.unlink()
    required = [RAW_DIR / "train_transaction.csv", RAW_DIR / "train_identity.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset download completed but files are missing: {missing}")
    print(f"Downloaded IEEE-CIS files to {RAW_DIR}")


if __name__ == "__main__":
    main()
