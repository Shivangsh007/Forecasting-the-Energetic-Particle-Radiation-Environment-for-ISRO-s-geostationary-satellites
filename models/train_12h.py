"""Train the +12 h XGBoost regressor (+ persistence baseline). See train_common.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_common import train_one  # noqa: E402

if __name__ == "__main__":
    train_one("12h")
