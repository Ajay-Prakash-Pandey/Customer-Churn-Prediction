import os
from pathlib import Path

# ===========================
# Base Directory
# ===========================

BASE_DIR = Path(__file__).resolve().parent

# ===========================
# Data
# ===========================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_PATH = DATA_DIR / "raw" / "customer.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ===========================
# Artifacts
# ===========================

ARTIFACTS_DIR = BASE_DIR / "artifacts"

BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"

MODEL_PATH = BEST_MODEL_PATH

FEATURE_COLUMNS_JSON_PATH = ARTIFACTS_DIR / "feature_columns.json"

FEATURE_COLUMNS_PATH = FEATURE_COLUMNS_JSON_PATH

LEGACY_MODEL_PATH = ARTIFACTS_DIR / "final_model.pkl"

LEGACY_SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

LEGACY_FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"

# ===========================
# Reports
# ===========================

REPORT_DIR = BASE_DIR / "reports"

# ===========================
# Logs
# ===========================

LOG_DIR = BASE_DIR / "logs"

# ===========================
# Flask
# ===========================

SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key")
