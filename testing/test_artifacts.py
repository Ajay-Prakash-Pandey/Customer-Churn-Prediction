import json

from config import BEST_MODEL_PATH, FEATURE_COLUMNS_JSON_PATH
from src.utils import load_object

print("=" * 50)
print("Checking Artifacts")
print("=" * 50)

# Load feature columns
feature_columns = json.loads(
    FEATURE_COLUMNS_JSON_PATH.read_text(encoding="utf-8")
)

print("\nTotal Features:", len(feature_columns))

print("\nFeature Names:")

for feature in feature_columns:
    print(feature)

# Load model
model = load_object(BEST_MODEL_PATH)

print("\nModel Type:")

print(type(model))
