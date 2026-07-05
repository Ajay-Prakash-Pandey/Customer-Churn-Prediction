import json

from config import FEATURE_COLUMNS_JSON_PATH

feature_columns = json.loads(
    FEATURE_COLUMNS_JSON_PATH.read_text(encoding="utf-8")
)

print("\nNumber of Features:")

print(len(feature_columns))

print("\nFeature Names:")

for feature in feature_columns:
    print(feature)
